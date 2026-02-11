import { createHash } from "crypto";
import { getRaycastHeaders, RAYCAST_MODELS_URL, type Env } from "./auth";
import { Logger } from "./logger";
import type { ModelInfo, RaycastMessage, RaycastChatRequest, RaycastSSEData, OpenAIMessage, RaycastFileUploadRequest, RaycastFileUploadResponse } from "./types";


const MODELS_CACHE_KEY = "system:models_cache";
const MODELS_CACHE_TTL = 86400;
const DEVICE_ID_CACHE_KEY = "system:device_id";
let cachedDeviceId: string | null = null;

function generateDeviceId(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function getDeviceId(env: Env): Promise<string> {
  if (env.DEVICE_ID) return env.DEVICE_ID;
  if (cachedDeviceId) return cachedDeviceId;
  if (env.R2A_ATTACHMENTS) {
    try {
      const stored = await env.R2A_ATTACHMENTS.get(DEVICE_ID_CACHE_KEY);
      if (stored) {
        cachedDeviceId = stored;
        return stored;
      }
    } catch {
      // ignore
    }
  }
  cachedDeviceId = generateDeviceId();
  if (env.R2A_ATTACHMENTS) {
    try {
      await env.R2A_ATTACHMENTS.put(DEVICE_ID_CACHE_KEY, cachedDeviceId);
    } catch {
      // ignore
    }
  }
  return cachedDeviceId;
}

async function rotateDeviceId(env: Env): Promise<string> {
  const next = generateDeviceId();
  cachedDeviceId = next;
  if (env.R2A_ATTACHMENTS) {
    try {
      await env.R2A_ATTACHMENTS.put(DEVICE_ID_CACHE_KEY, next);
    } catch {
      // ignore
    }
  }
  return next;
}

async function fetchRaycastWithRetry(
  env: Env,
  url: string,
  payload: string,
  init: RequestInit,
  extraHeaders?: Record<string, string>,
): Promise<Response> {
  const deviceId = await getDeviceId(env);
  let headers = { ...getRaycastHeaders(env, payload, deviceId), ...(extraHeaders || {}) };
  let response = await fetch(url, { ...init, headers });

  if (!env.DEVICE_ID && (response.status === 403 || response.status === 429)) {
    const rotated = await rotateDeviceId(env);
    headers = { ...getRaycastHeaders(env, payload, rotated), ...(extraHeaders || {}) };
    response = await fetch(url, { ...init, headers });
  }

  return response;
}

export async function fetchModels(env: Env): Promise<Map<string, ModelInfo>> {
  const logger = new Logger(env);

  if (env.R2A_ATTACHMENTS) {
    try {
      const cached = await env.R2A_ATTACHMENTS.get(MODELS_CACHE_KEY);
      if (cached) {
        logger.debug("Using cached models from KV");
        const cachedModels = JSON.parse(cached) as ModelInfo[];
        const modelsMap = new Map<string, ModelInfo>();
        for (const m of cachedModels) {
          modelsMap.set(m.id, m);
        }
        return modelsMap;
      }
    } catch (err) {
      logger.error("Failed to read models cache from KV:", err);
    }
  }

  logger.debug("Fetching models from Raycast...");
  const response = await fetch(RAYCAST_MODELS_URL, {
    method: "GET",
    headers: {},
  });

  if (!response.ok) {
    logger.warn(`Public models fetch failed (status ${response.status}), retrying with auth...`);
    const authResponse = await fetchRaycastWithRetry(env, RAYCAST_MODELS_URL, "", {
      method: "GET",
    });
    if (!authResponse.ok) {
      logger.error(`Raycast API error fetching models: ${authResponse.status}`);
      throw new Error(`Raycast API error: ${authResponse.status}`);
    }
    return processModelsResponse(await authResponse.json(), env);
  }

  return processModelsResponse(await response.json(), env);
}

async function processModelsResponse(parsedResponse: any, env: Env): Promise<Map<string, ModelInfo>> {
  const logger = new Logger(env);

  if (!parsedResponse?.models) {
    logger.error("Invalid response structure from Raycast API", { response: JSON.stringify(parsedResponse).slice(0, 500) });
    throw new Error("Invalid response structure from Raycast API");
  }

  const models = new Map<string, ModelInfo>();
  const modelsListForCache: ModelInfo[] = [];

  for (const modelData of parsedResponse.models) {
    const info: ModelInfo = {
      id: modelData.id,
      provider: modelData.provider,
      model: modelData.model,
      name: modelData.name,
      requires_better_ai: modelData.requires_better_ai,
      pro_plan_replacement_model_id: modelData.pro_plan_replacement_model_id ?? null,
      capabilities: modelData.abilities || {},
      context_window: modelData.context_window,
    };
    models.set(modelData.id, info);
    modelsListForCache.push(info);
  }

  if (env.R2A_ATTACHMENTS) {
    try {
      await env.R2A_ATTACHMENTS.put(MODELS_CACHE_KEY, JSON.stringify(modelsListForCache), {
        expirationTtl: MODELS_CACHE_TTL,
      });
      logger.debug("Models cache updated in KV");
    } catch (err) {
      logger.error("Failed to save models cache to KV:", err);
    }
  }

  return models;
}


export async function createStreamingChatCompletion(
  requestPayload: RaycastChatRequest,
  env: Env,
): Promise<ReadableStream> {
  const logger = new Logger(env);
  const payloadString = JSON.stringify(requestPayload);

  logger.debug("Sending streaming request to Raycast...");

  const response = await fetchRaycastWithRetry(env, "https://backend.raycast.com/api/v1/ai/chat_completions", payloadString, {
    method: "POST",
    body: payloadString,
  });

  if (!response.ok) {
    const errorText = await response.text();
    logger.error(`Raycast API error: ${response.status} ${errorText}`);
    throw new Error(`Raycast API error: ${response.status} - ${errorText}`);
  }

  if (!response.body) {
    logger.error("Raycast response body is empty");
    throw new Error("Stream completion response body is missing");
  }

  return new ReadableStream({
    async start(controller) {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let isClosed = false;

      const safeClose = () => {
        if (!isClosed) {
          isClosed = true;
          try {
            controller.close();
          } catch {
          }
        }
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const data = line.substring(6).trim();
                if (data === "[DONE]") {
                  safeClose();
                  break;
                }
                try {
                  const parsed = JSON.parse(data);
                  controller.enqueue(parsed);
                } catch {
                }
              }
            }
            if (isClosed) break;
          }
        }
        if (!isClosed && buffer.trim()) {
          const line = buffer.trim();
          if (line.startsWith("data: ")) {
            const data = line.substring(6).trim();
            if (data !== "[DONE]") {
              try {
                controller.enqueue(JSON.parse(data));
              } catch { }
            }
          }
        }
      } catch (error) {
        if (!isClosed) {
          controller.error(error);
          isClosed = true;
        }
      } finally {
        safeClose();
        reader.releaseLock();
      }
    },
  });
}

export function parseSSEResponse(responseText: string, logger?: Logger): { text: string; reasoning: string; tool_calls: any[]; model_update?: string; images: string[] } {
  let text = "";
  let reasoning = "";
  const toolCallsMap = new Map<string, any>();
  let modelUpdate: string | undefined;
  const images: string[] = [];

  for (const line of responseText.split("\n")) {
    if (line.startsWith("data:")) {
      try {
        const data = line.substring(5).trim();
        if (data === "[DONE]") break;
        const jsonData: RaycastSSEData = JSON.parse(data);
        if (jsonData.text) text += jsonData.text;
        if (jsonData.reasoning) reasoning += jsonData.reasoning;
        if (jsonData.notification_type === "model_updated") {
          const notification = (jsonData as any).notification || (jsonData as any).content?.notification;
          modelUpdate = notification ? String(notification) : "model_updated";
        }
        const directImage = (jsonData as any).image || (jsonData as any).content?.image;
        if (typeof directImage === "string" && directImage.length > 0) {
          images.push(directImage);
        }
        const imageList = (jsonData as any).images || (jsonData as any).content?.images;
        if (Array.isArray(imageList)) {
          for (const img of imageList) {
            if (typeof img === "string" && img.length > 0) images.push(img);
          }
        }

        const chunks = jsonData.tool_calls || (jsonData.tool_call ? [jsonData.tool_call] : []);
        for (const tc of chunks) {
          const id = tc.id || tc.call_id;
          if (id) {
            const existing = toolCallsMap.get(id);
            if (tc.name || !existing) {
              toolCallsMap.set(id, tc);
            }
          }
        }
      } catch (e) {
        if (logger) logger.error("Failed to parse SSE line", { line: line.slice(0, 200), error: String(e) });
      }
    }
  }
  return { text, reasoning, tool_calls: Array.from(toolCallsMap.values()), model_update: modelUpdate, images };
}

export async function createNonStreamingChatCompletion(
  requestPayload: RaycastChatRequest,
  env: Env,
): Promise<{ text: string; reasoning: string; tool_calls: any[]; model_update?: string; images: string[] }> {
  const logger = new Logger(env);
  const payloadString = JSON.stringify(requestPayload);

  logger.debug("Sending non-streaming request to Raycast...");

  const response = await fetchRaycastWithRetry(env, "https://backend.raycast.com/api/v1/ai/chat_completions", payloadString, {
    method: "POST",
    body: payloadString,
  });

  if (!response.ok) {
    const errorText = await response.text();
    logger.error(`Raycast API error: ${response.status} ${errorText}`);
    throw new Error(`Raycast API error: ${response.status} - ${errorText}`);
  }

  const responseText = await response.text();
  return parseSSEResponse(responseText, logger);
}

const imageCache = new Map<string, string>();

async function uploadImage(env: Env, imageUrl: string, threadId: string): Promise<string> {
  const logger = new Logger(env);
  let contentType = "image/png";
  let buffer: Buffer;

  logger.debug(`Uploading image: ${imageUrl.startsWith("data:") ? "base64 data" : imageUrl}`);

  if (imageUrl.startsWith("data:")) {
    const matches = imageUrl.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
    if (matches) {
      contentType = matches[1];
      buffer = Buffer.from(matches[2], "base64");
    } else {
      logger.error("Invalid base64 image data");
      throw new Error("Invalid base64 image data");
    }
  } else {
    const response = await fetch(imageUrl);
    if (!response.ok) {
      logger.error(`Failed to fetch image from URL: ${response.status}`);
      throw new Error(`Failed to fetch image from URL: ${response.status}`);
    }
    contentType = response.headers.get("Content-Type") || "image/png";
    buffer = Buffer.from(await response.arrayBuffer());
  }

  const hash = createHash("sha256").update(buffer).digest("hex");
  logger.debug(`Image hash: ${hash}`);

  if (imageCache.has(hash)) {
    logger.debug("Image hash found in memory cache");
    return imageCache.get(hash)!;
  }

  if (env.R2A_ATTACHMENTS) {
    try {
      const cachedId = await env.R2A_ATTACHMENTS.get(`img:${hash}`);

      if (cachedId) {
        logger.debug("Image hash found in KV cache");
        imageCache.set(hash, cachedId);
        return cachedId;
      }
    } catch (err) {
      logger.error("KV get error:", err);
    }
  }

  const checksum = createHash("md5").update(buffer).digest("base64");

  const filename = `image_${Date.now()}.${contentType.split("/")[1] || "png"}`;

  const uploadPayload: RaycastFileUploadRequest = {
    chat_id: threadId,
    blob: {
      filename,
      byte_size: buffer.length,
      checksum,
      content_type: contentType,
    },
  };

  logger.debug(`Registering file with Raycast: ${filename} (${buffer.length} bytes)`);
  const payloadString = JSON.stringify(uploadPayload);
  if (env.IMAGE_TOKEN) {
    logger.debug("Using IMAGE_TOKEN for file upload");
  }

  const response = await fetchRaycastWithRetry(env, "https://backend.raycast.com/api/v1/ai/files", payloadString, {
    method: "POST",
    body: payloadString,
  }, env.IMAGE_TOKEN ? { Authorization: `Bearer ${env.IMAGE_TOKEN}` } : undefined);

  if (!response.ok) {
    const errorText = await response.text();
    logger.error(`Raycast file upload preparation failed: ${response.status} ${errorText}`);
    throw new Error(`Raycast file upload preparation failed: ${response.status} ${errorText}`);
  }

  const uploadInfo = (await response.json()) as RaycastFileUploadResponse;
  logger.debug(`Raycast file registered, ID: ${uploadInfo.id}. Uploading to S3...`);

  const s3Response = await fetch(uploadInfo.direct_upload.url, {
    method: "PUT",
    headers: uploadInfo.direct_upload.headers,
    body: new Uint8Array(buffer),
  });


  if (!s3Response.ok) {
    const s3Error = await s3Response.text();
    logger.error(`S3 upload failed: ${s3Response.status} ${s3Error}`);
    throw new Error(`S3 upload failed: ${s3Response.status}`);
  }

  logger.debug("S3 upload successful");
  imageCache.set(hash, uploadInfo.id);

  if (env.R2A_ATTACHMENTS) {
    try {
      await env.R2A_ATTACHMENTS.put(`img:${hash}`, uploadInfo.id);
      logger.debug("Image ID persisted to KV");
    } catch (err) {
      logger.error("KV put error:", err);
    }
  }

  return uploadInfo.id;
}


export async function convertMessages(
  openaiMessages: OpenAIMessage[],
  env: Env,
  threadId: string,
): Promise<{
  raycastMessages: RaycastMessage[];
  systemInstruction: string;
}> {
  const logger = new Logger(env);
  logger.debug(`Converting ${openaiMessages.length} messages...`);
  let systemInstruction = "markdown";
  const raycastMessages: RaycastMessage[] = [];

  for (let i = 0; i < openaiMessages.length; i++) {
    const msg = openaiMessages[i];
    if (msg.role === "system" && i === 0) {
      systemInstruction = typeof msg.content === "string" ? msg.content : "markdown";
    } else if (msg.role === "user") {
      let text = "";
      const attachments: { id: string; type: "file" }[] = [];

      if (typeof msg.content === "string") {
        text = msg.content;
      } else if (Array.isArray(msg.content)) {
        for (const item of msg.content) {
          if (item.type === "text") {
            text += (text ? "\n" : "") + item.text;
          } else if (item.type === "image_url") {
            try {
              const attachmentId = await uploadImage(env, item.image_url.url, threadId);
              attachments.push({ id: attachmentId, type: "file" });
            } catch (error) {
              logger.error("Failed to upload image", { error: String(error), imageUrl: item.image_url.url });
            }
          }
        }
      }

      raycastMessages.push({
        author: "user",
        content: { text, attachments: attachments.length > 0 ? attachments : undefined },
      });
    } else if (msg.role === "assistant") {
      const content: any = {};
      if (typeof msg.content === "string") {
        content.text = msg.content;
      }
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        content.tool_call = msg.tool_calls[0];
      }
      raycastMessages.push({
        author: "assistant",
        content,
      });
    } else if (msg.role === "tool") {
      raycastMessages.push({
        author: "user",
        content: {
          tool_result: {
            call_id: msg.tool_call_id,
            result: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
          },
        },
      });
    }
  }

  return { raycastMessages, systemInstruction };
}
