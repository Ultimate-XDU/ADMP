import type { DocumentRecord, KnowledgeBaseRecord } from "./app-model";

export const apiBase = (
  import.meta.env.VITE_MANAGEMENT_API_BASE?.trim() || "/api"
).replace(/\/$/, "");

export type Evidence = {
  document_title?: string;
  source_url?: string;
  locator?: { label?: string; [key: string]: unknown };
  content: string;
  doc_name: string;
  heading: string;
  parent_heading: string;
  score: number;
  chunk_id: string;
  page_number: number;
  page_count: number;
  section_title: string;
  highlight_text: string;
  source_type: string;
  chunk_role?: string;
  article_label?: string;
  clause_path?: string;
  rerank_score?: number;
  channel_scores?: Record<string, number>;
  bbox?: { x: number; y: number; width: number; height: number };
  bbox_pages?: {
    page_number: number;
    bbox: { x: number; y: number; width: number; height: number };
  }[];
};

export type Citation = { id: number; sourceIndex: number; chunkId: string };
export type RetrievalTrace = {
  mode: string;
  planner?: string;
  reason?: string;
  steps: {
    hop: number;
    query: string;
    resultCount: number;
    basisChunkIds?: string[];
  }[];
};

export type QAResult = {
  answer: string;
  sources: Evidence[];
  confidence: number;
  abstainReason: string;
  evidenceIds: string[];
  queryType: string;
  conversationId: string;
  retrievalTrace?: RetrievalTrace;
  citations?: Citation[];
  visualContext?: {
    used: boolean;
    pages: { sourceIndex: number; page: number }[];
    reason: string;
  };
};

export type ChatSummary = {
  id: string;
  title: string;
  question: string;
  preview: string;
  createdAt: number;
  updatedAt: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  payload?: QAResult | null;
};

export type ChatDetail = ChatSummary & { messages: ChatMessage[] };
export type Health = {
  status: string;
  details: Record<string, boolean | string>;
};
export type Document = DocumentRecord & { storageFileName?: string };
export type Library = { bases: KnowledgeBaseRecord[]; documents: Document[] };

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const text = await response.text();
    let message = `请求失败（${response.status}）`;
    try {
      const data = JSON.parse(text) as { detail?: unknown };
      if (typeof data.detail === "string") message = data.detail;
    } catch {
      if (text && !text.includes("<html")) message = text.slice(0, 240);
    }
    throw new Error(message);
  }
  return response.status === 204
    ? (undefined as T)
    : ((await response.json()) as T);
}

export async function loadLibrary(): Promise<Library> {
  const [bases, documents] = await Promise.all([
    request<KnowledgeBaseRecord[]>("/knowledge-bases"),
    request<Document[]>("/documents"),
  ]);
  return { bases, documents };
}

export function pageImageUrl(name: string, page: number) {
  return `${apiBase}/evidence/page-image?${new URLSearchParams({ doc_name: name, page: String(page), dpi: "120" })}`;
}

export function errorMessage(error: unknown) {
  return error instanceof Error
    ? error.message
    : "服务暂时不可用，请稍后重试。";
}

export async function importDocument(
  file: File,
  knowledgeBaseId: number,
  onStage: (stage: string) => void,
) {
  const format = file.name.split(".").pop()!.toUpperCase();
  onStage("正在上传文件");
  const form = new FormData();
  form.append("file", file);
  const upload = await request<{ fileName: string; size: string }>("/upload", {
    method: "POST",
    body: form,
  });
  onStage("正在保存文档记录");
  const document = await request<Document>("/documents", {
    method: "POST",
    body: JSON.stringify({
      fileName: upload.fileName,
      knowledgeBaseId,
      type: format === "HTM" ? "HTML" : format,
      size: upload.size,
      status: "INDEXING",
      summary: "",
    }),
  });
  onStage("正在解析文档并构建检索索引，首次加载模型可能需要几分钟");
  const index = await request<{
    success: boolean;
    chunkCount: number;
    message: string;
  }>("/index", {
    method: "POST",
    body: JSON.stringify({ fileName: upload.fileName }),
  });
  if (!index.success) throw new Error(index.message);
  return { ...document, chunkCount: index.chunkCount, status: "SYNCED" };
}

export function evidenceLocationLabel(source: Evidence) {
  return (
    source.locator?.label ||
    (source.page_number > 0
      ? `第 ${source.page_number} 页`
      : source.section_title || source.heading || "原文片段")
  );
}
