import type { AssistantConversation } from "./assistant-types";

export type Chapter = {
  code: string;
  title: string;
  subtitle: string;
  accent: string;
  accentSoft: string;
  secondary: string;
  glow: string;
};

export type AppRoute = "launch" | "system" | "trace" | "upload" | "assistant";

export type KnowledgeBaseRecord = {
  id: number;
  name: string;
  description: string;
  creator: string;
  createdAt: string;
};

export type DocumentStatus = "READY" | "INDEXING" | "CHUNKING" | "SYNCED" | "FAILED";
export type UploadStage = "idle" | "uploading" | "indexing" | "saving" | "done" | "error";

export type DocumentRecord = {
  id: number;
  fileName: string;
  knowledgeBaseId: number;
  type: "PDF" | "DOCX" | "PPTX" | "XLSX";
  size: string;
  chunkCount: number;
  status: DocumentStatus;
  uploadedAt: string;
  summary: string;
};

export type KnowledgeBasePayload = Pick<KnowledgeBaseRecord, "name" | "description"> & Partial<Pick<KnowledgeBaseRecord, "creator">>;
export type DocumentPayload = Pick<DocumentRecord, "fileName" | "knowledgeBaseId" | "type" | "size" | "chunkCount" | "status" | "summary">;
export type ManagementDeleteState = { type: "knowledgeBase"; id: number } | { type: "document"; id: number };

export type AssistantLaunchState = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type AssistantHistoryItem = {
  id: string;
  title: string;
  question: string;
  createdAt: number;
  preview: string;
  conversation?: AssistantConversation;
};

export const routePathMap: Record<AppRoute, string> = {
  launch: "/",
  system: "/system",
  trace: "/trace",
  upload: "/upload",
  assistant: "/aviation-assistant",
};

export const chapterRoutes: AppRoute[] = ["system", "trace", "upload", "assistant"];

export function normalizePathname(pathname: string) {
  const normalized = pathname.replace(/\/+$/, "");
  return normalized === "" ? "/" : normalized;
}

export function resolveRoute(pathname: string): AppRoute {
  switch (normalizePathname(pathname)) {
    case "/": return "launch";
    case "/system":
    case "/qa": return "system";
    case "/trace": return "trace";
    case "/upload":
    case "/intake": return "upload";
    case "/aviation-assistant":
    case "/assistant": return "assistant";
    default: return "launch";
  }
}

export function getChapterIndexForRoute(route: AppRoute) {
  return route === "system" ? 0 : route === "trace" ? 1 : route === "upload" ? 2 : route === "assistant" ? 3 : 0;
}

export const introTheme: Chapter = {
  code: "00", title: "Launch Deck", subtitle: "Aerospace Knowledge Command Core", accent: "#ff5e7e",
  accentSoft: "rgba(255, 94, 126, 0.14)", secondary: "#ffcb5b", glow: "radial-gradient(circle at center, rgba(255,94,126,0.30), transparent 68%)",
};

export const chapters: Chapter[] = [
  { code: "01", title: "Grounded QA", subtitle: "Primary Question Answering Workspace", accent: "#ff78c8", accentSoft: "rgba(255, 120, 200, 0.15)", secondary: "#d9ff52", glow: "radial-gradient(circle at center, rgba(255,120,200,0.24), transparent 68%)" },
  { code: "02", title: "Retrieval Trace", subtitle: "Backend Hybrid Retrieval Observation", accent: "#d6ff3f", accentSoft: "rgba(214, 255, 63, 0.15)", secondary: "#ff7a8d", glow: "radial-gradient(circle at center, rgba(214,255,63,0.24), transparent 68%)" },
  { code: "03", title: "知识库和文档管理", subtitle: "Knowledge Base and Document Management", accent: "#ff9a3c", accentSoft: "rgba(255, 154, 60, 0.16)", secondary: "#d8ff58", glow: "radial-gradient(circle at center, rgba(255,154,60,0.28), transparent 68%)" },
  { code: "04", title: "智能问答检索", subtitle: "航空文档智能问答检索", accent: "#7fe0ff", accentSoft: "rgba(127, 224, 255, 0.14)", secondary: "#ffcb5b", glow: "radial-gradient(circle at center, rgba(127,224,255,0.26), transparent 68%)" },
];
