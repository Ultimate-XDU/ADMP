import { CSSProperties, ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  AnimatePresence,
  motion,
  useMotionTemplate,
  useMotionValue,
  useSpring,
  type MotionProps,
  type MotionValue,
  type TargetAndTransition,
  type Transition,
  type Variants,
} from "framer-motion";
import * as THREE from "three";
import {
  ArrowRight,
  Binary,
  Blocks,
  Bot,
  BrainCircuit,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Database,
  Eye,
  FileDigit,
  FileSearch,
  FileText,
  FolderOpen,
  Pencil,
  Plus,
  Radar,
  Search,
  ShieldCheck,
  Trash2,
  UploadCloud,
  Wrench,
  X,
} from "lucide-react";
import {
  cloneAssistantConversation,
  resolveAssistantConversation,
  type AssistantCitation,
  type AssistantConversation,
  type AssistantDocumentType,
  type AssistantMessage,
  type AssistantMessageSegment,
  type AssistantSource,
} from "./aviationAssistantMock";

type Chapter = {
  code: string;
  title: string;
  subtitle: string;
  accent: string;
  accentSoft: string;
  secondary: string;
  glow: string;
};

type AppRoute = "launch" | "system" | "trace" | "upload" | "assistant";

type KnowledgeBaseRecord = {
  id: number;
  name: string;
  description: string;
  creator: string;
  createdAt: string;
};

type DocumentStatus = "READY" | "INDEXING" | "CHUNKING" | "SYNCED" | "FAILED";

type UploadStage = "idle" | "uploading" | "indexing" | "saving" | "done" | "error";

type DocumentRecord = {
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

type KnowledgeBasePayload = Pick<KnowledgeBaseRecord, "name" | "description"> & Partial<Pick<KnowledgeBaseRecord, "creator">>;

type DocumentPayload = Pick<DocumentRecord, "fileName" | "knowledgeBaseId" | "type" | "size" | "chunkCount" | "status" | "summary">;

type ManagementDeleteState =
  | { type: "knowledgeBase"; id: number }
  | { type: "document"; id: number };

type AssistantLaunchState = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type AssistantHistoryItem = {
  id: string;
  title: string;
  question: string;
  createdAt: number;
  preview: string;
  conversation: AssistantConversation;
};

const routePathMap: Record<AppRoute, string> = {
  launch: "/",
  system: "/system",
  trace: "/trace",
  upload: "/upload",
  assistant: "/aviation-assistant",
};

const chapterRoutes: AppRoute[] = ["system", "trace", "upload", "assistant"];

function normalizePathname(pathname: string) {
  const normalized = pathname.replace(/\/+$/, "");
  return normalized === "" ? "/" : normalized;
}

function resolveRoute(pathname: string): AppRoute {
  switch (normalizePathname(pathname)) {
    case "/":
      return "launch";
    case "/system":
    case "/qa":
      return "system";
    case "/trace":
      return "trace";
    case "/upload":
    case "/intake":
      return "upload";
    case "/aviation-assistant":
    case "/assistant":
      return "assistant";
    default:
      return "launch";
  }
}

function getChapterIndexForRoute(route: AppRoute) {
  switch (route) {
    case "system":
      return 0;
    case "trace":
      return 1;
    case "upload":
      return 2;
    case "assistant":
      return 3;
    default:
      return 0;
  }
}

const introTheme: Chapter = {
  code: "00",
  title: "Launch Deck",
  subtitle: "Aerospace Knowledge Command Core",
  accent: "#ff5e7e",
  accentSoft: "rgba(255, 94, 126, 0.14)",
  secondary: "#ffcb5b",
  glow: "radial-gradient(circle at center, rgba(255,94,126,0.30), transparent 68%)",
};

const chapters: Chapter[] = [
  {
    code: "01",
    title: "Grounded QA",
    subtitle: "Primary Question Answering Workspace",
    accent: "#ff78c8",
    accentSoft: "rgba(255, 120, 200, 0.15)",
    secondary: "#d9ff52",
    glow: "radial-gradient(circle at center, rgba(255,120,200,0.24), transparent 68%)",
  },
  {
    code: "02",
    title: "Retrieval Trace",
    subtitle: "Backend Hybrid Retrieval Observation",
    accent: "#d6ff3f",
    accentSoft: "rgba(214, 255, 63, 0.15)",
    secondary: "#ff7a8d",
    glow: "radial-gradient(circle at center, rgba(214,255,63,0.24), transparent 68%)",
  },
  {
    code: "03",
    title: "知识库和文档管理",
    subtitle: "Knowledge Base and Document Management",
    accent: "#ff9a3c",
    accentSoft: "rgba(255, 154, 60, 0.16)",
    secondary: "#d8ff58",
    glow: "radial-gradient(circle at center, rgba(255,154,60,0.28), transparent 68%)",
  },
  {
    code: "04",
    title: "智能问答检索",
    subtitle: "航空文档智能问答检索",
    accent: "#7fe0ff",
    accentSoft: "rgba(127, 224, 255, 0.14)",
    secondary: "#ffcb5b",
    glow: "radial-gradient(circle at center, rgba(127,224,255,0.26), transparent 68%)",
  },
];

const pageVariants: Variants[] = [
  {
    initial: { opacity: 0, x: 60, filter: "blur(10px)" },
    animate: { opacity: 1, x: 0, filter: "blur(0px)", transition: { duration: 0.44, ease: [0.16, 1, 0.3, 1] } },
    exit: { opacity: 0, x: -30, filter: "blur(6px)", transition: { duration: 0.2 } },
  },
  {
    initial: { opacity: 0, scale: 0.92, filter: "blur(10px)" },
    animate: { opacity: 1, scale: 1, filter: "blur(0px)", transition: { duration: 0.42, ease: [0.16, 1, 0.3, 1] } },
    exit: { opacity: 0, scale: 1.03, filter: "blur(6px)", transition: { duration: 0.18 } },
  },
  {
    initial: { opacity: 0, y: 36, filter: "blur(12px)" },
    animate: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.44, ease: [0.19, 1, 0.22, 1] } },
    exit: { opacity: 0, y: -16, filter: "blur(6px)", transition: { duration: 0.2 } },
  },
  {
    initial: { opacity: 0, y: 28, filter: "blur(10px)" },
    animate: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
    exit: { opacity: 0, y: -12, filter: "blur(4px)", transition: { duration: 0.18 } },
  },
  {
    initial: { opacity: 0, scale: 0.95, filter: "blur(8px)" },
    animate: { opacity: 1, scale: 1, filter: "blur(0px)", transition: { duration: 0.42, ease: [0.19, 1, 0.22, 1] } },
    exit: { opacity: 0, scale: 1.02, filter: "blur(4px)", transition: { duration: 0.18 } },
  },
];

const headerMorphTransition: Transition = {
  type: "spring",
  stiffness: 175,
  damping: 24,
  mass: 0.9,
};

const headerContentVariants: Variants = {
  initial: { opacity: 0, x: 18, filter: "blur(6px)" },
  animate: {
    opacity: 1,
    x: 0,
    filter: "blur(0px)",
    transition: { duration: 0.44, ease: [0.22, 1, 0.36, 1] },
  },
  exit: {
    opacity: 0,
    x: -18,
    filter: "blur(6px)",
    transition: { duration: 0.22, ease: [0.7, 0, 0.84, 0] },
  },
};

const headerTelemetryItems = [
  { icon: <Radar size={16} />, label: "Retrieval Ready" },
  { icon: <Binary size={16} />, label: "Trace Linked" },
  { icon: <Wrench size={16} />, label: "Engineering Loop" },
];

const evidenceCards = [
  {
    doc: "ATA-29 Hydraulic System Manual",
    page: "P.118",
    section: "5.2 Pressure Drop Isolation",
    score: "0.97",
  },
  {
    doc: "Maintenance Bulletin MB-21-07",
    page: "P.16",
    section: "Leak Check Sequence",
    score: "0.92",
  },
  {
    doc: "Engine Bay Inspection Guide",
    page: "P.43",
    section: "Filter and Valve Verification",
    score: "0.88",
  },
];

const stackBadges = [
  "PyPDF2",
  "pdfplumber",
  "python-docx",
  "python-pptx",
  "MySQL",
  "Elasticsearch",
  "Chroma / Milvus",
  "BGE-M3 / m3e",
  "LangChain",
  "GPT / ChatGLM",
  "FastAPI",
  "Docker",
  "RESTful API",
];

const innovations = [
  {
    title: "规则优先、语义辅助的中文航空分块",
    icon: <Blocks size={18} />,
    body: "结合标题编号、ATA 章节、图表题注与语义边界对长文档进行精细切分，在保障专业语义完整性的同时，显著降低复杂中文航空技术资料的上下文割裂。",
  },
  {
    title: "领域词典增强的双层混合检索",
    icon: <Radar size={18} />,
    body: "以航空术语、缩略语与 ATA 编码词典增强关键词召回，再叠加向量模型完成语义精排，实现检索广度、专业精度与响应效率之间的高水平平衡。",
  },
  {
    title: "检索上下文驱动的可控问答",
    icon: <Bot size={18} />,
    body: "通过结构化提示词模板与高置信检索证据共同约束生成链路，使问答结果具备页码、章节与原文片段级引用能力，整体表达更加可信、可核验、可追溯。",
  },
  {
    title: "端到端可落地的工程化原型",
    icon: <Wrench size={18} />,
    body: "贯通文档接入、预处理、索引构建、混合检索、智能问答、前后端接口与 Docker 部署，形成一套具备展示价值与工程可复现性的完整原型闭环。",
  },
];

const supportHighlights = [
  {
    title: "Aimax A800",
    body: "依托国家级实验教学示范中心与创新开放实验室，提供大模型调用、向量检索与实验算力支持。",
  },
  {
    title: "公开航空仿真文档库",
    body: "以公开维修手册、技术标准和工程报告构建模拟企业文档环境，先完成原型验证再逐步试点。",
  },
  {
    title: "合成数据 + 主动学习",
    body: "初期通过大模型生成问答对并人工校验，逐步积累中文航空文档高质量训练样本。",
  },
];

const evaluationTargets = [
  { label: "Retrieval Accuracy", value: "92.6%", note: "围绕检索精度、查全率与查准率建立答辩评测口径" },
  { label: "Response Time", value: "1.8s", note: "面向原型系统的检索与生成联动平均响应目标" },
  { label: "Citation Coverage", value: "96.4%", note: "答案需覆盖页码、章节和原文片段，满足可追溯展示" },
  { label: "Deployment Readiness", value: "100%", note: "FastAPI、前端界面与 Docker 打包形成完整可演示闭环" },
];

const retrievalTraceSteps = [
  {
    stage: "Query Normalize",
    system: "FastAPI Gateway",
    detail: "抽取 ATA 章节、部件名、故障现象和检索偏好，统一封装为检索请求。",
    score: "42 ms",
  },
  {
    stage: "Keyword Recall",
    system: "Elasticsearch",
    detail: "术语词典、缩略语扩展与 ATA 编码并行召回，优先保障专业词命中。",
    score: "118 ms",
  },
  {
    stage: "Semantic Rank",
    system: "Vector Store",
    detail: "对候选片段执行嵌入相似度计算，并修复相邻 chunk 的上下文连续性。",
    score: "246 ms",
  },
  {
    stage: "Fusion Rerank",
    system: "Fusion Layer",
    detail: "按 BM25、向量相似度、章节权重和引用质量合成最终上下文。",
    score: "88 ms",
  },
  {
    stage: "Grounded Answer",
    system: "RAG Orchestrator",
    detail: "仅将高置信证据注入大模型生成链路，并回写页码、章节和来源卡片。",
    score: "1.8 s",
  },
];

const initialKnowledgeBaseRecords: KnowledgeBaseRecord[] = [
  {
    id: 101,
    name: "民航规章",
    description: "中国民航局发布的适航规章，如 CCAR-25、CCAR-33、CCAR-35、CCAR-121 等运输类飞机/发动机/螺旋桨适航标准与运行规则。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
  {
    id: 102,
    name: "规范性文件",
    description: "民航局及下属司局发布的管理办法、实施细则、咨询通告（AC）等规范性文件。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
  {
    id: 103,
    name: "法律法规",
    description: "《中华人民共和国民用航空法》及相关行政法规、国务院令等上位法律法规。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
  {
    id: 104,
    name: "国际公约",
    description: "ICAO 国际民航公约及附件、FAA/EASA 对等适航协议与国际标准参考。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
  {
    id: 105,
    name: "适航指令与通告",
    description: "适航指令（AD）、服务通告（SB）、维护通告与适航改进通知。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
  {
    id: 106,
    name: "维修手册与技术文档",
    description: "飞机/发动机维修手册（AMM/EMM）、故障隔离手册（FIM/TSM）、结构修理手册等技术文档。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
  {
    id: 107,
    name: "培训与参考资料",
    description: "培训课件、答辩汇报、术语词典、缩略语映射与其他参考资料。",
    creator: "系统预置",
    createdAt: "2026-04-01 00:00",
  },
];

const initialDocumentRecords: DocumentRecord[] = [];

const cloneKnowledgeBaseRecord = (item: KnowledgeBaseRecord): KnowledgeBaseRecord => ({ ...item });
const cloneDocumentRecord = (item: DocumentRecord): DocumentRecord => ({ ...item });

let fallbackKnowledgeBaseStore = initialKnowledgeBaseRecords.map(cloneKnowledgeBaseRecord);
let fallbackDocumentStore = initialDocumentRecords.map(cloneDocumentRecord);

const managementApiBase = (import.meta.env.VITE_MANAGEMENT_API_BASE?.trim() || "/api").replace(/\/$/, "");

async function requestManagement<T>(path: string, init?: RequestInit) {
  const response = await fetch(`${managementApiBase}${path}`, {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Management request failed: ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

async function uploadFileWithProgress(
  file: File,
  onProgress?: (progressPercent: number) => void,
): Promise<{ fileName: string; size: string; path: string }> {
  return await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.open("POST", `${managementApiBase}/upload`);

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.min(100, Math.max(0, Math.round((event.loaded / event.total) * 100)));
      onProgress?.(percent);
    };

    xhr.onerror = () => reject(new Error("文件上传失败"));
    xhr.onabort = () => reject(new Error("文件上传已取消"));
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(xhr.responseText || `上传失败: ${xhr.status}`));
        return;
      }
      try {
        resolve(JSON.parse(xhr.responseText) as { fileName: string; size: string; path: string });
      } catch {
        reject(new Error("上传响应解析失败"));
      }
    };

    xhr.send(formData);
  });
}

// ------- 真实 QA API 调用 -------
type QAApiSource = {
  content: string;
  doc_name: string;
  heading: string;
  parent_heading: string;
  score: number;
  chunk_id: string;
};

type QAApiResponse = {
  answer: string;
  sources: QAApiSource[];
};

async function callQAApi(question: string): Promise<AssistantConversation | null> {
  try {
    const response = await fetch(`${managementApiBase}/qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error(`QA API error: ${response.status}`);
    const data: QAApiResponse = await response.json();
    const ts = Date.now();
    const convId = `api-${ts}`;

    // 为每个 source 提取用于高亮的关键句（取 heading 或内容前两行中较短的行）
    const extractHighlight = (src: QAApiSource): string => {
      if (src.heading) return src.heading;
      const firstLines = src.content.split("\n").filter(Boolean).slice(0, 3);
      return firstLines.find((l) => l.length >= 8 && l.length <= 200) ?? firstLines[0] ?? "";
    };

    // 将原始内容整理为格式良好的行：按换行拆分，合并过短的相邻行以避免竖排效果
    const formatLines = (content: string): string[] => {
      const raw = content
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0);
      // 合并连续的短行（单字或极短行可能来自 PDF 竖排提取）
      const merged: string[] = [];
      let buffer = "";
      for (const line of raw) {
        if (line.length <= 2 && !/^[\d]+[.、)]?$/.test(line)) {
          buffer += line;
        } else {
          if (buffer) {
            // 若 buffer 自身够长则单独成行，否则拼到当前行前面
            if (buffer.length >= 10) {
              merged.push(buffer);
              buffer = "";
            } else {
              merged.push(buffer + line);
              buffer = "";
              continue;
            }
          }
          merged.push(line);
        }
      }
      if (buffer) merged.push(buffer);
      return merged.length > 0 ? merged : ["（无内容）"];
    };

    const citations: AssistantCitation[] = data.sources.map((src, i) => {
      const highlight = extractHighlight(src);
      return {
        id: `${convId}-c${i + 1}`,
        label: `来源${i + 1}: ${src.doc_name}${src.heading ? " · " + src.heading : ""}`.slice(0, 48),
        sourceId: `${convId}-s${i + 1}`,
        documentTitle: src.doc_name || "检索文档",
        documentType: "AMM" as AssistantDocumentType,
        chapter: src.heading || src.parent_heading || "",
        page: 1,
        quotedText: src.content.slice(0, 400),
        highlightText: highlight,
        snippet: src.content.slice(0, 400),
        evidenceLocation: {
          page: 1,
          chapter: src.heading || "",
          textAnchor: highlight,
        },
      };
    });

    const sources: AssistantSource[] = data.sources.map((src, i) => ({
      id: `${convId}-s${i + 1}`,
      documentTitle: src.doc_name || "检索文档",
      documentType: "AMM" as AssistantDocumentType,
      aircraftModel: "ALL",
      chapter: src.heading || src.parent_heading || "",
      pageCount: 1,
      snippet: src.content.slice(0, 300),
      pages: [
        {
          pageNumber: 1,
          sectionTitle: src.heading || src.parent_heading || "检索结果",
          lines: formatLines(src.content),
        },
      ],
    }));

    // 回答整体作为一个 segment，所有 citation 附在底部
    const segments: AssistantMessageSegment[] = [
      {
        id: `${convId}-seg-answer`,
        kind: "paragraph",
        text: data.answer || "未获取到回答。",
        citationIds: citations.map((c) => c.id),
      },
    ];

    return {
      id: convId,
      quickPrompt: question,
      userQuestion: question,
      aircraftModel: "ALL",
      documentTypeFocus: "ALL",
      defaultCitationId: citations[0]?.id ?? "",
      messages: [
        { id: `${convId}-user`, role: "user", content: question },
        { id: `${convId}-assistant`, role: "assistant", segments },
      ],
      citations,
      sources,
    };
  } catch (err) {
    console.warn("QA API unavailable, falling back to mock:", err);
    return null;
  }
}

const knowledgeBaseDocumentService = {
  async listKnowledgeBases() {
    try {
      const records = await requestManagement<KnowledgeBaseRecord[]>("/knowledge-bases");
      fallbackKnowledgeBaseStore = records.map(cloneKnowledgeBaseRecord);
      return records;
    } catch (error) {
      console.warn("Management API unavailable, using local fallback store for knowledge bases.", error);
      return fallbackKnowledgeBaseStore.map(cloneKnowledgeBaseRecord);
    }
  },
  async createKnowledgeBase(payload: KnowledgeBasePayload) {
    try {
      const record = await requestManagement<KnowledgeBaseRecord>("/knowledge-bases", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      fallbackKnowledgeBaseStore = [cloneKnowledgeBaseRecord(record), ...fallbackKnowledgeBaseStore.filter((item) => item.id !== record.id)];
      return record;
    } catch (error) {
      console.warn("Management API unavailable, creating knowledge base in local fallback store.", error);
      const nextId = Math.max(100, ...fallbackKnowledgeBaseStore.map((item) => item.id)) + 1;
      const record: KnowledgeBaseRecord = {
        id: nextId,
        name: payload.name,
        description: payload.description,
        creator: payload.creator?.trim() || "he114TnT",
        createdAt: getManagementTimestamp(),
      };
      fallbackKnowledgeBaseStore = [record, ...fallbackKnowledgeBaseStore];
      return cloneKnowledgeBaseRecord(record);
    }
  },
  async updateKnowledgeBase(id: number, payload: KnowledgeBasePayload) {
    try {
      const record = await requestManagement<KnowledgeBaseRecord>(`/knowledge-bases/${id}`, {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      fallbackKnowledgeBaseStore = fallbackKnowledgeBaseStore.map((item) => (item.id === id ? cloneKnowledgeBaseRecord(record) : item));
      return record;
    } catch (error) {
      console.warn("Management API unavailable, updating knowledge base in local fallback store.", error);
      let updatedRecord: KnowledgeBaseRecord | null = null;

      fallbackKnowledgeBaseStore = fallbackKnowledgeBaseStore.map((item) => {
        if (item.id !== id) return item;

        updatedRecord = {
          ...item,
          name: payload.name,
          description: payload.description,
        };

        return updatedRecord;
      });

      if (!updatedRecord) {
        throw new Error(`Knowledge base ${id} not found.`);
      }

      return cloneKnowledgeBaseRecord(updatedRecord);
    }
  },
  async deleteKnowledgeBase(id: number) {
    try {
      await requestManagement<void>(`/knowledge-bases/${id}`, { method: "DELETE" });
    } catch (error) {
      console.warn("Management API unavailable, deleting knowledge base in local fallback store.", error);
    }

    fallbackKnowledgeBaseStore = fallbackKnowledgeBaseStore.filter((item) => item.id !== id);
    fallbackDocumentStore = fallbackDocumentStore.filter((item) => item.knowledgeBaseId !== id);
  },
  async listDocuments() {
    try {
      const records = await requestManagement<DocumentRecord[]>("/documents");
      fallbackDocumentStore = records.map(cloneDocumentRecord);
      return records;
    } catch (error) {
      console.warn("Management API unavailable, using local fallback store for documents.", error);
      return fallbackDocumentStore.map(cloneDocumentRecord);
    }
  },
  async createDocument(payload: DocumentPayload) {
    try {
      const record = await requestManagement<DocumentRecord>("/documents", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      fallbackDocumentStore = [cloneDocumentRecord(record), ...fallbackDocumentStore.filter((item) => item.id !== record.id)];
      return record;
    } catch (error) {
      console.warn("Management API unavailable, creating document in local fallback store.", error);
      const nextId = Math.max(3000, ...fallbackDocumentStore.map((item) => item.id)) + 1;
      const record: DocumentRecord = {
        id: nextId,
        ...payload,
        uploadedAt: getManagementTimestamp(),
      };
      fallbackDocumentStore = [record, ...fallbackDocumentStore];
      return cloneDocumentRecord(record);
    }
  },
  async deleteDocument(id: number) {
    try {
      await requestManagement<void>(`/documents/${id}`, { method: "DELETE" });
    } catch (error) {
      console.warn("Management API unavailable, deleting document in local fallback store.", error);
    }

    fallbackDocumentStore = fallbackDocumentStore.filter((item) => item.id !== id);
  },
};

async function summarizeConversationTitle(question: string, answer: string): Promise<string> {
  try {
    const res = await fetch(`${managementApiBase}/summarize-title`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, answer: answer.slice(0, 300) }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.title) return data.title;
    }
  } catch { /* fallback to truncated question */ }
  return question.length > 15 ? question.slice(0, 15) + "…" : question;
}

const retrievalObservations = [
  {
    label: "Recall Layer",
    title: "前端只发送统一查询，后端自动触发关键词召回，不暴露 Elasticsearch 操作细节。",
    score: "Elastic",
  },
  {
    label: "Semantic Layer",
    title: "前端只消费 trace record，向量检索与 rerank 在服务侧完成，不让用户直接操作向量库。",
    score: "Vector",
  },
  {
    label: "Fusion Output",
    title: "界面负责展示命中文档、证据来源与检索轨迹，让流程可解释但不让实现细节侵入交互。",
    score: "Grounded",
  },
];

function useInteractivePointer({
  maxShiftRatio = 0.028,
  maxShift = 11,
  maxRotate = 4.2,
}: {
  maxShiftRatio?: number;
  maxShift?: number;
  maxRotate?: number;
} = {}) {
  const glowX = useMotionValue(120);
  const glowY = useMotionValue(40);
  const shiftX = useMotionValue(0);
  const shiftY = useMotionValue(0);
  const rawRotateX = useMotionValue(0);
  const rawRotateY = useMotionValue(0);

  const springGlowX = useSpring(glowX, { stiffness: 320, damping: 32, mass: 0.3 });
  const springGlowY = useSpring(glowY, { stiffness: 320, damping: 32, mass: 0.3 });
  const springShiftX = useSpring(shiftX, { stiffness: 220, damping: 18, mass: 0.35 });
  const springShiftY = useSpring(shiftY, { stiffness: 220, damping: 18, mass: 0.35 });
  const rotateX = useSpring(rawRotateX, { stiffness: 180, damping: 18, mass: 0.45 });
  const rotateY = useSpring(rawRotateY, { stiffness: 180, damping: 18, mass: 0.45 });

  const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

  const handleMove = (event: React.MouseEvent<HTMLElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const dx = localX - rect.width / 2;
    const dy = localY - rect.height / 2;
    const ratioX = clamp(dx / Math.max(rect.width / 2, 1), -1, 1);
    const ratioY = clamp(dy / Math.max(rect.height / 2, 1), -1, 1);
    const easedX = Math.sin(ratioX * (Math.PI / 2));
    const easedY = Math.sin(ratioY * (Math.PI / 2));
    const maxShiftX = Math.min(rect.width * maxShiftRatio, maxShift);
    const maxShiftY = Math.min(rect.height * maxShiftRatio, maxShift);

    glowX.set(localX);
    glowY.set(localY);
    shiftX.set(easedX * maxShiftX);
    shiftY.set(easedY * maxShiftY);
    rawRotateX.set(-easedY * maxRotate);
    rawRotateY.set(easedX * maxRotate);
  };

  const handleLeave = () => {
    shiftX.set(0);
    shiftY.set(0);
    rawRotateX.set(0);
    rawRotateY.set(0);
  };

  return {
    glowX: springGlowX,
    glowY: springGlowY,
    shiftX: springShiftX,
    shiftY: springShiftY,
    rotateX,
    rotateY,
    handleMove,
    handleLeave,
  };
}

function GlowTileButton({
  children,
  accent,
  active = false,
  onClick,
  className = "",
}: {
  children: ReactNode;
  accent: string;
  active?: boolean;
  onClick?: () => void;
  className?: string;
}) {
  const pointer = useInteractivePointer();
  const glow = useMotionTemplate`radial-gradient(180px circle at ${pointer.glowX}px ${pointer.glowY}px, ${accent}66 0%, transparent 72%)`;

  return (
    <motion.button
      onClick={onClick}
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ scale: 1.012, y: -3 }}
      whileTap={{ scale: 0.992 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1200,
        clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)",
        borderColor: active ? `${accent}a6` : "rgba(255,255,255,0.08)",
        background: active ? `${accent}12` : "rgba(255,255,255,0.02)",
        boxShadow: active ? `0 0 32px ${accent}20, inset 0 0 0 1px ${accent}44` : "0 0 0 transparent",
      }}
      className={`group relative overflow-hidden border text-left transition-[border-color,background,box-shadow] duration-300 ${className}`}
    >
      <motion.span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <span className="pointer-events-none absolute inset-0 -translate-x-full bg-[linear-gradient(120deg,transparent_0%,rgba(255,255,255,0.16)_46%,transparent_100%)] opacity-0 transition-all duration-500 group-hover:translate-x-full group-hover:opacity-100" />
      <span className="pointer-events-none absolute inset-x-6 bottom-0 h-px origin-left scale-x-0 bg-white/70 transition-transform duration-300 group-hover:scale-x-100" />
      <span className="relative z-10 block">{children}</span>
    </motion.button>
  );
}

function useStageScrollReveal<T extends HTMLElement>(amount = 0.18) {
  const ref = useRef<T | null>(null);
  const [active, setActive] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const scrollHost = node.closest(".stage-scroll");
    if (!(scrollHost instanceof HTMLElement)) return;

    let frameId = 0;

    const evaluateVisibility = () => {
      const hostRect = scrollHost.getBoundingClientRect();
      const nodeRect = node.getBoundingClientRect();
      const visibleTop = Math.max(nodeRect.top, hostRect.top);
      const visibleBottom = Math.min(nodeRect.bottom, hostRect.bottom);
      const visibleHeight = Math.max(0, visibleBottom - visibleTop);
      const visibleRatio = visibleHeight / Math.max(nodeRect.height, 1);
      const isVisible = visibleHeight > 0 && visibleRatio >= amount;

      setActive(isVisible);
    };

    const handleScroll = () => {
      window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(evaluateVisibility);
    };

    evaluateVisibility();
    scrollHost.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);

    return () => {
      window.cancelAnimationFrame(frameId);
      scrollHost.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    };
  }, [amount]);

  return { ref, active };
}

function StageScrollReveal({
  children,
  initial,
  reveal,
  transition,
  className,
  style,
  whileHover,
  amount = 0.18,
}: {
  children?: ReactNode;
  initial: TargetAndTransition;
  reveal: TargetAndTransition;
  transition?: Transition;
  className?: string;
  style?: MotionProps["style"];
  whileHover?: MotionProps["whileHover"];
  amount?: number;
}) {
  const { ref, active } = useStageScrollReveal<HTMLDivElement>(amount);

  return (
    <motion.div
      ref={ref}
      initial={false}
      animate={active ? reveal : initial}
      transition={transition}
      className={className}
      style={style}
      whileHover={whileHover}
    >
      {children}
    </motion.div>
  );
}

function ChapterRail({
  activeChapter,
  setActiveChapter,
  chapter,
  mode,
  onBackToLaunch,
}: {
  activeChapter: number;
  setActiveChapter: (value: number) => void;
  chapter: Chapter;
  mode: "mobile" | "desktop";
  onBackToLaunch?: () => void;
}) {
  return (
    <>
      <div className={`mb-5 flex gap-2 overflow-x-auto pb-2 ${mode === "mobile" ? "xl:hidden" : "hidden"}`}>
        {onBackToLaunch ? (
          <GlowTileButton accent={introTheme.accent} onClick={onBackToLaunch} className="min-w-[10.5rem] px-4 py-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="font-display text-sm uppercase tracking-[0.18em]" style={{ color: introTheme.accent }}>
                  Back
                </div>
                <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-white/55">Launch Deck</div>
              </div>
              <ChevronLeft size={15} className="text-white/45" />
            </div>
          </GlowTileButton>
        ) : null}
        {chapters.map((item, index) => (
          <GlowTileButton
            key={item.code}
            accent={index === activeChapter ? chapter.accent : item.accent}
            active={index === activeChapter}
            onClick={() => setActiveChapter(index)}
            className="min-w-[10.5rem] px-4 py-3"
          >
            <div className="font-display text-lg uppercase tracking-[0.18em]" style={{ color: index === activeChapter ? chapter.accent : "#f8f9fb" }}>
              {item.code}
            </div>
            <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-white/45">{item.title}</div>
          </GlowTileButton>
        ))}
      </div>
      <div className={mode === "desktop" ? "hidden h-full xl:flex xl:flex-col" : "hidden"}>
        {onBackToLaunch ? (
          <div className="mb-3">
            <GlowTileButton accent={introTheme.accent} onClick={onBackToLaunch} className="w-full px-4 py-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="font-display text-sm uppercase tracking-[0.18em]" style={{ color: introTheme.accent }}>
                    Back To
                  </div>
                  <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-white/55">Launch Deck</div>
                </div>
                <ChevronLeft size={15} className="text-white/45 transition-transform duration-300 group-hover:-translate-x-1" />
              </div>
            </GlowTileButton>
          </div>
        ) : null}
        <div className="mb-4 border-b border-white/8 pb-4">
          <div className="text-[10px] uppercase tracking-[0.34em] text-white/35">System Directory</div>
          <div className="mt-2 font-display text-xl uppercase tracking-[0.16em] text-white">{chapter.title}</div>
        </div>
        <div className="flex flex-1 flex-col gap-3">
          {chapters.map((item, index) => (
            <GlowTileButton
              key={item.code}
              accent={index === activeChapter ? chapter.accent : item.accent}
              active={index === activeChapter}
              onClick={() => setActiveChapter(index)}
              className="px-4 py-4"
            >
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="font-display text-xl uppercase tracking-[0.18em]" style={{ color: index === activeChapter ? chapter.accent : "#f8f9fb" }}>
                    {item.code}
                  </div>
                  <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-white/45">{item.title}</div>
                </div>
                <ArrowRight size={15} className="text-white/35 transition-transform duration-300 group-hover:translate-x-1" />
              </div>
            </GlowTileButton>
          ))}
        </div>
      </div>
    </>
  );
}

function hexToRgba(hex: string, alpha: number) {
  const normalized = hex.replace("#", "");
  const value = normalized.length === 3 ? normalized.split("").map((char) => char + char).join("") : normalized;
  const num = Number.parseInt(value, 16);
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function createWaveParticleTexture(tint: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;

  const context = canvas.getContext("2d");
  if (!context) return null;

  const gradient = context.createRadialGradient(48, 48, 0, 48, 48, 48);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.24, "rgba(255,255,255,0.96)");
  gradient.addColorStop(0.5, hexToRgba(tint, 0.34));
  gradient.addColorStop(0.72, "rgba(219,224,232,0.16)");
  gradient.addColorStop(1, "rgba(219,224,232,0)");

  context.fillStyle = gradient;
  context.fillRect(0, 0, 96, 96);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function DigitalWaveField({
  mouseX,
  mouseY,
  chapter,
}: {
  mouseX: MotionValue<number>;
  mouseY: MotionValue<number>;
  chapter: Chapter;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let renderer: THREE.WebGLRenderer | null = null;

    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        powerPreference: "high-performance",
      });
    } catch {
      return;
    }

    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(56, 1, 1, 6000);
    const particleTexture = createWaveParticleTexture(chapter.secondary);
    const tintedWhite = new THREE.Color(chapter.secondary).lerp(new THREE.Color("#edf1f5"), 0.78);

    const amountX = 54;
    const amountY = 42;
    const separation = 88;
    const totalPoints = amountX * amountY;
    const positions = new Float32Array(totalPoints * 3);
    const baseX = new Float32Array(totalPoints);
    const baseZ = new Float32Array(totalPoints);

    let pointIndex = 0;

    for (let iy = 0; iy < amountY; iy += 1) {
      for (let ix = 0; ix < amountX; ix += 1) {
        const x = ix * separation - (amountX * separation) / 2;
        const z = iy * separation - amountY * separation * 0.9 - 120;

        baseX[pointIndex] = x;
        baseZ[pointIndex] = z;
        positions[pointIndex * 3] = x;
        positions[pointIndex * 3 + 1] = 0;
        positions[pointIndex * 3 + 2] = z;

        pointIndex += 1;
      }
    }

    const geometry = new THREE.BufferGeometry();
    const positionAttribute = new THREE.BufferAttribute(positions, 3);
    geometry.setAttribute("position", positionAttribute);

    const material = new THREE.PointsMaterial({
      color: tintedWhite,
      size: 28,
      map: particleTexture ?? undefined,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
      alphaTest: 0.08,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(geometry, material);
    particles.position.set(0, -150, -340);
    particles.rotation.set(-0.2, 0, 0);
    scene.add(particles);

    let frameId = 0;
    let width = 0;
    let height = 0;
    let smoothX = 0;
    let smoothY = 0;
    const lookAt = new THREE.Vector3(0, -120, -620);
    const targetLookAt = new THREE.Vector3(0, -120, -620);

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      renderer?.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
      renderer?.setSize(width, height, false);
    };

    const render = (time: number) => {
      const t = time * 0.001;
      const pointerX = mouseX.get() || width * 0.5;
      const pointerY = mouseY.get() || height * 0.5;
      const normX = pointerX / Math.max(width, 1) - 0.5;
      const normY = pointerY / Math.max(height, 1) - 0.5;

      smoothX += (normX - smoothX) * 0.07;
      smoothY += (normY - smoothY) * 0.07;

      for (let iy = 0; iy < amountY; iy += 1) {
        for (let ix = 0; ix < amountX; ix += 1) {
          const index = iy * amountX + ix;
          const positionCursor = index * 3;
          const ridge =
            Math.sin((ix + t * 3.2) * 0.32) * 46 +
            Math.sin((iy + t * 2.6) * 0.5) * 44 +
            Math.sin((ix * 0.45 + iy * 0.22 + t * 1.9) * 0.9) * 18;
          const rollBand = Math.cos(iy * 0.28 - t * 2.1 + smoothX * 7.5) * 12;
          const pull = Math.cos(ix * 0.18 + t * 1.6 + smoothY * 6) * 8;

          positions[positionCursor] = baseX[index];
          positions[positionCursor + 1] = ridge + rollBand + pull;
          positions[positionCursor + 2] = baseZ[index];
        }
      }

      positionAttribute.needsUpdate = true;

      camera.position.x += (smoothX * 560 - camera.position.x) * 0.045;
      camera.position.y += ((220 - smoothY * 400) - camera.position.y) * 0.05;
      camera.position.z += ((900 - smoothY * 120) - camera.position.z) * 0.035;
      camera.rotation.z += ((-smoothX * 0.08) - camera.rotation.z) * 0.04;

      targetLookAt.set(smoothX * 320, -130 + smoothY * 170, -620);
      lookAt.lerp(targetLookAt, 0.08);
      camera.lookAt(lookAt);

      particles.position.x += (smoothX * 120 - particles.position.x) * 0.035;
      particles.rotation.z += ((-smoothX * 0.16) - particles.rotation.z) * 0.045;
      particles.rotation.y += ((smoothX * 0.32) - particles.rotation.y) * 0.04;
      particles.rotation.x += ((-0.28 - smoothY * 0.2) - particles.rotation.x) * 0.05;

      renderer?.render(scene, camera);
      frameId = window.requestAnimationFrame(render);
    };

    resize();
    camera.position.set(0, 220, 900);
    window.addEventListener("resize", resize);
    frameId = window.requestAnimationFrame(render);

    return () => {
      window.removeEventListener("resize", resize);
      window.cancelAnimationFrame(frameId);
      geometry.dispose();
      material.dispose();
      particleTexture?.dispose();
      renderer?.dispose();
    };
  }, [chapter.secondary, mouseX, mouseY]);

  return (
    <div
      className="pointer-events-none absolute inset-0 overflow-hidden"
      style={{ maskImage: "linear-gradient(180deg, transparent 0%, transparent 34%, rgba(0,0,0,0.18) 50%, rgba(0,0,0,0.82) 72%, black 100%)" }}
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full opacity-[0.94]" style={{ mixBlendMode: "screen" }} />
    </div>
  );
}

function FeatureCard({
  accent,
  icon,
  title,
  desc,
}: {
  accent: string;
  icon: ReactNode;
  title: string;
  desc: string;
}) {
  const pointer = useInteractivePointer();
  const glow = useMotionTemplate`radial-gradient(220px circle at ${pointer.glowX}px ${pointer.glowY}px, ${accent}4a 0%, transparent 70%)`;

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ y: -10, scale: 1.025 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1400,
      }}
      className="group relative"
    >
      <motion.div className="pointer-events-none absolute inset-0 rounded-sm opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <CutPanel accent={accent} className="h-full transition-transform duration-300 group-hover:shadow-[0_0_46px_rgba(255,255,255,0.08)]">
        <motion.div
          className="mb-4 inline-flex h-12 w-12 items-center justify-center border border-white/10 bg-white/5 text-white/90"
          animate={{ rotate: [0, 6, -6, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          whileHover={{ scale: 1.12, borderColor: accent, boxShadow: `0 0 26px ${accent}44` }}
        >
          {icon}
        </motion.div>
        <div className="font-display text-lg uppercase tracking-[0.16em] text-white">{title}</div>
        <div className="mt-3 text-sm leading-6 text-white/60">{desc}</div>
        <motion.div
          className="mt-4 h-[2px] w-16"
          style={{ background: `linear-gradient(90deg, ${accent}, transparent)` }}
          initial={{ scaleX: 0.55, originX: 0 }}
          whileHover={{ scaleX: 1.55 }}
        />
      </CutPanel>
    </motion.div>
  );
}

function MetricCard({
  chapter,
  label,
  value,
  note,
  index,
}: {
  chapter: Chapter;
  label: string;
  value: string;
  note: string;
  index: number;
}) {
  const pointer = useInteractivePointer();
  const { ref, active } = useStageScrollReveal<HTMLDivElement>(0.22);
  const highlight = index % 2 === 0 ? chapter.accent : chapter.secondary;
  const glow = useMotionTemplate`radial-gradient(220px circle at ${pointer.glowX}px ${pointer.glowY}px, ${highlight}58 0%, transparent 72%)`;

  return (
    <motion.div
      ref={ref}
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      className="group relative overflow-hidden border border-white/10 bg-white/[0.008] p-5"
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1200,
        clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 18px), calc(100% - 18px) 100%, 0 100%)",
      }}
      initial={false}
      animate={
        active
          ? { opacity: 1, x: 0, y: 0, scale: 1, filter: "blur(0px)", transition: { duration: 0.48, delay: 0.08 * index } }
          : { opacity: 0, x: index % 2 === 0 ? -28 : 28, y: 28, scale: 0.92, filter: "blur(12px)", transition: { duration: 0.28 } }
      }
      whileHover={{ y: -12, scale: 1.025 }}
    >
      <motion.span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <motion.div
        className="pointer-events-none absolute inset-x-0 top-0 h-[3px]"
        style={{ background: `linear-gradient(90deg, transparent, ${highlight}, transparent)` }}
        animate={{ x: ["-100%", "100%"] }}
        transition={{ duration: 3 + index * 0.35, repeat: Infinity, ease: "linear" }}
      />
      <div className="relative z-10 text-[10px] uppercase tracking-[0.32em] text-white/35">{label}</div>
      <motion.div
        className="relative z-10 mt-4 font-display text-5xl uppercase tracking-[0.12em]"
        style={{ color: highlight }}
        whileHover={{ scale: 1.05, textShadow: `0 0 24px ${highlight}` }}
      >
        {value}
      </motion.div>
      <div className="relative z-10 mt-3 text-sm leading-6 text-white/55">{note}</div>
    </motion.div>
  );
}

function MagneticButton({
  children,
  accent,
  secondary = false,
  onClick,
  className = "",
}: {
  children: ReactNode;
  accent: string;
  secondary?: boolean;
  onClick?: () => void;
  className?: string;
}) {
  const pointer = useInteractivePointer();
  const glow = useMotionTemplate`radial-gradient(180px circle at ${pointer.glowX}px ${pointer.glowY}px, ${
    secondary ? `${accent}40` : `${accent}7a`
  } 0%, transparent 70%)`;

  return (
    <motion.button
      whileHover={{ scale: 1.03, y: -4 }}
      whileTap={{ scale: 0.985 }}
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      onClick={onClick}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1200,
        borderColor: secondary ? "rgba(255,255,255,0.16)" : accent,
        background: secondary ? "rgba(255,255,255,0.04)" : accent,
        color: secondary ? "#f5f7fa" : "#08090b",
        boxShadow: secondary
          ? `0 0 0 1px rgba(255,255,255,0.05), 0 0 24px ${accent}18`
          : `0 0 24px ${accent}55, 0 0 54px ${accent}3d, inset 0 0 20px rgba(255,255,255,0.12)`,
      }}
      className={`group relative overflow-hidden rounded-none border px-5 py-3 font-display text-[11px] uppercase tracking-[0.28em] transition-colors duration-300 ${className}`}
    >
      <motion.span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/45 to-transparent transition-transform duration-500 group-hover:translate-x-full" />
      <span className="pointer-events-none absolute inset-x-3 bottom-0 h-px origin-left scale-x-0 bg-white/80 transition-transform duration-300 group-hover:scale-x-100" />
      <span className="relative z-10 flex items-center gap-3 [&_svg]:transition-transform [&_svg]:duration-300 group-hover:[&_svg]:translate-x-1 group-hover:[&_svg]:rotate-6">
        {children}
      </span>
    </motion.button>
  );
}

function CutPanel({
  accent,
  title,
  kicker,
  className = "",
  headerClassName = "",
  actions,
  children,
}: {
  accent: string;
  title?: string;
  kicker?: string;
  className?: string;
  headerClassName?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div
      className={`cut-panel relative overflow-hidden border border-white/10 bg-black/[0.045] p-4 shadow-panel ${className}`}
      style={{ "--panel-accent": accent } as CSSProperties}
    >
      {(title || kicker) && (
        <div className={`mb-3 flex items-start gap-3 ${headerClassName}`}>
          <div>
            {kicker && <div className="mb-1 text-[10px] uppercase tracking-[0.34em] text-white/45">{kicker}</div>}
            {title && <div className="font-display text-lg uppercase tracking-[0.18em] text-white">{title}</div>}
          </div>
          <div className="h-[1px] flex-1 translate-y-3 bg-white/10" />
          {actions ? <div className="shrink-0">{actions}</div> : null}
        </div>
      )}
      {children}
    </div>
  );
}

function InteractiveCutPanel({
  accent,
  secondary,
  title,
  kicker,
  className = "",
  wrapperClassName = "",
  actions,
  children,
}: {
  accent: string;
  secondary?: string;
  title?: string;
  kicker?: string;
  className?: string;
  wrapperClassName?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const pointer = useInteractivePointer({
    maxShiftRatio: 0.014,
    maxShift: 5,
    maxRotate: 2.1,
  });
  const glow = useMotionTemplate`radial-gradient(190px circle at ${pointer.glowX}px ${pointer.glowY}px, ${
    secondary ?? accent
  }4a 0%, transparent 72%)`;

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ y: -6, scale: 1.006 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1400,
      }}
      className={`group relative ${wrapperClassName}`}
    >
      <motion.div className="pointer-events-none absolute inset-0 rounded-sm opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <CutPanel
        accent={accent}
        title={title}
        kicker={kicker}
        actions={actions}
        className={`transition-[box-shadow,transform] duration-300 group-hover:shadow-[0_0_54px_rgba(255,255,255,0.08)] ${className}`}
      >
        {children}
      </CutPanel>
    </motion.div>
  );
}

function TacticalTag({ children, accent }: { children: ReactNode; accent: string }) {
  return (
    <span
      className="inline-flex items-center border px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-white/80"
      style={{ borderColor: `${accent}80`, background: `${accent}14` }}
    >
      {children}
    </span>
  );
}

function StageHeader({
  chapter,
  title,
  tags,
  className = "",
  actions,
  compact = false,
}: {
  chapter: Chapter;
  title?: string;
  tags?: string[];
  className?: string;
  actions?: ReactNode;
  compact?: boolean;
}) {
  const tagItems = tags ?? ["Aviation Domain", "Thesis Demo Ready", "Traceable Intelligence"];
  const hasTags = tagItems.length > 0;

  if (actions) {
    return (
      <div
        className={`${compact ? "mb-3" : "mb-6"} grid ${compact ? "gap-3" : "gap-4"} ${
          hasTags ? "xl:grid-cols-[minmax(0,1fr)_auto_auto]" : "xl:grid-cols-[minmax(0,1fr)_auto]"
        } xl:items-center ${className}`}
      >
        <div className="min-w-0">
          <div className={`${compact ? "mb-1 text-[10px] tracking-[0.38em]" : "mb-2 text-[11px] tracking-[0.42em]"} uppercase text-white/45`}>
            System Chapter {chapter.code}
          </div>
          <h2 className={`font-display uppercase tracking-[0.18em] text-white ${compact ? "text-[1.55rem] md:text-[2rem]" : "text-3xl md:text-4xl"}`}>
            {title ?? chapter.subtitle}
          </h2>
        </div>
        {hasTags ? (
          <div className={`flex flex-wrap ${compact ? "gap-1.5" : "gap-2"} xl:justify-end`}>
            {tagItems.map((tag, index) => (
              <TacticalTag key={tag} accent={index % 2 === 0 ? chapter.accent : chapter.secondary}>
                {tag}
              </TacticalTag>
            ))}
          </div>
        ) : null}
        <div className="flex justify-start xl:justify-end">{actions}</div>
      </div>
    );
  }

  return (
    <div className={`${compact ? "mb-3 gap-4" : "mb-6 gap-6"} flex flex-wrap items-end justify-between ${className}`}>
      <div>
        <div className={`${compact ? "mb-1 text-[10px] tracking-[0.38em]" : "mb-2 text-[11px] tracking-[0.42em]"} uppercase text-white/45`}>
          System Chapter {chapter.code}
        </div>
        <h2 className={`font-display uppercase tracking-[0.18em] text-white ${compact ? "text-[1.55rem] md:text-[2rem]" : "text-3xl md:text-4xl"}`}>
          {title ?? chapter.subtitle}
        </h2>
      </div>
      {hasTags ? (
        <div className={`flex flex-wrap ${compact ? "gap-1.5" : "gap-2"}`}>
          {tagItems.map((tag, index) => (
            <TacticalTag key={tag} accent={index % 2 === 0 ? chapter.accent : chapter.secondary}>
              {tag}
            </TacticalTag>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function getManagementTimestamp(date = new Date()) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
    .format(date)
    .replace(/\//g, "-");
}

function getDocumentStatusAccent(status: DocumentStatus, chapter: Chapter) {
  switch (status) {
    case "READY":
    case "SYNCED":
      return chapter.secondary;
    case "FAILED":
      return "#ff6c86";
    default:
      return chapter.accent;
  }
}

function getDocumentSummaryText(summary: string) {
  const trimmed = summary.trim();
  return trimmed ? trimmed : "用户未输入摘要";
}

function parseChunkCountFromIndexMessage(message: string) {
  const match = message.match(/共\s*(\d+)\s*个语义块/) ?? message.match(/(\d+)\s*个语义块/);
  if (!match) return 0;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function getUploadStageLabel(stage: UploadStage) {
  switch (stage) {
    case "uploading":
      return "上传文件";
    case "indexing":
      return "语义分块与索引";
    case "saving":
      return "写入文档记录";
    case "done":
      return "完成";
    case "error":
      return "失败";
    default:
      return "待开始";
  }
}

function handleFramePointerMove(event: React.MouseEvent<HTMLElement>) {
  const target = event.currentTarget;
  const rect = target.getBoundingClientRect();
  const dx = event.clientX - (rect.left + rect.width / 2);
  const dy = event.clientY - (rect.top + rect.height / 2);
  const nx = rect.width > 0 ? Math.max(-1, Math.min(1, dx / (rect.width / 2))) : 0;
  const ny = rect.height > 0 ? Math.max(-1, Math.min(1, dy / (rect.height / 2))) : 0;
  const shiftX = nx * 4;
  const shiftY = ny * 3;
  const rotateX = -ny * 1.6;
  const rotateY = nx * 1.6;
  target.style.transform = `perspective(1200px) translate3d(${shiftX}px, ${shiftY}px, 0) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
}

function handleFramePointerLeave(event: React.MouseEvent<HTMLElement>) {
  event.currentTarget.style.transform = "";
}

function inferKnowledgeBaseIdForDocument(
  payload: Pick<DocumentRecord, "fileName" | "summary" | "type">,
  knowledgeBases: KnowledgeBaseRecord[],
  preferredKnowledgeBaseId?: number | "all",
) {
  const haystack = `${payload.fileName} ${payload.summary} ${payload.type}`.toLowerCase();
  const rules: Array<{ id: number; keywords: string[] }> = [
    { id: 101, keywords: ["液压", "hydraulic", "ata 29", "pressure", "actuator"] },
    { id: 102, keywords: ["发动机", "engine", "ata 72", "compressor", "turbine"] },
    { id: 103, keywords: ["起落架", "landing gear", "轮刹", "刹车", "收放"] },
    { id: 104, keywords: ["术语", "缩略语", "映射", "编码", "词典", "glossary"] },
    { id: 105, keywords: ["章节", "手册", "manual", "流程卡", "检修流程", "chapter"] },
    { id: 106, keywords: ["适航", "指令", "通告", "ad ", "bulletin", "directive"] },
    { id: 107, keywords: ["培训", "课件", "汇报", "演示", "答辩", "presentation"] },
  ];

  const matchedRule = rules.find((rule) => rule.keywords.some((keyword) => haystack.includes(keyword)));
  const matchedKnowledgeBase = matchedRule ? knowledgeBases.find((item) => item.id === matchedRule.id) : null;

  if (matchedKnowledgeBase) {
    return matchedKnowledgeBase.id;
  }

  if (preferredKnowledgeBaseId && preferredKnowledgeBaseId !== "all" && knowledgeBases.some((item) => item.id === preferredKnowledgeBaseId)) {
    return preferredKnowledgeBaseId;
  }

  return knowledgeBases[0]?.id ?? 0;
}

function ManagementFieldShell({
  accent,
  icon,
  className = "",
  children,
}: {
  accent: string;
  icon?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  const pointer = useInteractivePointer({
    maxShiftRatio: 0.01,
    maxShift: 3,
    maxRotate: 1.2,
  });

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      className={`relative flex h-10 items-center gap-2.5 overflow-hidden border border-white/10 bg-white/[0.03] px-3.5 ${className}`}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1000,
        clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)",
      }}
    >
      {icon ? <span className="shrink-0 text-white/30">{icon}</span> : null}
      <div className="min-w-0 flex-1">{children}</div>
      <span className="pointer-events-none absolute inset-x-3 bottom-0 h-px opacity-55" style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }} />
    </motion.div>
  );
}

function ManagementActionButton({
  accent,
  children,
  icon,
  onClick,
  secondary = false,
  hoverOutlineOnly = false,
  disabled = false,
  type = "button",
  className = "",
}: {
  accent: string;
  children: ReactNode;
  icon?: ReactNode;
  onClick?: () => void;
  secondary?: boolean;
  hoverOutlineOnly?: boolean;
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  const pointer = useInteractivePointer({
    maxShiftRatio: 0.01,
    maxShift: 2.5,
    maxRotate: 1,
  });
  const [hovered, setHovered] = useState(false);
  const showSurface = hoverOutlineOnly ? hovered && !disabled : true;

  return (
    <motion.button
      type={type}
      whileHover={disabled ? undefined : { y: -2, scale: 1.01 }}
      whileTap={disabled ? undefined : { scale: 0.985 }}
      onHoverStart={() => setHovered(true)}
      onHoverEnd={() => {
        setHovered(false);
        pointer.handleLeave();
      }}
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      onClick={onClick}
      disabled={disabled}
      className={`group relative inline-flex h-10 items-center gap-2 overflow-hidden border px-3.5 font-display text-[9px] uppercase tracking-[0.26em] transition-[transform,border-color,background,opacity] duration-300 ${
        disabled ? "cursor-not-allowed opacity-45" : ""
      } ${className}`}
      style={{
        borderColor: showSurface ? (secondary ? "rgba(255,255,255,0.16)" : `${accent}88`) : "transparent",
        background: showSurface ? (secondary ? "rgba(255,255,255,0.035)" : `${accent}16`) : "transparent",
        color: "#f5f7fa",
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1000,
        clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)",
      }}
    >
      <span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: `radial-gradient(circle at 18% 22%, ${accent}36 0%, transparent 68%)` }} />
      <span className="pointer-events-none absolute inset-x-3 bottom-0 h-px origin-left scale-x-0 bg-white/75 transition-transform duration-300 group-hover:scale-x-100" />
      <span className="relative z-10 inline-flex items-center gap-2">
        {icon}
        {children}
      </span>
    </motion.button>
  );
}

function ManagementEmptyState({
  accent,
  icon,
  title,
  description,
  action,
}: {
  accent: string;
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  const pointer = useInteractivePointer({
    maxShiftRatio: 0.012,
    maxShift: 4,
    maxRotate: 1.6,
  });

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ y: -4, scale: 1.01 }}
      className="flex min-h-[14rem] flex-col items-center justify-center border border-dashed border-white/12 bg-white/[0.015] px-6 py-10 text-center"
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1200,
        clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)",
      }}
    >
      <div className="flex h-14 w-14 items-center justify-center border border-white/10 bg-white/[0.03]" style={{ color: accent }}>
        {icon}
      </div>
      <div className="mt-4 font-display text-lg uppercase tracking-[0.16em] text-white">{title}</div>
      <div className="mt-2 max-w-xl text-sm leading-6 text-white/48">{description}</div>
      {action ? <div className="mt-5">{action}</div> : null}
    </motion.div>
  );
}

function ManagementPagination({
  accent,
  page,
  pageSize,
  totalItems,
  totalPages: totalPagesProp,
  rangeStart,
  rangeEnd,
  onPageChange,
}: {
  accent: string;
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages?: number;
  rangeStart?: number;
  rangeEnd?: number;
  onPageChange: (page: number) => void;
}) {
  const totalPages = totalPagesProp ?? Math.max(1, Math.ceil(totalItems / pageSize));
  const start = rangeStart ?? (totalItems === 0 ? 0 : (page - 1) * pageSize + 1);
  const end = rangeEnd ?? (totalItems === 0 ? 0 : Math.min(page * pageSize, totalItems));

  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.06] pt-4">
      <div className="text-[10px] uppercase tracking-[0.28em] text-white/34">
        Showing {start}-{end} / {totalItems}
      </div>
      <div className="flex items-center gap-2">
        <ManagementActionButton accent={accent} secondary disabled={page <= 1} onClick={() => onPageChange(page - 1)} className="h-9 px-3">
          Prev
        </ManagementActionButton>
        <div className="inline-flex h-9 min-w-[4.8rem] items-center justify-center border border-white/10 bg-white/[0.03] px-3 text-[10px] uppercase tracking-[0.26em] text-white/72">
          {page} / {totalPages}
        </div>
        <ManagementActionButton accent={accent} secondary disabled={page >= totalPages} onClick={() => onPageChange(page + 1)} className="h-9 px-3">
          Next
        </ManagementActionButton>
      </div>
    </div>
  );
}

function ManagementModal({
  open,
  accent,
  kicker,
  title,
  onClose,
  widthClass = "max-w-2xl",
  children,
}: {
  open: boolean;
  accent: string;
  kicker: string;
  title: string;
  onClose: () => void;
  widthClass?: string;
  children: ReactNode;
}) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!mounted) return null;

  return createPortal(
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-[140] grid place-items-center overflow-y-auto bg-black/84 px-4 py-8 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="pointer-events-none absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{
              opacity: 1,
              background: `radial-gradient(circle at 50% 18%, ${accent}26 0%, rgba(8,10,15,0.88) 42%, rgba(7,8,11,0.96) 100%)`,
            }}
            exit={{ opacity: 0 }}
          />
          <motion.div
            initial={{ opacity: 0, y: 38, scale: 0.955, filter: "blur(10px)" }}
            animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)", transition: { duration: 0.34, ease: [0.16, 1, 0.3, 1] } }}
            exit={{ opacity: 0, y: 24, scale: 0.975, filter: "blur(8px)", transition: { duration: 0.22 } }}
            className={`relative w-full ${widthClass} max-h-[calc(100dvh-2.5rem)] overflow-y-auto`}
            onClick={(event) => event.stopPropagation()}
          >
            <div
              className="cut-panel relative overflow-hidden border border-white/10 bg-[#080a0f] p-5 shadow-[0_40px_120px_rgba(0,0,0,0.56)]"
              style={{ "--panel-accent": accent } as CSSProperties}
            >
              <motion.div
                className="pointer-events-none absolute inset-x-0 top-0 h-px"
                initial={{ opacity: 0.2, scaleX: 0.4 }}
                animate={{ opacity: 0.75, scaleX: 1 }}
                transition={{ duration: 0.45, delay: 0.08 }}
                style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
              />
              <button
                type="button"
                onClick={onClose}
                className="absolute right-5 top-5 flex h-10 w-10 items-center justify-center border border-white/10 bg-white/[0.03] text-white/68 transition-colors duration-300 hover:border-white/20 hover:text-white"
              >
                <X size={16} />
              </button>
              <div className="mb-5 pr-12">
                <div className="text-[10px] uppercase tracking-[0.34em] text-white/38">{kicker}</div>
                <div className="mt-2 font-display text-2xl uppercase tracking-[0.16em] text-white">{title}</div>
              </div>
              {children}
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>,
    document.body,
  );
}

function getAssistantDocumentTypeAccent(documentType: AssistantDocumentType, chapter: Chapter) {
  switch (documentType) {
    case "FIM":
      return chapter.secondary;
    case "TSM":
      return "#ff8b62";
    case "SRM":
      return "#8fd6ff";
    case "IPC":
      return "#dca6ff";
    default:
      return chapter.accent;
  }
}

function matchesAssistantFilters(
  source: AssistantSource,
  documentTypeFilter: AssistantDocumentType | "ALL",
  aircraftModelFilter: string,
) {
  const matchesDocumentType = documentTypeFilter === "ALL" ? true : source.documentType === documentTypeFilter;
  const matchesAircraftModel = aircraftModelFilter === "ALL" ? true : source.aircraftModel === aircraftModelFilter;

  return matchesDocumentType && matchesAircraftModel;
}

function renderHighlightedPreviewLine(line: string, highlightText: string) {
  const normalizedLine = line.toLowerCase();
  const normalizedHighlight = highlightText.trim().toLowerCase();
  const index = normalizedHighlight ? normalizedLine.indexOf(normalizedHighlight) : -1;

  if (index === -1) {
    return line;
  }

  const endIndex = index + highlightText.length;

  return (
    <>
      {line.slice(0, index)}
      <span className="rounded-sm bg-[#fff17b]/60 px-1 text-[#121316]">{line.slice(index, endIndex)}</span>
      {line.slice(endIndex)}
    </>
  );
}

function CitationTag({
  citation,
  accent,
  secondary,
  active,
  onClick,
}: {
  citation: AssistantCitation;
  accent: string;
  secondary: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <motion.button
      type="button"
      whileHover={{ y: -3, scale: 1.035 }}
      whileTap={{ scale: 0.985 }}
      onClick={onClick}
      className={`group relative inline-flex items-center gap-2 overflow-hidden border px-2.5 py-1 text-[10px] uppercase tracking-[0.24em] transition-[border-color,background,transform,color,box-shadow] duration-250 ${
        active ? "text-white" : "text-white/72"
      }`}
      style={{
        borderColor: active ? `${secondary}aa` : `${accent}55`,
        background: active ? `${secondary}18` : `${accent}0f`,
        clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)",
        boxShadow: active ? `0 0 20px ${secondary}22` : `0 0 0 rgba(0,0,0,0)`,
      }}
    >
      <span
        className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-250 group-hover:opacity-100"
        style={{ background: `radial-gradient(circle at 18% 28%, ${active ? secondary : accent}40 0%, transparent 72%)` }}
      />
      <span
        className="pointer-events-none absolute inset-x-2 bottom-0 h-px origin-left scale-x-0 transition-transform duration-250 group-hover:scale-x-100"
        style={{ background: `linear-gradient(90deg, transparent, ${active ? secondary : accent}, transparent)` }}
      />
      <motion.span
        className="relative z-10 h-[5px] w-[5px] rounded-full"
        style={{ background: active ? secondary : accent, boxShadow: `0 0 12px ${active ? secondary : accent}` }}
        whileHover={{ scale: 1.25 }}
      />
      <span className="relative z-10">{citation.label}</span>
    </motion.button>
  );
}

function formatAssistantHistoryTime(timestamp: number) {
  const diff = Date.now() - timestamp;

  if (diff < 60 * 1000) {
    return "刚刚";
  }

  if (diff < 60 * 60 * 1000) {
    return `${Math.max(1, Math.floor(diff / (60 * 1000)))} 分钟前`;
  }

  if (diff < 24 * 60 * 60 * 1000) {
    return `${Math.floor(diff / (60 * 60 * 1000))} 小时前`;
  }

  return `${Math.floor(diff / (24 * 60 * 60 * 1000))} 天前`;
}

function buildAssistantHistoryPreview(conversation: AssistantConversation) {
  return (
    conversation.messages
      .find((message) => message.role === "assistant")
      ?.segments?.map((segment) => segment.text)
      .join(" ")
      .slice(0, 116) ?? ""
  );
}

function createInitialAssistantHistory(): AssistantHistoryItem[] {
  return [];
}

function ChatMessageList({
  chapter,
  messages,
  citationsById,
  selectedCitationId,
  onCitationClick,
  emptyState,
}: {
  chapter: Chapter;
  messages: AssistantMessage[];
  citationsById: Map<string, AssistantCitation>;
  selectedCitationId: string | null;
  onCitationClick: (citationId: string) => void;
  emptyState: ReactNode;
}) {
  if (messages.length === 0) {
    return <div className="flex h-full min-h-0 items-center justify-center">{emptyState}</div>;
  }

  return (
    <div className="grid gap-3 pb-2">
      {messages.map((message, index) =>
        message.role === "user" ? (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, x: 28 }}
            animate={{ opacity: 1, x: 0, transition: { duration: 0.38, delay: 0.04 * index } }}
            className="ml-auto max-w-[48rem]"
          >
            <div className="rounded-[1.6rem] border border-white/10 bg-white/[0.04] px-5 py-4 shadow-[0_18px_50px_rgba(0,0,0,0.28)]">
              <div className="mb-2 text-[10px] uppercase tracking-[0.32em] text-white/34">用户提问</div>
              <div className="text-[15px] leading-7 text-white/86">{message.content}</div>
            </div>
          </motion.div>
        ) : (
          <AssistantAnswerCard
            key={message.id}
            chapter={chapter}
            message={message}
            citationsById={citationsById}
            selectedCitationId={selectedCitationId}
            onCitationClick={onCitationClick}
            index={index}
          />
        ),
      )}
    </div>
  );
}

function AssistantAnswerCard({
  chapter,
  message,
  citationsById,
  selectedCitationId,
  onCitationClick,
  index,
}: {
  chapter: Chapter;
  message: AssistantMessage;
  citationsById: Map<string, AssistantCitation>;
  selectedCitationId: string | null;
  onCitationClick: (citationId: string) => void;
  index: number;
}) {
  const pointer = useInteractivePointer({
    maxShiftRatio: 0.01,
    maxShift: 4,
    maxRotate: 1.15,
  });
  const glow = useMotionTemplate`radial-gradient(150px circle at ${pointer.glowX}px ${pointer.glowY}px, ${chapter.accent}22 0%, transparent 72%)`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0, transition: { duration: 0.42, delay: 0.06 * index } }}
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ y: -5, scale: 1.006 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1400,
      }}
      className="group mr-auto max-w-[56rem]"
    >
      <motion.span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-220 group-hover:opacity-100" style={{ background: glow }} />
      <div className="cut-panel relative overflow-hidden border border-white/10 bg-black/[0.045] px-5 py-5 shadow-panel" style={{ "--panel-accent": chapter.accent } as CSSProperties}>
        <div className="mb-3 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center border border-white/10 bg-white/[0.03]">
            <Bot size={16} color={chapter.accent} />
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.32em] text-white/34">依据回答</div>
            <div className="mt-1 font-display text-sm uppercase tracking-[0.16em] text-white/88">航空文档智能问答检索</div>
          </div>
        </div>
        {(() => {
          const segments = message.segments ?? [];
          const isUnifiedAnswer = segments.length === 1 && segments[0].kind === "paragraph" && segments[0].citationIds.length > 1;

          if (isUnifiedAnswer) {
            const seg = segments[0];
            return (
              <div className="grid gap-4">
                <div className="border border-white/[0.07] bg-white/[0.018] px-4 py-4" style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}>
                  <div className="text-[14px] leading-8 text-white/82 whitespace-pre-line">{seg.text}</div>
                </div>
                <div className="border-t border-white/[0.06] pt-3">
                  <div className="mb-2 text-[10px] uppercase tracking-[0.3em] text-white/34">参考来源</div>
                  <div className="flex flex-wrap gap-2">
                    {seg.citationIds
                      .map((citationId) => citationsById.get(citationId))
                      .filter((citation): citation is AssistantCitation => Boolean(citation))
                      .map((citation) => (
                        <CitationTag
                          key={citation.id}
                          citation={citation}
                          accent={getAssistantDocumentTypeAccent(citation.documentType, chapter)}
                          secondary={chapter.secondary}
                          active={selectedCitationId === citation.id}
                          onClick={() => onCitationClick(citation.id)}
                        />
                      ))}
                  </div>
                </div>
              </div>
            );
          }

          return (
            <div className="grid gap-3">
              {segments.map((segment, segmentIndex) => (
                <div key={segment.id} className="border border-white/[0.07] bg-white/[0.018] px-4 py-3" style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}>
                  <div className="flex flex-wrap items-start gap-3">
                    <div className="mt-[2px] shrink-0 font-display text-xs uppercase tracking-[0.24em]" style={{ color: segment.kind === "note" ? chapter.secondary : chapter.accent }}>
                      {segment.kind === "step" ? `步骤 0${segmentIndex + 1}` : segment.kind === "note" ? "说明" : "依据"}
                    </div>
                    <div className="min-w-0 flex-1 text-[14px] leading-7 text-white/80">
                      <span>{segment.text} </span>
                      <span className="inline-flex flex-wrap gap-2 align-middle">
                        {segment.citationIds
                          .map((citationId) => citationsById.get(citationId))
                          .filter((citation): citation is AssistantCitation => Boolean(citation))
                          .map((citation) => (
                            <CitationTag
                              key={citation.id}
                              citation={citation}
                              accent={getAssistantDocumentTypeAccent(citation.documentType, chapter)}
                              secondary={chapter.secondary}
                              active={selectedCitationId === citation.id}
                              onClick={() => onCitationClick(citation.id)}
                            />
                          ))}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          );
        })()}
      </div>
    </motion.div>
  );
}

function ChatComposer({
  chapter,
  value,
  onChange,
  onSubmit,
}: {
  chapter: Chapter;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}) {
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useLayoutEffect(() => {
    const node = textareaRef.current;
    if (!node) return;

    const minHeight = 28;
    const maxHeight = minHeight * 4;

    node.style.height = `${minHeight}px`;
    const nextHeight = Math.min(Math.max(node.scrollHeight, minHeight), maxHeight);
    node.style.height = `${nextHeight}px`;
    node.style.overflowY = node.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [value]);

  useEffect(() => {
    const handleResize = () => {
      const node = textareaRef.current;
      if (!node) return;

      const minHeight = 28;
      const maxHeight = minHeight * 4;

      node.style.height = `${minHeight}px`;
      const nextHeight = Math.min(Math.max(node.scrollHeight, minHeight), maxHeight);
      node.style.height = `${nextHeight}px`;
      node.style.overflowY = node.scrollHeight > maxHeight ? "auto" : "hidden";
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <form
      className="mt-4 mb-2 shrink-0"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div
        className={`cut-panel relative overflow-hidden border bg-[#090b11]/94 px-4 py-3.5 shadow-[0_30px_90px_rgba(0,0,0,0.42)] backdrop-blur-md transition-colors duration-250 ${
          focused ? "border-white/16" : "border-white/12"
        }`}
        style={{ "--panel-accent": chapter.accent } as CSSProperties}
      >
        <span className="pointer-events-none absolute inset-x-5 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, ${chapter.secondary}, transparent)` }} />
        <div className="pointer-events-none absolute inset-x-4 bottom-3 flex items-center gap-2 opacity-80">
          <span className="h-px flex-1 bg-white/[0.05]" />
          <span className="h-px w-12" style={{ background: `linear-gradient(90deg, transparent, ${chapter.secondary}, transparent)` }} />
        </div>
        <div className="pointer-events-none absolute right-4 top-4 flex items-center gap-2 opacity-75">
          <span className="h-px w-8 bg-white/[0.08]" />
          <span className="h-[5px] w-[5px]" style={{ background: chapter.accent, boxShadow: `0 0 14px ${chapter.accent}` }} />
        </div>
        <div className="flex items-end gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center border border-white/10 bg-white/[0.035]">
            <Search size={16} color={chapter.accent} />
          </div>
          <div className="min-w-0 flex-1 pb-1">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  if (value.trim()) onSubmit();
                }
              }}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              rows={1}
              placeholder="请输入航空维修问题，系统将结合手册、AMM、FIM、TSM 等文档给出带依据的回答。"
              className="panel-scroll w-full resize-none bg-transparent text-[15px] leading-7 text-white outline-none placeholder:text-white/24"
              style={{ minHeight: "28px", maxHeight: "112px" }}
            />
          </div>
          <ManagementActionButton accent={chapter.accent} type="submit" className="h-11 justify-center px-4" icon={<ArrowRight size={14} />}>
            提问
          </ManagementActionButton>
        </div>
      </div>
    </form>
  );
}

function AssistantHistoryRow({
  chapter,
  item,
  active,
  onSelect,
}: {
  chapter: Chapter;
  item: AssistantHistoryItem;
  active: boolean;
  onSelect: (item: AssistantHistoryItem) => void;
}) {
  const pointer = useInteractivePointer({
    maxShiftRatio: 0.008,
    maxShift: 2.4,
    maxRotate: 0.65,
  });
  const glow = useMotionTemplate`radial-gradient(128px circle at ${pointer.glowX}px ${pointer.glowY}px, ${
    active ? `${chapter.secondary}2c` : `${chapter.accent}18`
  } 0%, transparent 72%)`;

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={active ? { y: -2.5, scale: 1.012 } : { x: 4, y: -1 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1200,
        borderColor: active ? `${chapter.secondary}6a` : "transparent",
        background: active
          ? `linear-gradient(135deg, ${chapter.secondary}18 0%, rgba(255,255,255,0.062) 34%, ${chapter.accent}20 100%)`
          : "transparent",
        boxShadow: active
          ? `0 0 0 1px ${chapter.secondary}20, 0 0 16px ${chapter.secondary}12, 0 0 28px ${chapter.accent}10, inset 0 0 0 1px rgba(255,255,255,0.04)`
          : undefined,
        clipPath: "polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%)",
      }}
      className={`group relative overflow-hidden border transition-[border-color,background-color,box-shadow] duration-220 ${
        active ? "" : "hover:bg-white/[0.024]"
      }`}
    >
      <motion.span
        className={`pointer-events-none absolute inset-0 transition-opacity duration-220 ${
          active ? "opacity-80 group-hover:opacity-100" : "opacity-0 group-hover:opacity-100"
        }`}
        style={{ background: glow }}
      />
      {active ? (
        <>
          <motion.span
            className="pointer-events-none absolute inset-0"
            style={{
              background: `linear-gradient(118deg, ${chapter.secondary}10 0%, rgba(255,255,255,0.016) 38%, ${chapter.accent}18 100%)`,
            }}
            animate={{ opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
          />
          <span
            className="pointer-events-none absolute inset-y-0 left-0 w-[4px]"
            style={{
              background: `linear-gradient(180deg, ${chapter.secondary}, ${chapter.accent})`,
              boxShadow: `0 0 16px ${chapter.secondary}, 0 0 26px ${chapter.accent}66`,
            }}
          />
          <motion.span
            className="pointer-events-none absolute inset-x-0 top-0 h-px"
            style={{ background: `linear-gradient(90deg, transparent, ${chapter.secondary}, ${chapter.accent}, transparent)` }}
            animate={{ x: ["-100%", "100%"] }}
            transition={{ duration: 2.2, repeat: Infinity, ease: "linear" }}
          />
          <span
            className="pointer-events-none absolute inset-x-4 bottom-0 h-px opacity-60"
            style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, transparent)` }}
          />
          <span
            className="pointer-events-none absolute right-0 top-0 h-5 w-5 opacity-80"
            style={{
              background: `linear-gradient(135deg, ${chapter.secondary}88, ${chapter.accent}00 74%)`,
              clipPath: "polygon(100% 0, 0 0, 100% 100%)",
            }}
          />
        </>
      ) : null}
      <button
        type="button"
        onClick={() => onSelect(item)}
        className="relative z-10 flex w-full min-w-0 items-center px-3 py-2.5 text-left transition-colors duration-200"
      >
        <div className="flex min-w-0 flex-1 items-center gap-2.5">
          <div
            className={`min-w-0 flex-1 truncate text-[13px] tracking-[0.01em] ${
              active ? "text-white drop-shadow-[0_0_12px_rgba(255,255,255,0.08)]" : "text-white/82 group-hover:text-white/92"
            }`}
          >
            {item.title || item.question}
          </div>
          <div
            className={`shrink-0 text-[10px] uppercase tracking-[0.18em] ${
              active ? "text-white/56 drop-shadow-[0_0_10px_rgba(255,255,255,0.05)]" : "text-white/28"
            }`}
          >
            {formatAssistantHistoryTime(item.createdAt)}
          </div>
        </div>
      </button>
    </motion.div>
  );
}

function AssistantLaunchOverlay({
  launchState,
  chapter,
}: {
  launchState: AssistantLaunchState;
  chapter: Chapter;
}) {
  const viewportWidth = typeof window === "undefined" ? 1440 : window.innerWidth;
  const viewportHeight = typeof window === "undefined" ? 900 : window.innerHeight;
  const horizontalInset = viewportWidth >= 1280 ? 40 : viewportWidth >= 768 ? 32 : 20;
  const targetLeft = horizontalInset;
  const targetTop = viewportWidth >= 1280 ? 108 : viewportWidth >= 768 ? 104 : 102;
  const targetWidth = Math.max(viewportWidth - horizontalInset * 2, launchState.width);
  const targetHeight = Math.max(viewportHeight - targetTop - (viewportWidth >= 768 ? 18 : 16), 460);
  const startX = launchState.left - targetLeft;
  const startY = launchState.top - targetTop;
  const startScaleX = Math.max(launchState.width / targetWidth, 0.08);
  const startScaleY = Math.max(launchState.height / targetHeight, 0.08);

  return (
    <motion.div
      className="pointer-events-none fixed inset-0 z-[90] overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.2 } }}
    >
      <motion.div
        className="absolute inset-0"
        initial={{ opacity: 0.65 }}
        animate={{
          opacity: 1,
          background: `radial-gradient(circle at ${launchState.left + launchState.width / 2}px ${
            launchState.top + launchState.height / 2
          }px, ${chapter.accent}26 0%, rgba(7,8,11,0.82) 38%, rgba(7,8,11,0.95) 100%)`,
        }}
        exit={{ opacity: 0 }}
      />

      <motion.div
        className="absolute overflow-hidden border border-white/12 bg-[#07090d]/86 backdrop-blur-md"
        style={{
          left: targetLeft,
          top: targetTop,
          width: targetWidth,
          height: targetHeight,
          transformOrigin: "top left",
          willChange: "transform, opacity",
          clipPath: "polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 0 100%)",
          boxShadow: `0 0 0 1px ${chapter.accent}18, 0 18px 80px rgba(0,0,0,0.36), 0 0 56px ${chapter.accent}18`,
        }}
        initial={{
          x: startX,
          y: startY,
          scaleX: startScaleX,
          scaleY: startScaleY,
          opacity: 0.96,
        }}
        animate={{
          x: 0,
          y: 0,
          scaleX: 1,
          scaleY: 1,
          opacity: 1,
          transition: { duration: 0.62, ease: [0.19, 1, 0.22, 1] },
        }}
        exit={{
          opacity: 0,
          scaleX: 1.008,
          scaleY: 1.008,
          filter: "blur(4px)",
          transition: { duration: 0.18, ease: [0.7, 0, 0.84, 0] },
        }}
      >
        <span
          className="absolute inset-x-4 top-0 h-px opacity-90 md:inset-x-6"
          style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, ${chapter.secondary}, transparent)` }}
        />
        <motion.span
          className="absolute inset-y-0 left-[-16%] w-[18%]"
          style={{ background: `linear-gradient(90deg, transparent 0%, ${chapter.accent}16 48%, transparent 100%)` }}
          animate={{ x: ["0%", "680%"] }}
          transition={{ duration: 0.95, ease: "easeInOut" }}
        />
        <span
          className="absolute inset-x-4 bottom-0 h-px opacity-70 md:inset-x-6"
          style={{ background: `linear-gradient(90deg, transparent, ${chapter.secondary}, ${chapter.accent}, transparent)` }}
        />
        <div className="absolute inset-0 bg-grid bg-[size:42px_42px] opacity-[0.05]" />
        <div className="absolute inset-0 scan-overlay opacity-[0.08]" />

        <div className="relative z-10 flex h-full flex-col px-4 py-4 md:px-6 md:py-5">
          <motion.div
            initial={{ opacity: 0, y: 10, filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.28, delay: 0.08 } }}
            className="flex items-center justify-between gap-4"
          >
            <div className="min-w-0">
              <div className="text-[10px] uppercase tracking-[0.32em] text-white/34">Dock 00 / 搜索框展开中</div>
              <div className="mt-2 font-display text-sm uppercase tracking-[0.16em] text-white/82 md:text-base">航空文档智能问答检索</div>
            </div>
            <div className="hidden items-center gap-2 md:flex">
              {Array.from({ length: 3 }).map((_, index) => (
                <span
                  key={`launch-pill-${index}`}
                  className="h-7 w-24 border border-white/10 bg-white/[0.03]"
                  style={{ clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)" }}
                />
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.985, y: 14 }}
            animate={{ opacity: 1, scale: 1, y: 0, transition: { duration: 0.36, delay: 0.14, ease: [0.16, 1, 0.3, 1] } }}
            className="grid flex-1 gap-4 py-4 xl:grid-cols-[17rem_minmax(0,1fr)_22rem]"
          >
            <div className="grid content-start gap-3">
              {Array.from({ length: 5 }).map((_, index) => (
                <motion.div
                  key={`launch-side-${index}`}
                  className="border border-white/10 bg-white/[0.025] px-4 py-4"
                  style={{ clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)" }}
                  initial={{ opacity: 0, x: -14, filter: "blur(6px)" }}
                  animate={{ opacity: 1, x: 0, filter: "blur(0px)", transition: { duration: 0.24, delay: 0.18 + 0.05 * index } }}
                >
                  <div className="h-2.5 w-24 bg-white/12" />
                  <div className="mt-3 h-8 w-full bg-white/[0.06]" />
                  <div className="mt-2 h-8 w-[68%] bg-white/[0.04]" />
                </motion.div>
              ))}
            </div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0, transition: { duration: 0.3, delay: 0.2 } }} className="grid min-h-0 gap-3">
              <div
                className="border border-white/10 bg-white/[0.03] px-4 py-4"
                style={{ clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)" }}
              >
                <div className="h-2.5 w-40 bg-white/12" />
                <div className="mt-4 h-10 w-[84%] bg-white/[0.08]" />
                <div className="mt-2 h-10 w-[72%] bg-white/[0.06]" />
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="h-7 w-28 border border-white/10 bg-white/[0.03]" style={{ clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)" }} />
                  <span className="h-7 w-28 border border-white/10 bg-white/[0.03]" style={{ clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)" }} />
                </div>
              </div>

              <div
                className="grid min-h-0 flex-1 gap-3 border border-white/10 bg-white/[0.02] px-4 py-4"
                style={{ clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)" }}
              >
                {Array.from({ length: 3 }).map((_, index) => (
                  <div
                    key={`launch-chat-${index}`}
                    className={`grid gap-2 border border-white/8 px-4 py-4 ${index === 0 ? "ml-auto w-[68%] bg-white/[0.04]" : "w-full bg-black/[0.08]"}`}
                    style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
                  >
                    <div className={`h-2.5 ${index === 0 ? "w-20 bg-white/18" : "w-28 bg-white/12"}`} />
                    <div className="h-8 w-full bg-white/[0.07]" />
                    <div className="h-8 w-[78%] bg-white/[0.05]" />
                    {index !== 0 ? (
                      <div className="flex gap-2">
                        <span className="h-7 w-24 border border-white/10 bg-white/[0.03]" style={{ clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)" }} />
                        <span className="h-7 w-24 border border-white/10 bg-white/[0.03]" style={{ clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)" }} />
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>

              <div className="relative overflow-hidden rounded-[1.7rem] border border-white/12 bg-[#090b11]/92 px-4 py-4 shadow-[0_18px_60px_rgba(0,0,0,0.24)]">
                <span
                  className="pointer-events-none absolute inset-x-5 top-0 h-px"
                  style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, ${chapter.secondary}, transparent)` }}
                />
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/[0.035]">
                    <Search size={14} color={chapter.accent} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="h-2.5 w-[88%] bg-white/[0.1]" />
                    <div className="mt-2 h-2.5 w-[64%] bg-white/[0.06]" />
                  </div>
                  <div className="h-10 w-20 border border-white/10 bg-white/[0.03]" style={{ clipPath: "polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%)" }} />
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 14 }}
              animate={{ opacity: 1, x: 0, transition: { duration: 0.28, delay: 0.22 } }}
              className="hidden content-start gap-3 xl:grid"
            >
              <div
                className="border border-white/10 bg-white/[0.03] px-4 py-4"
                style={{ clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)" }}
              >
                <div className="h-2.5 w-28 bg-white/12" />
                <div className="mt-4 h-9 w-[72%] bg-white/[0.07]" />
                <div className="mt-2 h-2.5 w-[56%] bg-white/[0.05]" />
                <div className="mt-3 h-2.5 w-[84%] bg-white/[0.05]" />
              </div>
              <div className="paper-shell relative min-h-[18rem] overflow-hidden rounded-[1.55rem] px-5 py-5 text-[#17181b]">
                <div className="relative z-10 h-2.5 w-20 bg-black/10" />
                <div className="relative z-10 mt-4 h-4 w-[74%] bg-black/12" />
                <div className="relative z-10 mt-7 grid gap-3">
                  <div className="h-3 w-[86%] bg-black/10" />
                  <div className="h-3 w-[78%] bg-[#ffe26a]/42" />
                  <div className="h-3 w-[82%] bg-black/10" />
                  <div className="h-3 w-[58%] bg-black/10" />
                </div>
                <div className="pointer-events-none absolute left-[12%] top-[45%] h-[12%] w-[72%] border border-[#ffe26a]/80 bg-[#fff17b]/18 shadow-[0_0_24px_rgba(255,226,106,0.18)]" />
              </div>
            </motion.div>
          </motion.div>
        </div>
      </motion.div>
    </motion.div>
  );
}

function PdfPreviewPane({
  source,
  citation,
  pageNumber,
  onPageChange,
}: {
  source: AssistantSource | null;
  citation: AssistantCitation | null;
  pageNumber: number;
  onPageChange: (pageNumber: number) => void;
}) {
  if (!source) {
    return (
      <div className="flex min-h-[20rem] items-center justify-center border border-white/8 bg-white/[0.02] px-4">
        <div className="text-center">
          <div className="text-[10px] uppercase tracking-[0.3em] text-white/32">证据预览</div>
          <div className="mt-3 text-sm leading-7 text-white/54">点击回答中的引用标签后，这里会打开对应证据页并跳转到相关位置。</div>
        </div>
      </div>
    );
  }

  const currentPage = source.pages.find((page) => page.pageNumber === pageNumber) ?? source.pages[0];
  const currentIndex = source.pages.findIndex((page) => page.pageNumber === currentPage.pageNumber);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < source.pages.length - 1;
  const activeHighlight = citation && citation.page === currentPage.pageNumber ? citation.highlightText : "";

  return (
    <div className="grid gap-3">
      <div className="paper-shell relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-[1.8rem] px-6 py-6 text-[#17181b]">
        <div className="relative z-10 grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-start">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.28em] text-[#5c5d61]/70">模拟 PDF 预览</div>
            <div className="text-[10px] uppercase tracking-[0.28em] text-[#5c5d61]/70">{source.documentType}</div>
            <div className="mt-2 text-xl font-semibold">{source.documentTitle}</div>
            <div className="mt-2 text-sm text-[#57585c]">{currentPage.sectionTitle}</div>
          </div>
          <div className="grid gap-3 md:justify-items-end">
            <div className="text-left md:text-right">
              <div className="text-[10px] uppercase tracking-[0.28em] text-[#66686d]/70">章节</div>
              <div className="mt-2 text-sm font-medium">{source.chapter}</div>
            </div>
          </div>
        </div>
        <div className="relative z-10 mt-6 flex-1">
          <div className="grid gap-3 text-[14px] leading-7 text-[#232427]">
          {currentPage.lines.map((line, index) => (
            <div key={`${currentPage.pageNumber}-${index}`}>{renderHighlightedPreviewLine(line, activeHighlight)}</div>
          ))}
          </div>
        </div>
        <div className="relative z-10 mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-black/10 pt-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.28em] text-[#66686d]/70">页码</div>
            <div className="mt-1 font-display text-sm uppercase tracking-[0.14em] text-[#3f4248]">
              第 {currentPage.pageNumber} 页 / 共 {source.pageCount} 页
            </div>
          </div>
          <div className="flex flex-wrap gap-2 md:justify-end">
            <motion.button
              type="button"
              whileHover={hasPrev ? { y: -2, scale: 1.01 } : undefined}
              whileTap={hasPrev ? { scale: 0.985 } : undefined}
              disabled={!hasPrev}
              onClick={() => hasPrev && onPageChange(source.pages[currentIndex - 1].pageNumber)}
              className={`inline-flex h-9 items-center gap-2 overflow-hidden border px-3 font-display text-[9px] uppercase tracking-[0.22em] transition-[transform,border-color,background,opacity] duration-300 ${
                hasPrev ? "" : "cursor-not-allowed opacity-35"
              }`}
              style={{
                borderColor: "rgba(22,21,18,0)",
                background: "rgba(255,255,255,0)",
                color: "#17181bcc",
                clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)",
              }}
              onHoverStart={(event) => {
                const target = event.currentTarget as HTMLButtonElement;
                target.style.borderColor = "rgba(22,21,18,0.14)";
                target.style.background = "rgba(255,255,255,0.36)";
              }}
              onHoverEnd={(event) => {
                const target = event.currentTarget as HTMLButtonElement;
                target.style.borderColor = "rgba(22,21,18,0)";
                target.style.background = "rgba(255,255,255,0)";
              }}
            >
              <ChevronLeft size={12} />
              上一页
            </motion.button>
            <motion.button
              type="button"
              whileHover={hasNext ? { y: -2, scale: 1.01 } : undefined}
              whileTap={hasNext ? { scale: 0.985 } : undefined}
              disabled={!hasNext}
              onClick={() => hasNext && onPageChange(source.pages[currentIndex + 1].pageNumber)}
              className={`inline-flex h-9 items-center gap-2 overflow-hidden border px-3 font-display text-[9px] uppercase tracking-[0.22em] transition-[transform,border-color,background,opacity] duration-300 ${
                hasNext ? "" : "cursor-not-allowed opacity-35"
              }`}
              style={{
                borderColor: "rgba(22,21,18,0)",
                background: "rgba(255,255,255,0)",
                color: "#17181bcc",
                clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)",
              }}
              onHoverStart={(event) => {
                const target = event.currentTarget as HTMLButtonElement;
                target.style.borderColor = "rgba(22,21,18,0.14)";
                target.style.background = "rgba(255,255,255,0.36)";
              }}
              onHoverEnd={(event) => {
                const target = event.currentTarget as HTMLButtonElement;
                target.style.borderColor = "rgba(22,21,18,0)";
                target.style.background = "rgba(255,255,255,0)";
              }}
            >
              下一页
              <ChevronRight size={12} />
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  );
}

function EvidencePanel({
  chapter,
  citations: _citations,
  selectedCitationId: _selectedCitationId,
  onSelectCitation: _onSelectCitation,
  onClose,
  mode: _mode,
  onModeChange: _onModeChange,
  source,
  citation,
  previewPage,
  onPreviewPageChange,
}: {
  chapter: Chapter;
  citations: AssistantCitation[];
  selectedCitationId: string | null;
  onSelectCitation: (citationId: string) => void;
  onClose?: () => void;
  mode: "all" | "active";
  onModeChange: (mode: "all" | "active") => void;
  source: AssistantSource | null;
  citation: AssistantCitation | null;
  previewPage: number;
  onPreviewPageChange: (pageNumber: number) => void;
}) {
  return (
    <CutPanel
      accent={chapter.secondary}
      title="Evidence"
      className="assistant-evidence-panel flex h-full min-h-0 flex-col overflow-auto px-2 pt-1"
      headerClassName="mb-1"
      actions={
        onClose ? (
          <ManagementActionButton accent={chapter.secondary} secondary onClick={onClose} className="h-8 -mt-1 px-3 text-[8px] tracking-[0.2em]">
            <X size={12} />
            Close
          </ManagementActionButton>
        ) : undefined
      }
    >
      <div className="panel-scroll evidence-scroll grid min-h-0 flex-1 gap-4 overflow-y-auto pr-1 md:pr-1.5">
        <PdfPreviewPane source={source} citation={citation} pageNumber={previewPage} onPageChange={onPreviewPageChange} />
      </div>
    </CutPanel>
  );
}

function SketchDocCard({
  accent,
  type,
  state,
  title,
  description,
  footer,
  delay = 0,
}: {
  accent: string;
  type: string;
  state: string;
  title: string;
  description: string;
  footer: string;
  delay?: number;
}) {
  const pointer = useInteractivePointer();
  const glow = useMotionTemplate`radial-gradient(240px circle at ${pointer.glowX}px ${pointer.glowY}px, ${accent}36 0%, transparent 76%)`;
  const titleWithExtension = `${title}.${type}`;

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ y: -6, scale: 1.008 }}
      initial={{ opacity: 0, y: 28, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1, transition: { duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] } }}
      className="group relative overflow-hidden border border-white/10 bg-white/[0.012] px-4 py-3.5 md:px-4 md:py-3.5"
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1400,
        clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 16px), calc(100% - 16px) 100%, 0 100%)",
      }}
    >
      <motion.span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <motion.div
        className="pointer-events-none absolute inset-x-4 top-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
        animate={{ x: ["-100%", "100%"] }}
        transition={{ duration: 3.2 + delay, repeat: Infinity, ease: "linear" }}
      />
      <div className="relative z-10 flex h-full flex-col gap-3">
        <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-start">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.28em] text-white/35">
              <span className="text-white/52">{type}</span> Source
            </div>
            <div className="mt-2 min-w-0 font-medium text-[1rem] leading-7 text-white md:text-[1.02rem]">{titleWithExtension}</div>
          </div>
          <div className="flex items-start md:justify-end">
            <TacticalTag accent={accent}>{state}</TacticalTag>
          </div>
        </div>
        <div className="text-[12px] leading-6 text-white/58">{description}</div>
        <div className="mt-auto grid gap-3 border-t border-white/[0.05] pt-3 md:grid-cols-[minmax(0,1fr)_5rem] md:items-center">
          <div className="text-[10px] uppercase tracking-[0.24em] text-white/40">{footer}</div>
          <div className="h-1.5 overflow-hidden bg-white/8 md:w-20 md:justify-self-end">
            <motion.div
              className="pointer-events-none h-px w-full opacity-70"
              style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }}
              animate={{ x: ["-100%", "100%"] }}
              transition={{ duration: 3.8 + delay, repeat: Infinity, ease: "linear" }}
            />
            <div className="relative -mt-px h-1.5 overflow-hidden">
              <motion.div
                className="h-full"
                style={{ background: `linear-gradient(90deg, ${accent}, ${accent}55)` }}
                initial={{ width: "28%" }}
                animate={{ width: ["28%", "82%", "46%"] }}
                transition={{ duration: 4.2 + delay, repeat: Infinity, ease: "easeInOut" }}
              />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function SketchSection({
  title,
  subtitle,
  description,
  accent,
  eyebrow,
  signal,
  titleStyle,
  delay = 0,
  children,
}: {
  title: string;
  subtitle: string;
  description: string;
  accent: string;
  eyebrow: string;
  signal: string;
  titleStyle?: CSSProperties;
  delay?: number;
  children: ReactNode;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 34, scale: 0.985 }}
      animate={{
        opacity: 1,
        y: 0,
        scale: 1,
        transition: { duration: 0.75, delay, ease: [0.16, 1, 0.3, 1] },
      }}
      style={{ "--panel-accent": accent } as CSSProperties}
      className="cut-panel sketch-section-panel relative flex h-full min-h-0 flex-col overflow-hidden border border-white/[0.14] bg-white/[0.018] p-4 shadow-panel md:p-5"
    >
      <div className="pointer-events-none absolute inset-x-4 top-4 flex items-center gap-3 opacity-75 md:inset-x-5">
        <span className="h-px w-16 bg-white/[0.1]" />
        <span className="h-px flex-1 bg-white/[0.045]" />
      </div>
      <div className="pointer-events-none absolute inset-y-5 right-4 hidden w-px bg-gradient-to-b from-white/[0.16] via-white/[0.04] to-transparent md:block" />
      <div className="mb-5 flex items-start justify-between gap-6 pt-4">
        <div className="max-w-[44rem]">
          <div className="flex items-center gap-3">
            <div className="text-[10px] uppercase tracking-[0.34em] text-white/34">{eyebrow}</div>
            <span className="h-px w-10" style={{ background: `linear-gradient(90deg, ${accent}, transparent)` }} />
          </div>
          <div
            className="mt-2 text-[clamp(1.66rem,2.55vw,2.18rem)] font-semibold uppercase tracking-[0.08em]"
            style={{
              fontFamily: '"Bahnschrift SemiBold", "Segoe UI Variable Display", "DengXian", "Microsoft YaHei UI", sans-serif',
              color: `color-mix(in srgb, white 64%, ${accent} 36%)`,
              textShadow: `0 0 24px ${accent}24`,
              ...titleStyle,
            }}
          >
            {title}
          </div>
          <div className="mt-1 text-[12px] leading-6 text-white/52">{subtitle}</div>
          {description ? <div className="mt-1 text-[12px] leading-6 text-white/36">{description}</div> : null}
        </div>
        <div className="hidden min-w-[8rem] shrink-0 md:mr-4 md:grid md:justify-items-end md:gap-2 xl:mr-6">
          <div className="text-[9px] uppercase tracking-[0.36em] text-white/26">{signal}</div>
          <div className="flex items-center gap-2">
            <span className="h-px w-12 bg-white/[0.08]" />
            <span className="h-[5px] w-[5px] rounded-full" style={{ background: accent, boxShadow: `0 0 16px ${accent}` }} />
            <span className="h-px w-6" style={{ background: `linear-gradient(90deg, ${accent}80, transparent)` }} />
          </div>
          <div className="grid w-[5rem] grid-cols-5 gap-1">
            {Array.from({ length: 5 }).map((_, index) => (
              <span
                key={`${signal}-${index}`}
                className="h-[2px]"
                style={{ background: index % 2 === 0 ? `${accent}55` : "rgba(255,255,255,0.12)" }}
              />
            ))}
          </div>
        </div>
      </div>
      <div className="panel-scroll stage-scroll relative min-h-0 flex-1 overflow-x-hidden overflow-y-auto pr-1 pb-1">
        {children}
      </div>
    </motion.section>
  );
}

function CoreDevice({ accent, secondary }: { accent: string; secondary: string }) {
  const pointer = useInteractivePointer();
  const halo = useMotionTemplate`radial-gradient(220px circle at ${pointer.glowX}px ${pointer.glowY}px, ${secondary}52 0%, transparent 72%)`;

  return (
    <motion.div
      className="group relative mx-auto flex aspect-square w-full max-w-[30rem] items-center justify-center"
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1600,
      }}
      whileHover={{ scale: 1.035 }}
    >
      <motion.div className="pointer-events-none absolute inset-[8%] rounded-full opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: halo }} />
      <motion.div
        className="absolute inset-[3%] border"
        style={{
          borderColor: `${accent}50`,
          clipPath: "polygon(16% 0, 84% 0, 100% 16%, 100% 84%, 84% 100%, 16% 100%, 0 84%, 0 16%)",
          boxShadow: `0 0 40px ${accent}30`,
        }}
        animate={{ rotate: 360 }}
        transition={{ duration: 22, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute inset-[12%] border"
        style={{ borderColor: `${secondary}70`, clipPath: "polygon(50% 0, 100% 50%, 50% 100%, 0 50%)" }}
        animate={{ rotate: -360, scale: [1, 1.03, 1] }}
        transition={{ duration: 16, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute inset-[22%] rounded-full border"
        style={{ borderColor: `${accent}60`, background: `radial-gradient(circle, ${accent}18, transparent 68%)` }}
        animate={{ scale: [0.94, 1.04, 0.94], opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 3.8, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute inset-[34%] border"
        style={{
          borderColor: `${secondary}80`,
          clipPath: "polygon(50% 0, 100% 50%, 50% 100%, 0 50%)",
          background: `linear-gradient(135deg, ${secondary}28, transparent 60%)`,
        }}
        animate={{ rotate: 360 }}
        transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute h-[9rem] w-[9rem] rounded-full"
        style={{
          background: `radial-gradient(circle at 30% 30%, ${secondary}, ${accent} 42%, rgba(0,0,0,0.1) 75%)`,
          boxShadow: `0 0 70px ${accent}66, 0 0 140px ${secondary}24`,
        }}
        animate={{ scale: [1, 1.08, 1], filter: ["brightness(1)", "brightness(1.18)", "brightness(1)"] }}
        transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="absolute inset-0">
        {[
          { label: "RAG", top: "8%", left: "50%" },
          { label: "INDEX", top: "50%", left: "88%" },
          { label: "TRACE", top: "88%", left: "50%" },
          { label: "FUSION", top: "50%", left: "12%" },
        ].map((item, index) => (
          <motion.div
            key={item.label}
            className="absolute -translate-x-1/2 -translate-y-1/2 border px-3 py-1 text-[10px] uppercase tracking-[0.34em] text-white/85"
            style={{ top: item.top, left: item.left, borderColor: `${accent}55`, background: "rgba(0,0,0,0.45)" }}
            animate={{ y: [0, index % 2 === 0 ? -12 : 12, 0] }}
            transition={{ duration: 4 + index, repeat: Infinity, ease: "easeInOut" }}
          >
            {item.label}
          </motion.div>
        ))}
      </div>
      {[0, 1, 2].map((item) => (
        <motion.div
          key={item}
          className="absolute h-3 w-3 rounded-full"
          style={{
            background: item === 1 ? secondary : accent,
            boxShadow: `0 0 18px ${item === 1 ? secondary : accent}`,
          }}
          animate={{
            x: [0, item === 0 ? 118 : item === 1 ? -124 : 54, 0],
            y: [0, item === 0 ? -88 : item === 1 ? 106 : 142, 0],
            scale: [0.8, 1.25, 0.8],
          }}
          transition={{ duration: 5 + item * 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}
    </motion.div>
  );
}

function UploadBay({ chapter, onFileSelect }: { chapter: Chapter; onFileSelect?: (file: File) => void }) {
  const pointer = useInteractivePointer();
  const glow = useMotionTemplate`radial-gradient(240px circle at ${pointer.glowX}px ${pointer.glowY}px, ${chapter.accent}52 0%, transparent 72%)`;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    if (!["pdf", "docx"].includes(ext)) return;
    onFileSelect?.(file);
  };

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={(e) => { pointer.handleLeave(e); setDragOver(false); }}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
      onClick={() => fileInputRef.current?.click()}
      whileHover={{ y: -8, scale: 1.008 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1400,
        clipPath: "polygon(0 0, 94% 0, 100% 12%, 100% 100%, 6% 100%, 0 88%)",
      }}
      className={`group relative flex min-h-[14.25rem] cursor-pointer flex-col items-center justify-center overflow-hidden border border-dashed px-4 text-center transition-colors ${dragOver ? "border-white/50 bg-white/[0.04]" : "border-white/20 bg-white/[0.008]"}`}
    >
      <input ref={fileInputRef} type="file" accept=".pdf,.docx" className="hidden" onChange={(e) => handleFile(e.target.files?.[0])} />
      <motion.div className="pointer-events-none absolute inset-0 opacity-[0.08]" style={{ backgroundImage: glow }} />
      <div className="absolute inset-0 bg-grid bg-[size:36px_36px] opacity-[0.08]" />
      <motion.div
        className="absolute h-28 w-28 rounded-full border border-white/10"
        animate={{ rotate: 360 }}
        transition={{ duration: 16, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute h-36 w-36 border"
        style={{ borderColor: `${chapter.secondary}35`, clipPath: "polygon(50% 0, 100% 50%, 50% 100%, 0 50%)" }}
        animate={{ rotate: -360 }}
        transition={{ duration: 24, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="relative z-10"
        animate={{
          y: [0, -12, 0],
          filter: ["drop-shadow(0 0 0px transparent)", `drop-shadow(0 0 24px ${chapter.accent})`, "drop-shadow(0 0 0px transparent)"],
        }}
        transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
        whileHover={{ scale: 1.18, rotate: -8 }}
      >
        <UploadCloud size={40} color={chapter.accent} />
      </motion.div>
      <motion.div className="mt-3 font-display text-lg uppercase tracking-[0.16em] text-white" whileHover={{ letterSpacing: "0.2em", color: chapter.secondary }}>
        Drop Aerospace Files
      </motion.div>
      <div className="mt-1.5 max-w-sm truncate text-[12px] leading-5 text-white/58">
        接入公开航空维修手册、技术标准、检修流程与培训资料，自动完成解析、清洗、分块、向量化与来源绑定。
      </div>
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        {["PDF", "DOC", "PPT", "Metadata", "OCR Ready"].map((tag) => (
          <TacticalTag key={tag} accent={chapter.secondary}>
            {tag}
          </TacticalTag>
        ))}
      </div>
      <motion.div
        className="pointer-events-none absolute inset-x-8 bottom-4 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, ${chapter.secondary}, transparent)` }}
        animate={{ scaleX: [0.35, 1, 0.35], opacity: [0.35, 1, 0.35] }}
        transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
      />
    </motion.div>
  );
}

function HomePage({
  chapter,
  enterQA,
  openTrace,
}: {
  chapter: Chapter;
  enterQA: () => void;
  openTrace: () => void;
}) {
  const detailRef = useRef<HTMLDivElement | null>(null);
  const advantageButtonRef = useRef<HTMLButtonElement | null>(null);

  const scrollToAdvantage = () => {
    const targetNode = advantageButtonRef.current;
    if (!targetNode) return;

    const scrollHost = targetNode.closest(".stage-scroll");
    if (scrollHost instanceof HTMLElement) {
      const targetTop = scrollHost.scrollTop + targetNode.getBoundingClientRect().top - scrollHost.getBoundingClientRect().top;
      scrollHost.scrollTo({
        top: Math.max(targetTop, 0),
        behavior: "smooth",
      });
      return;
    }

    targetNode.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="relative min-h-full py-4">
      <section className="relative flex min-h-[32rem] flex-col justify-between overflow-hidden pt-0 sm:min-h-[40rem] lg:min-h-[calc(100dvh-16rem)] xl:min-h-[calc(100dvh-15rem)]">
        <div className="pointer-events-none absolute left-0 top-8 font-display text-[7rem] leading-none text-white/[0.05] md:text-[12rem]">
          00
        </div>
        <div className="grid flex-1 gap-8 lg:grid-cols-[1.04fr_0.96fr]">
          <div className="relative z-10 flex max-w-3xl flex-col pb-4 pt-0 md:pb-6">
            <motion.div
              className="mb-4 flex flex-wrap gap-4"
              initial="initial"
              animate="animate"
              variants={{ initial: {}, animate: { transition: { staggerChildren: 0.08, delayChildren: 0.25 } } }}
            >
              {["Innovation Training Project", "Xidian University", "Chinese Aviation Corpus"].map((tag) => (
                <motion.div
                  key={tag}
                  className="flex h-8 min-w-[9.5rem] items-center justify-center border px-4 text-[10px] uppercase tracking-[0.28em] text-white/74 md:h-9 md:min-w-[10.5rem]"
                  style={{ borderColor: `${chapter.accent}66`, background: `${chapter.accent}10` }}
                  variants={{
                    initial: { opacity: 0, x: -40 },
                    animate: { opacity: 1, x: 0, transition: { duration: 0.5 } },
                  }}
                >
                  {tag}
                </motion.div>
              ))}
            </motion.div>
            <motion.div
              className="mb-6 mt-2 text-[1.15rem] tracking-[0.2em] text-white/56 md:mb-7 md:text-[1.45rem] md:tracking-[0.26em]"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.55, delay: 0.22 } }}
            >
              基于大模型的智能文档数据管理平台
            </motion.div>
            <motion.h1
              className="max-w-5xl font-display text-[3.4rem] uppercase leading-[0.88] tracking-[0.12em] text-white md:text-[5rem] xl:text-[6rem]"
              initial="initial"
              animate="animate"
              variants={{ initial: {}, animate: { transition: { staggerChildren: 0.08 } } }}
            >
              {["Document", "Intelligence", "Core"].map((word) => (
                <motion.span
                  key={word}
                  className="mr-4 inline-block"
                  variants={{
                    initial: { opacity: 0, y: 90, rotate: -8, filter: "blur(8px)" },
                    animate: {
                      opacity: 1,
                      y: 0,
                      rotate: 0,
                      filter: "blur(0px)",
                      transition: { duration: 0.7, ease: [0.16, 1, 0.3, 1] },
                    },
                  }}
                >
                  {word}
                </motion.span>
              ))}
            </motion.h1>
            <motion.div
              className="mt-8 flex flex-wrap items-end gap-5 min-w-0 md:mt-9"
              initial={{ opacity: 0, y: 22 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.55, delay: 0.6 } }}
            >
              <motion.button
                onClick={enterQA}
                whileHover={{ scale: 1.018, y: -5 }}
                whileTap={{ scale: 0.988 }}
                className="group relative flex h-[7rem] min-w-0 flex-1 items-center overflow-hidden border px-5 py-4 text-left sm:h-[8.35rem] md:h-[9.55rem] md:px-6"
                style={{
                  borderColor: `${chapter.accent}88`,
                  background: `linear-gradient(180deg, ${chapter.accent}20, rgba(255,255,255,0.02))`,
                  boxShadow: `0 0 26px ${chapter.accent}24`,
                  clipPath: "polygon(0 0, calc(100% - 32px) 0, 100% 32px, 100% 100%, 0 100%)",
                }}
                >
                  <motion.span
                    className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                    style={{ background: `radial-gradient(circle at 20% 26%, ${chapter.accent}4f 0%, transparent 62%)` }}
                  />
                  <span className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/45 to-transparent opacity-40" />
                  <span
                    className="pointer-events-none absolute inset-[2px] border border-white/8 opacity-70"
                    style={{ clipPath: "polygon(0 0, calc(100% - 28px) 0, 100% 28px, 100% 100%, 0 100%)" }}
                  />
                  <span
                    className="pointer-events-none absolute left-0 top-0 h-full w-[6px]"
                    style={{ background: `linear-gradient(180deg, ${chapter.accent}, transparent 82%)` }}
                  />
                  <div className="relative z-10 grid w-full grid-cols-[1fr_auto] items-center gap-5">
                    <div className="pl-2">
                      <div className="font-display text-[1.9rem] uppercase leading-[0.82] tracking-[0.24em] text-white md:text-[2.35rem]">
                        Enter
                      </div>
                      <div className="mt-2 font-display text-[2.35rem] uppercase leading-[0.8] tracking-[0.14em] text-white md:text-[3.1rem]">
                        System
                      </div>
                    </div>
                    <div className="flex h-14 w-14 items-center justify-center border border-white/10 bg-black/20 md:h-16 md:w-16">
                      <ArrowRight
                        size={24}
                        className="text-white/56 transition-all duration-300 group-hover:translate-x-1 group-hover:text-white"
                      />
                    </div>
                  </div>
                </motion.button>

              <motion.button
                onClick={openTrace}
                whileHover={{ scale: 1.018, y: -5 }}
                whileTap={{ scale: 0.988 }}
                className="group relative flex h-[7rem] w-[4.5rem] shrink-0 items-center justify-center overflow-hidden border sm:h-[8.35rem] sm:w-[5rem] md:h-[9.55rem] md:w-[5.8rem]"
                style={{
                  borderColor: `${chapter.secondary}7a`,
                  background: `linear-gradient(180deg, ${chapter.secondary}12, rgba(255,255,255,0.02))`,
                  boxShadow: `0 0 22px ${chapter.secondary}16`,
                  clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 24px), calc(100% - 24px) 100%, 0 100%)",
                }}
              >
                  <motion.span
                    className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                    style={{ background: `radial-gradient(circle at 50% 16%, ${chapter.secondary}44 0%, transparent 62%)` }}
                  />
                  <span
                    className="pointer-events-none absolute left-0 top-0 h-full w-[4px]"
                    style={{ background: `linear-gradient(180deg, ${chapter.secondary}, transparent 78%)` }}
                  />
                  <span className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/35 to-transparent opacity-40" />
                  <span
                    className="pointer-events-none absolute inset-[2px] border border-white/8 opacity-70"
                    style={{ clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 20px), calc(100% - 20px) 100%, 0 100%)" }}
                  />
                  <div className="relative z-10 flex h-full w-full items-center justify-center px-[0.5rem] py-3 md:px-[0.7rem]">
                    <div className="flex items-start justify-center gap-[0.38rem] md:gap-[0.48rem]">
                      <div className="flex flex-col items-center justify-center gap-[0.14rem]">
                        {"TRACE".split("").map((letter, index) => (
                          <span
                            key={`trace-${index}`}
                            className="font-display text-[18px] uppercase leading-none tracking-[0.18em] text-white/90 transition-colors duration-300 group-hover:text-white md:text-[21px]"
                            style={{ display: "inline-block", transform: "scaleX(1.18)" }}
                          >
                            {letter}
                          </span>
                        ))}
                      </div>
                      <div className="flex flex-col items-center justify-center gap-[0.08rem] pt-[0.08rem] md:pt-[0.12rem]">
                        {"OPEN".split("").map((letter, index) => (
                          <span
                            key={`open-${index}`}
                            className="font-display text-[16px] uppercase leading-none tracking-[0.24em] text-white/42 transition-colors duration-300 group-hover:text-white/62 md:text-[18px]"
                            style={{ display: "inline-block", transform: "scaleX(1.18)" }}
                          >
                            {letter}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  <motion.span
                    className="pointer-events-none absolute bottom-2 right-[1.2rem] h-8 w-px md:right-[1.45rem]"
                    style={{ background: `linear-gradient(180deg, ${chapter.secondary}, transparent)` }}
                    animate={{ scaleY: [0.72, 1, 0.72], opacity: [0.45, 1, 0.45] }}
                    transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
                  />
                </motion.button>
            </motion.div>

            <motion.button
              ref={advantageButtonRef}
              onClick={scrollToAdvantage}
              whileHover={{ x: 6 }}
              whileTap={{ scale: 0.985 }}
              className="group mt-auto inline-flex items-center gap-3 border-0 bg-transparent p-0 pt-6 text-left"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.55, delay: 0.72 } }}
            >
              <span className="font-display text-2xl uppercase tracking-[0.08em] text-white md:text-[2rem]">Advantage</span>
              <motion.span
                className="text-white/42"
                animate={{ y: [0, 7, 0] }}
                transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
              >
                <ArrowRight size={18} className="rotate-90" />
              </motion.span>
              <span className="pointer-events-none absolute -bottom-2 left-0 h-px w-full origin-left scale-x-0 bg-gradient-to-r from-white/80 to-transparent transition-transform duration-300 group-hover:scale-x-100" />
            </motion.button>
          </div>

          <motion.div
            className="relative flex items-center justify-center overflow-hidden"
            initial={{ opacity: 0, scale: 0.84 }}
            animate={{ opacity: 1, scale: 1, transition: { duration: 0.85, delay: 0.22 } }}
          >
            <div className="pointer-events-none absolute inset-x-[10%] top-1/2 hidden h-px bg-gradient-to-r from-transparent via-white/12 to-transparent lg:block" />
            <div className="pointer-events-none absolute inset-y-[16%] left-1/2 hidden w-px bg-gradient-to-b from-transparent via-white/10 to-transparent lg:block" />
            <CoreDevice accent={chapter.accent} secondary={chapter.secondary} />
          </motion.div>
        </div>

      </section>

      <section ref={detailRef} className="relative scroll-mt-6 space-y-5 overflow-hidden pb-4 pt-6">
        <StageScrollReveal
          className="pointer-events-none absolute left-[6%] top-6 h-36 w-36 rounded-full blur-3xl"
          style={{ background: `radial-gradient(circle, ${chapter.accent}50 0%, transparent 72%)` }}
          initial={{ opacity: 0, scale: 0.36, x: 0, y: 0 }}
          reveal={{ opacity: [0, 0.58, 0.18], scale: [0.42, 1.18, 1], x: [0, 46, 20], y: [0, -18, 0] }}
          transition={{ duration: 1.05, ease: [0.16, 1, 0.3, 1] }}
          amount={0.06}
        />
        <StageScrollReveal
          className="pointer-events-none absolute right-[8%] top-20 h-44 w-44 rounded-full blur-3xl"
          style={{ background: `radial-gradient(circle, ${chapter.secondary}4a 0%, transparent 72%)` }}
          initial={{ opacity: 0, scale: 0.34, x: 0, y: 0 }}
          reveal={{ opacity: [0, 0.5, 0.12], scale: [0.36, 1.12, 1], x: [0, -52, -22], y: [0, 24, 0] }}
          transition={{ duration: 1.1, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
          amount={0.06}
        />
        <StageScrollReveal
          className="pointer-events-none absolute inset-x-[6%] top-2 h-px origin-center"
          style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, ${chapter.secondary}, transparent)` }}
          initial={{ opacity: 0, scaleX: 0.04 }}
          reveal={{ opacity: [0, 1, 0.42], scaleX: [0.04, 1, 0.88] }}
          transition={{ duration: 0.95, ease: [0.16, 1, 0.3, 1] }}
          amount={0.06}
        />

        <div className="relative z-10 space-y-4">
          <div className="max-w-fit">
            <StageScrollReveal
              initial={{ opacity: 0, x: -48, y: 18, filter: "blur(12px)" }}
              reveal={{ opacity: 1, x: 0, y: 0, filter: "blur(0px)" }}
              transition={{ duration: 0.58, ease: [0.16, 1, 0.3, 1] }}
              amount={0.12}
            >
              <div className="text-[10px] uppercase tracking-[0.34em] text-white/35">Our Advantage</div>
              <div className="mt-2 font-display text-2xl uppercase tracking-[0.12em] text-white md:text-[2rem]">
                Project Highlights
              </div>
            </StageScrollReveal>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {innovations.map((item, index) => (
              <StageScrollReveal
                key={item.title}
                initial={
                  index === 0
                    ? { opacity: 0, x: -180, y: 80, rotate: -8, scale: 0.9, filter: "blur(16px)" }
                    : index === 1
                      ? { opacity: 0, x: -48, y: 150, rotate: -2, scale: 0.84, filter: "blur(18px)" }
                      : index === 2
                        ? { opacity: 0, x: 48, y: 150, rotate: 2, scale: 0.84, filter: "blur(18px)" }
                        : { opacity: 0, x: 180, y: 80, rotate: 8, scale: 0.9, filter: "blur(16px)" }
                }
                reveal={{ opacity: 1, x: 0, y: 0, rotate: 0, scale: 1, filter: "blur(0px)" }}
                transition={{ duration: 0.78, delay: 0.06 * index, ease: [0.16, 1, 0.3, 1] }}
                amount={0.18}
              >
                <FeatureCard
                  accent={index % 2 === 1 ? chapter.secondary : chapter.accent}
                  icon={item.icon}
                  title={item.title}
                  desc={item.body}
                />
              </StageScrollReveal>
            ))}
          </div>
        </div>

        <div className="relative z-10 grid gap-5 xl:grid-cols-[1fr_1fr]">
          <StageScrollReveal
            initial={{ opacity: 0, x: -180, y: 60, rotate: -3, scale: 0.94, filter: "blur(22px)", clipPath: "inset(0 100% 0 0)" }}
            reveal={{ opacity: 1, x: 0, y: 0, rotate: 0, scale: 1, filter: "blur(0px)", clipPath: "inset(0 0% 0 0)" }}
            transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            amount={0.12}
          >
            <InteractiveCutPanel accent={chapter.secondary} secondary={chapter.accent} kicker="Evaluation Targets" title="Defense Snapshot">
              <div className="grid gap-4 md:grid-cols-2">
                {evaluationTargets.map((metric, index) => (
                  <MetricCard
                    key={metric.label}
                    chapter={chapter}
                    label={metric.label}
                    value={metric.value}
                    note={metric.note}
                    index={index}
                  />
                ))}
              </div>
            </InteractiveCutPanel>
          </StageScrollReveal>

          <StageScrollReveal
            initial={{ opacity: 0, x: 180, y: 60, rotate: 3, scale: 0.94, filter: "blur(22px)", clipPath: "inset(0 0 0 100%)" }}
            reveal={{ opacity: 1, x: 0, y: 0, rotate: 0, scale: 1, filter: "blur(0px)", clipPath: "inset(0 0 0 0%)" }}
            transition={{ duration: 0.9, delay: 0.06, ease: [0.16, 1, 0.3, 1] }}
            amount={0.12}
          >
            <InteractiveCutPanel accent={chapter.accent} secondary={chapter.secondary} kicker="Engineering Route" title="Project Foundation">
              <div className="flex flex-wrap gap-2">
                {stackBadges.map((badge, index) => (
                  <StageScrollReveal
                    key={badge}
                    initial={{ opacity: 0, scale: 0.72, y: 18, filter: "blur(10px)" }}
                    reveal={{ opacity: 1, scale: 1, y: 0, filter: "blur(0px)" }}
                    transition={{ duration: 0.35, delay: 0.025 * index }}
                    amount={0.1}
                  >
                    <TacticalTag accent={index % 2 === 0 ? chapter.secondary : chapter.accent}>{badge}</TacticalTag>
                  </StageScrollReveal>
                ))}
              </div>
              <div className="mt-5 grid gap-3 md:grid-cols-3">
                {supportHighlights.map((item, index) => (
                  <StageScrollReveal
                    key={item.title}
                    className="border border-white/8 bg-white/[0.006] px-4 py-4"
                    initial={{ opacity: 0, x: index % 2 === 0 ? -24 : 24, y: 16, scale: 0.95, filter: "blur(12px)" }}
                    reveal={{ opacity: 1, x: 0, y: 0, scale: 1, filter: "blur(0px)" }}
                    transition={{ delay: 0.08 * index }}
                    whileHover={{ y: -6, scale: 1.015, borderColor: `${chapter.accent}55` }}
                    amount={0.18}
                  >
                    <div className="font-display text-base uppercase tracking-[0.14em] text-white">{item.title}</div>
                    <div className="mt-2 text-sm leading-6 text-white/58">{item.body}</div>
                  </StageScrollReveal>
                ))}
              </div>
            </InteractiveCutPanel>
          </StageScrollReveal>
        </div>

      </section>
    </div>
  );
}

// 新增：带光效的预览信息卡片组件
function InteractivePreviewInfoCard({ 
  label, 
  value, 
  accent 
}: { 
  label: string; 
  value: string; 
  accent: string;
}) {
  const pointer = useInteractivePointer({ maxShift: 4, maxRotate: 1.5 });
  const glow = useMotionTemplate`radial-gradient(180px circle at ${pointer.glowX}px ${pointer.glowY}px, ${accent}3a 0%, transparent 70%)`;

  return (
    <motion.div
      onMouseMove={pointer.handleMove}
      onMouseLeave={pointer.handleLeave}
      whileHover={{ y: -4, scale: 1.015 }}
      style={{
        x: pointer.shiftX,
        y: pointer.shiftY,
        rotateX: pointer.rotateX,
        rotateY: pointer.rotateY,
        transformPerspective: 1200,
      }}
      className="group relative overflow-hidden border border-white/8 bg-white/[0.02] px-4 py-4"
    >
      <motion.div className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: glow }} />
      <div className="relative z-10">
        <div className="text-[10px] uppercase tracking-[0.28em] text-white/34">{label}</div>
        <div className="mt-2 font-display text-base uppercase tracking-[0.14em]" style={{ color: accent }}>
          {value}
        </div>
      </div>
    </motion.div>
  );
}
function IntakePage({
  chapter,
  goSystem,
}: {
  chapter: Chapter;
  goSystem: () => void;
}) {
  const documentPanelRef = useRef<HTMLDivElement | null>(null);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseRecord[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [knowledgeBaseKeyword, setKnowledgeBaseKeyword] = useState("");
  const [documentKeyword, setDocumentKeyword] = useState("");
  const [activeKnowledgeBaseId, setActiveKnowledgeBaseId] = useState<number | "all">("all");
  const [knowledgeBasePage, setKnowledgeBasePage] = useState(1);
  const [documentPage, setDocumentPage] = useState(1);
  const [knowledgeBaseModalOpen, setKnowledgeBaseModalOpen] = useState(false);
  const [knowledgeBaseModalMode, setKnowledgeBaseModalMode] = useState<"create" | "edit">("create");
  const [editingKnowledgeBaseId, setEditingKnowledgeBaseId] = useState<number | null>(null);
  const [knowledgeBaseDraft, setKnowledgeBaseDraft] = useState({ name: "", description: "" });
  const [documentModalOpen, setDocumentModalOpen] = useState(false);
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [documentIndexing, setDocumentIndexing] = useState(false);
  const [uploadStage, setUploadStage] = useState<UploadStage>("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadDetail, setUploadDetail] = useState("等待上传");
  const [documentSelectedKbId, setDocumentSelectedKbId] = useState<number | null>(null);
  const [aiClassifyResult, setAiClassifyResult] = useState<{
    suggestedKbId: number;
    suggestedKbName: string;
    newKbName: string;
    newKbDescription: string;
    reason: string;
  } | null>(null);
  const [aiClassifying, setAiClassifying] = useState(false);
  const [documentDraft, setDocumentDraft] = useState({
    fileName: "",
    type: "PDF" as DocumentRecord["type"],
    size: "",
    chunkCount: "",
    status: "INDEXING" as DocumentStatus,
    summary: "",
  });
  const [previewDocumentId, setPreviewDocumentId] = useState<number | null>(null);
  const [pendingDelete, setPendingDelete] = useState<ManagementDeleteState | null>(null);

  useEffect(() => {
    let active = true;

    void Promise.all([
      knowledgeBaseDocumentService.listKnowledgeBases(),
      knowledgeBaseDocumentService.listDocuments(),
    ]).then(([knowledgeBaseRecords, documentRecords]) => {
      if (!active) return;

      setKnowledgeBases(knowledgeBaseRecords);
      setDocuments(documentRecords);
    });

    return () => {
      active = false;
    };
  }, []);

  const knowledgeBaseNameMap = useMemo(
    () => new Map(knowledgeBases.map((item) => [item.id, item.name])),
    [knowledgeBases],
  );

  const documentCountByKnowledgeBase = useMemo(() => {
    const counts = new Map<number, number>();

    documents.forEach((item) => {
      counts.set(item.knowledgeBaseId, (counts.get(item.knowledgeBaseId) ?? 0) + 1);
    });

    return counts;
  }, [documents]);

  const activeKnowledgeBase = useMemo(
    () => (activeKnowledgeBaseId === "all" ? null : knowledgeBases.find((item) => item.id === activeKnowledgeBaseId) ?? null),
    [knowledgeBases, activeKnowledgeBaseId],
  );

  const processingDocumentsCount = useMemo(
    () => documents.filter((item) => item.status === "INDEXING" || item.status === "CHUNKING").length,
    [documents],
  );

  const filteredKnowledgeBases = useMemo(() => {
    const keyword = knowledgeBaseKeyword.trim().toLowerCase();

    return knowledgeBases.filter((item) => !keyword || item.name.toLowerCase().includes(keyword));
  }, [knowledgeBases, knowledgeBaseKeyword]);

  const knowledgeBasePageSize = 8;
  const knowledgeBaseTotalItems = filteredKnowledgeBases.length;
  const knowledgeBaseTotalPages = Math.max(1, Math.ceil(knowledgeBaseTotalItems / knowledgeBasePageSize));

  useEffect(() => {
    setKnowledgeBasePage((current) => Math.min(current, knowledgeBaseTotalPages));
  }, [knowledgeBaseTotalPages]);

  const pagedKnowledgeBases = useMemo(() => {
    const start = (knowledgeBasePage - 1) * knowledgeBasePageSize;
    return filteredKnowledgeBases.slice(start, start + knowledgeBasePageSize);
  }, [filteredKnowledgeBases, knowledgeBasePage]);
  const knowledgeBaseRangeStart =
    knowledgeBaseTotalItems === 0 || pagedKnowledgeBases.length === 0
      ? 0
      : filteredKnowledgeBases.findIndex((item) => item.id === pagedKnowledgeBases[0]?.id) + 1;
  const knowledgeBaseRangeEnd =
    knowledgeBaseRangeStart === 0 ? 0 : knowledgeBaseRangeStart + pagedKnowledgeBases.length - 1;

  const filteredDocuments = useMemo(() => {
    const keyword = documentKeyword.trim().toLowerCase();

    return documents.filter((item) => {
      const matchesKnowledgeBase = activeKnowledgeBaseId === "all" ? true : item.knowledgeBaseId === activeKnowledgeBaseId;
      const matchesKeyword =
        !keyword ||
        item.fileName.toLowerCase().includes(keyword) ||
        item.summary.toLowerCase().includes(keyword) ||
        (knowledgeBaseNameMap.get(item.knowledgeBaseId)?.toLowerCase().includes(keyword) ?? false);

      return matchesKnowledgeBase && matchesKeyword;
    });
  }, [documents, activeKnowledgeBaseId, documentKeyword, knowledgeBaseNameMap]);

  const documentPageSize = 4;
  const documentTotalItems = filteredDocuments.length;
  const documentTotalPages = Math.max(1, Math.ceil(documentTotalItems / documentPageSize));

  useEffect(() => {
    setDocumentPage((current) => Math.min(current, documentTotalPages));
  }, [documentTotalPages]);

  const pagedDocuments = useMemo(() => {
    const start = (documentPage - 1) * documentPageSize;
    return filteredDocuments.slice(start, start + documentPageSize);
  }, [filteredDocuments, documentPage]);
  const documentRangeStart =
    documentTotalItems === 0 || pagedDocuments.length === 0
      ? 0
      : filteredDocuments.findIndex((item) => item.id === pagedDocuments[0]?.id) + 1;
  const documentRangeEnd = documentRangeStart === 0 ? 0 : documentRangeStart + pagedDocuments.length - 1;

  const previewDocument = useMemo(
    () => documents.find((item) => item.id === previewDocumentId) ?? null,
    [documents, previewDocumentId],
  );

  const pendingDeleteKnowledgeBase = useMemo(
    () =>
      pendingDelete?.type === "knowledgeBase"
        ? knowledgeBases.find((item) => item.id === pendingDelete.id) ?? null
        : null,
    [knowledgeBases, pendingDelete],
  );

  const pendingDeleteDocument = useMemo(
    () =>
      pendingDelete?.type === "document"
        ? documents.find((item) => item.id === pendingDelete.id) ?? null
        : null,
    [documents, pendingDelete],
  );

  const scrollToDocumentPanel = () => {
    const target = documentPanelRef.current;
    if (!target) return;

    const scrollHost = target.closest(".stage-scroll");
    if (scrollHost instanceof HTMLElement) {
      const targetTop = scrollHost.scrollTop + target.getBoundingClientRect().top - scrollHost.getBoundingClientRect().top - 12;
      scrollHost.scrollTo({ top: Math.max(targetTop, 0), behavior: "smooth" });
      return;
    }

    target.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const focusKnowledgeBase = (knowledgeBaseId: number, options?: { scrollToDocuments?: boolean }) => {
    setActiveKnowledgeBaseId(knowledgeBaseId);
    setDocumentPage(1);

    if (options?.scrollToDocuments) {
      requestAnimationFrame(scrollToDocumentPanel);
    }
  };

  const openCreateKnowledgeBase = () => {
    setKnowledgeBaseModalMode("create");
    setEditingKnowledgeBaseId(null);
    setKnowledgeBaseDraft({ name: "", description: "" });
    setKnowledgeBaseModalOpen(true);
  };

  const openEditKnowledgeBase = (record: KnowledgeBaseRecord) => {
    setKnowledgeBaseModalMode("edit");
    setEditingKnowledgeBaseId(record.id);
    setKnowledgeBaseDraft({ name: record.name, description: record.description });
    setKnowledgeBaseModalOpen(true);
  };

  const handleSaveKnowledgeBase = async () => {
    const name = knowledgeBaseDraft.name.trim();
    const description = knowledgeBaseDraft.description.trim();

    if (!name || !description) return;

    if (knowledgeBaseModalMode === "edit" && editingKnowledgeBaseId !== null) {
      const updatedRecord = await knowledgeBaseDocumentService.updateKnowledgeBase(editingKnowledgeBaseId, {
        name,
        description,
      });

      setKnowledgeBases((current) =>
        current.map((item) => (item.id === editingKnowledgeBaseId ? updatedRecord : item)),
      );
    } else {
      const createdRecord = await knowledgeBaseDocumentService.createKnowledgeBase({
        name,
        description,
        creator: "he114TnT",
      });

      setKnowledgeBases((current) => [createdRecord, ...current.filter((item) => item.id !== createdRecord.id)]);
    }

    setKnowledgeBaseModalOpen(false);
  };

  const handleDeleteKnowledgeBase = async (knowledgeBaseId: number) => {
    const removedDocumentIds = new Set(
      documents.filter((item) => item.knowledgeBaseId === knowledgeBaseId).map((item) => item.id),
    );

    await knowledgeBaseDocumentService.deleteKnowledgeBase(knowledgeBaseId);
    setKnowledgeBases((current) => current.filter((item) => item.id !== knowledgeBaseId));
    setDocuments((current) => current.filter((item) => item.knowledgeBaseId !== knowledgeBaseId));
    setPreviewDocumentId((current) => (current !== null && removedDocumentIds.has(current) ? null : current));

    if (activeKnowledgeBaseId === knowledgeBaseId) {
      setActiveKnowledgeBaseId("all");
    }
  };

  const openDocumentUpload = () => {
    setDocumentFile(null);
    setDocumentIndexing(false);
    setUploadStage("idle");
    setUploadProgress(0);
    setUploadDetail("等待上传");
    setDocumentSelectedKbId(null);
    setAiClassifyResult(null);
    setAiClassifying(false);
    setDocumentDraft({
      fileName: "",
      type: "PDF",
      size: "",
      chunkCount: "",
      status: "INDEXING",
      summary: "",
    });
    setDocumentModalOpen(true);
  };

  const classifyFile = async (fileName: string) => {
    setAiClassifying(true);
    setAiClassifyResult(null);
    try {
      const res = await fetch(`${managementApiBase}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fileName }),
      });
      if (res.ok) {
        const data = await res.json();
        setAiClassifyResult(data);
        if (data.suggestedKbId) {
          setDocumentSelectedKbId(data.suggestedKbId);
        }
      }
    } catch {
      // AI 分类失败不阻断流程
    } finally {
      setAiClassifying(false);
    }
  };

  const handleSaveDocument = async () => {
    const summary = documentDraft.summary.trim();
    let knowledgeBaseId = documentSelectedKbId;

    // 如果 AI 建议创建新知识库且用户未选择已有知识库
    if (!knowledgeBaseId && aiClassifyResult?.newKbName) {
      const createdKb = await knowledgeBaseDocumentService.createKnowledgeBase({
        name: aiClassifyResult.newKbName,
        description: aiClassifyResult.newKbDescription || "",
      });
      setKnowledgeBases((current) => [createdKb, ...current.filter((item) => item.id !== createdKb.id)]);
      knowledgeBaseId = createdKb.id;
    }

    if (!knowledgeBaseId) {
      // 兜底：使用第一个知识库
      knowledgeBaseId = knowledgeBases[0]?.id ?? 0;
    }
    if (!knowledgeBaseId) return;

    // 如果选择了文件，走真实上传 + 增量索引流程
    if (documentFile) {
      const fileName = documentFile.name;
      const sizeMB = (documentFile.size / (1024 * 1024)).toFixed(1) + " MB";
      const ext = fileName.split(".").pop()?.toUpperCase() || "PDF";

      setDocumentIndexing(true);
      setUploadStage("uploading");
      setUploadProgress(1);
      setUploadDetail("正在上传文件到服务端…");
      try {
        // 1. 上传文件到后端
        const uploadData = await uploadFileWithProgress(documentFile, (progressPercent) => {
          setUploadStage("uploading");
          setUploadProgress(Math.max(1, Math.round(progressPercent * 0.45)));
          setUploadDetail(`正在上传文件（${progressPercent}%）`);
        });

        setUploadStage("indexing");
        setUploadProgress(68);
        setUploadDetail("正在解析文档、生成语义分块并写入检索索引…");

        // 2. 触发增量索引
        const indexRes = await fetch(`${managementApiBase}/index`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fileName: uploadData.fileName }),
        });
        if (!indexRes.ok) throw new Error("索引失败");
        const indexData = await indexRes.json();
        const parsedChunkCount = Number(indexData.chunkCount);
        const finalChunkCount =
          Number.isFinite(parsedChunkCount) && parsedChunkCount > 0
            ? parsedChunkCount
            : parseChunkCountFromIndexMessage(String(indexData.message ?? ""));

        setUploadStage("saving");
        setUploadProgress(90);
        setUploadDetail("正在写入文档管理记录…");

        // 3. 创建文档记录
        const createdRecord = await knowledgeBaseDocumentService.createDocument({
          fileName: fileName.replace(/\.[^.]+$/, ""),
          knowledgeBaseId,
          type: ext as DocumentRecord["type"],
          size: uploadData.size || sizeMB,
          chunkCount: finalChunkCount,
          status: "READY",
          summary: summary || "",
        });

        setDocuments((current) => [createdRecord, ...current.filter((item) => item.id !== createdRecord.id)]);
        setActiveKnowledgeBaseId(knowledgeBaseId);
        setDocumentPage(1);
        setUploadStage("done");
        setUploadProgress(100);
        setUploadDetail(`上传完成，已生成 ${finalChunkCount} 个语义块。`);
        setDocumentModalOpen(false);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "未知错误";
        console.error("上传索引失败:", err);
        setUploadStage("error");
        setUploadDetail(`上传失败：${errorMessage}`);
        alert(`上传索引失败: ${errorMessage}`);
      } finally {
        setDocumentIndexing(false);
      }
      return;
    }

    // 兜底：手动填写模式（无文件）
    const fileName = documentDraft.fileName.trim();
    const size = documentDraft.size.trim();
    const chunkCount = Number(documentDraft.chunkCount);
    if (!fileName || !size || Number.isNaN(chunkCount) || chunkCount <= 0) return;

    const createdRecord = await knowledgeBaseDocumentService.createDocument({
      fileName,
      knowledgeBaseId,
      type: documentDraft.type,
      size,
      chunkCount,
      status: documentDraft.status,
      summary: summary || "",
    });

    setDocuments((current) => [createdRecord, ...current.filter((item) => item.id !== createdRecord.id)]);
    setActiveKnowledgeBaseId(knowledgeBaseId);
    setDocumentPage(1);
    setDocumentModalOpen(false);
  };

  const handleDeleteDocument = async (documentId: number) => {
    await knowledgeBaseDocumentService.deleteDocument(documentId);
    setDocuments((current) => current.filter((item) => item.id !== documentId));
    if (previewDocumentId === documentId) {
      setPreviewDocumentId(null);
    }
  };

  const requestDeleteKnowledgeBase = (knowledgeBaseId: number) => {
    setPendingDelete({ type: "knowledgeBase", id: knowledgeBaseId });
  };

  const requestDeleteDocument = (documentId: number) => {
    setPendingDelete({ type: "document", id: documentId });
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;

    if (pendingDelete.type === "knowledgeBase") {
      await handleDeleteKnowledgeBase(pendingDelete.id);
    } else {
      await handleDeleteDocument(pendingDelete.id);
    }

    setPendingDelete(null);
  };

  return (
    <div className="relative min-h-full pb-2">
      <StageHeader
        chapter={chapter}
        title="知识库和文档管理"
        tags={["Knowledge Base Ops", "Document Lifecycle", "Linked Filtering"]}
        className="mb-3 shrink-0"
        actions={
          <MagneticButton accent={chapter.accent} onClick={goSystem} className="justify-center">
            Return To System <ChevronLeft size={16} />
          </MagneticButton>
        }
      />
      <div className="grid gap-3">
        <div className="grid gap-3 xl:grid-cols-[0.58fr_1.42fr] xl:items-start">
          <motion.div
            initial={{ opacity: 0, x: -80, skewX: -8 }}
            animate={{ opacity: 1, x: 0, skewX: 0, transition: { duration: 0.7 } }}
            className="min-w-0"
          >
            <InteractiveCutPanel
              accent={chapter.accent}
              secondary={chapter.secondary}
              kicker="Dock 01"
              title="Document Intake Bay"
              className="flex min-h-[33.75rem] flex-col"
              actions={
                <ManagementActionButton accent={chapter.accent} icon={<UploadCloud size={14} />} onClick={openDocumentUpload} disabled={knowledgeBases.length === 0} className="h-10 px-4">
                  Upload Document
                </ManagementActionButton>
              }
            >
              <UploadBay chapter={chapter} onFileSelect={(file) => {
                setDocumentFile(file);
                const name = file.name.replace(/\.[^.]+$/, "");
                const ext = file.name.split(".").pop()?.toUpperCase() || "PDF";
                const sizeMB = (file.size / (1024 * 1024)).toFixed(1) + " MB";
                setDocumentDraft((current) => ({ ...current, fileName: name, type: ext as DocumentRecord["type"], size: sizeMB }));
                setDocumentIndexing(false);
                setUploadStage("idle");
                setUploadProgress(0);
                setUploadDetail("等待上传");
                setDocumentSelectedKbId(null);
                setAiClassifyResult(null);
                setDocumentModalOpen(true);
                void classifyFile(file.name);
              }} />
              <div className="mt-2 flex flex-wrap gap-2">
                <TacticalTag accent={chapter.accent}>{knowledgeBases.length} Knowledge Bases</TacticalTag>
                <TacticalTag accent={chapter.secondary}>{documents.length} Managed Documents</TacticalTag>
                <TacticalTag accent={chapter.accent}>{processingDocumentsCount} In Processing</TacticalTag>
              </div>
            </InteractiveCutPanel>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 80 }}
            animate={{ opacity: 1, x: 0, transition: { duration: 0.7, delay: 0.08 } }}
            className="min-w-0"
          >
            <InteractiveCutPanel accent={chapter.secondary} secondary={chapter.accent} kicker="Knowledge Base" title="Knowledge Base Management" className="flex min-h-[35.75rem] flex-col">
              <div className="mb-3 grid gap-2.5 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center">
                <div className="grid gap-2.5 md:grid-cols-[minmax(0,1fr)_auto]">
                  <ManagementFieldShell accent={chapter.secondary} icon={<Search size={16} />}>
                    <input
                      value={knowledgeBaseKeyword}
                      onChange={(event) => {
                        setKnowledgeBaseKeyword(event.target.value);
                        setKnowledgeBasePage(1);
                      }}
                      placeholder="搜索知识库名称"
                      className="w-full bg-transparent text-sm text-white outline-none placeholder:text-white/24"
                    />
                  </ManagementFieldShell>
                  {activeKnowledgeBase ? (
                    <div className="flex items-center gap-2 border border-white/10 bg-white/[0.03] px-3.5 text-[9px] uppercase tracking-[0.24em] text-white/52">
                      <span>Linked Filter</span>
                      <span className="h-[5px] w-[5px] rounded-full" style={{ background: chapter.secondary, boxShadow: `0 0 16px ${chapter.secondary}` }} />
                      <span className="text-white/82">{activeKnowledgeBase.name}</span>
                    </div>
                  ) : null}
                </div>
                <ManagementActionButton accent={chapter.secondary} icon={<Plus size={14} />} onClick={openCreateKnowledgeBase} className="justify-center">
                  New Knowledge Base
                </ManagementActionButton>
              </div>

              <div className="flex flex-1 flex-col overflow-hidden border border-white/8 bg-white/[0.008]" style={{ clipPath: "polygon(0 0, calc(100% - 22px) 0, 100% 22px, 100% 100%, 0 100%)" }}>
                <div className="flex-1 overflow-hidden">
                  <div className="min-h-full w-full min-w-0">
                    <div className="grid grid-cols-[3rem_minmax(0,1.04fr)_minmax(0,1.28fr)_4.25rem_5rem_8rem_14rem] gap-2 border-b border-white/[0.06] bg-white/[0.02] px-3 py-2.5 text-[9px] uppercase tracking-[0.22em] text-white/36">
                      <div>ID</div>
                      <div>知识库名称</div>
                      <div>描述</div>
                      <div>文档数量</div>
                      <div>创建者</div>
                      <div>创建时间</div>
                      <div>操作</div>
                    </div>
                    <div className="max-h-[24rem] min-h-[16rem] divide-y divide-white/[0.06] overflow-y-auto panel-scroll">
                      {pagedKnowledgeBases.length === 0 ? (
                        <div className="flex h-[16rem] items-center p-4">
                          <ManagementEmptyState
                            accent={chapter.secondary}
                            icon={<Database size={24} />}
                            title="No Knowledge Base"
                            description="当前筛选下没有匹配的知识库，可以新建一个知识库后继续上传和归档文档。"
                            action={
                              <ManagementActionButton accent={chapter.secondary} icon={<Plus size={14} />} onClick={openCreateKnowledgeBase}>
                                New Knowledge Base
                              </ManagementActionButton>
                            }
                          />
                        </div>
                      ) : (
                        pagedKnowledgeBases.map((item) => (
                          <button
                            type="button"
                            key={item.id}
                            onClick={() => focusKnowledgeBase(item.id)}
                            onMouseMove={handleFramePointerMove}
                            onMouseLeave={handleFramePointerLeave}
                            className="grid h-[4rem] w-full grid-cols-[3rem_minmax(0,1.04fr)_minmax(0,1.28fr)_4.25rem_5rem_8rem_14rem] items-center gap-2 px-3 py-2 text-left transition-colors duration-300 hover:bg-white/[0.03]"
                            style={{ background: activeKnowledgeBaseId === item.id ? `${chapter.secondary}10` : undefined }}
                          >
                            <div className="text-[13px] text-white/64">{item.id}</div>
                            <div className="min-w-0">
                              <div className="truncate text-[13px] font-medium text-white">{item.name}</div>
                            </div>
                            <div
                              className="truncate text-[13px] leading-5 text-white/54"
                            >
                              {item.description}
                            </div>
                            <div className="text-[13px] text-white/72">{documentCountByKnowledgeBase.get(item.id) ?? 0}</div>
                            <div className="truncate whitespace-nowrap text-[13px] text-white/54">{item.creator}</div>
                            <div className="whitespace-nowrap text-[13px] text-white/54">{item.createdAt}</div>
                            <div className="flex flex-nowrap items-center gap-1.5" onClick={(event) => event.stopPropagation()}>
                              <ManagementActionButton accent={chapter.secondary} secondary onClick={() => openEditKnowledgeBase(item)} className="h-8 justify-center px-2 text-[8px] tracking-[0.18em]">
                                <Pencil size={12} />
                                Edit
                              </ManagementActionButton>
                              <ManagementActionButton accent={chapter.accent} secondary onClick={() => focusKnowledgeBase(item.id, { scrollToDocuments: true })} className="h-8 justify-center px-2 text-[8px] tracking-[0.18em]">
                                <FolderOpen size={12} />
                                Docs
                              </ManagementActionButton>
                              <ManagementActionButton accent="#ff6c86" secondary onClick={() => requestDeleteKnowledgeBase(item.id)} className="h-8 justify-center px-2 text-[8px] tracking-[0.18em]">
                                <Trash2 size={12} />
                                Delete
                              </ManagementActionButton>
                            </div>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <ManagementPagination
                accent={chapter.secondary}
                page={knowledgeBasePage}
                pageSize={knowledgeBasePageSize}
                totalItems={knowledgeBaseTotalItems}
                rangeStart={knowledgeBaseRangeStart}
                rangeEnd={knowledgeBaseRangeEnd}
                onPageChange={setKnowledgeBasePage}
              />
            </InteractiveCutPanel>
          </motion.div>
        </div>

        <motion.div
          ref={documentPanelRef}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0, transition: { duration: 0.72, delay: 0.12 } }}
          className="min-w-0"
        >
          <InteractiveCutPanel accent={chapter.accent} secondary={chapter.secondary} kicker="Document Grid" title="Document Management" className="flex h-[39.75rem] flex-col">
            <div className="mb-3 grid gap-2.5 xl:grid-cols-[minmax(0,1fr)_14rem_auto] xl:items-center">
              <ManagementFieldShell accent={chapter.accent} icon={<FileSearch size={16} />}>
                <input
                  value={documentKeyword}
                  onChange={(event) => {
                    setDocumentKeyword(event.target.value);
                    setDocumentPage(1);
                  }}
                  placeholder="搜索文件名或摘要"
                  className="w-full bg-transparent text-sm text-white outline-none placeholder:text-white/24"
                />
              </ManagementFieldShell>

              <ManagementFieldShell accent={chapter.secondary} icon={<FolderOpen size={16} />}>
                <select
                  value={activeKnowledgeBaseId === "all" ? "all" : String(activeKnowledgeBaseId)}
                  onChange={(event) => {
                    setActiveKnowledgeBaseId(event.target.value === "all" ? "all" : Number(event.target.value));
                    setDocumentPage(1);
                  }}
                  className="w-full appearance-none bg-transparent pr-6 text-sm text-white outline-none"
                >
                  <option value="all" className="bg-[#0a0c11] text-white">
                    全部知识库
                  </option>
                  {knowledgeBases.map((item) => (
                    <option key={item.id} value={item.id} className="bg-[#0a0c11] text-white">
                      {item.name}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-white/28" />
              </ManagementFieldShell>

              <div className="flex flex-wrap gap-3">
                {activeKnowledgeBase ? (
                  <ManagementActionButton accent={chapter.secondary} secondary onClick={() => setActiveKnowledgeBaseId("all")} className="justify-center">
                    Clear Linked Filter
                  </ManagementActionButton>
                ) : null}
                <ManagementActionButton accent={chapter.accent} icon={<UploadCloud size={14} />} onClick={openDocumentUpload} disabled={knowledgeBases.length === 0} className="justify-center">
                  Upload Document
                </ManagementActionButton>
              </div>
            </div>

          {activeKnowledgeBase ? (
            <div
              className="mb-3 flex flex-wrap items-center justify-between gap-2.5 border border-white/8 bg-white/[0.02] px-3.5 py-3"
              style={{ clipPath: "polygon(0 0, calc(100% - 18px) 0, 100% 18px, 100% 100%, 0 100%)" }}
              onMouseMove={handleFramePointerMove}
              onMouseLeave={handleFramePointerLeave}
            >
              <div>
                <div className="text-[10px] uppercase tracking-[0.3em] text-white/34">Linked Knowledge Base</div>
                <div className="mt-1.5 font-display text-base uppercase tracking-[0.14em] text-white">{activeKnowledgeBase.name}</div>
              </div>
              <div className="max-w-[34rem] truncate text-[13px] leading-5 text-white/54">{activeKnowledgeBase.description}</div>
            </div>
          ) : null}

            <div className="flex flex-1 flex-col overflow-hidden border border-white/8 bg-white/[0.008]" style={{ clipPath: "polygon(0 0, calc(100% - 22px) 0, 100% 22px, 100% 100%, 0 100%)" }}>
              <div className="flex-1 overflow-x-auto">
                <div className="min-h-full min-w-[78.4rem]">
                  <div className="grid grid-cols-[3rem_minmax(0,1.16fr)_minmax(0,0.88fr)_4.15rem_5rem_4rem_6.5rem_8.8rem_12rem] gap-1.5 border-b border-white/[0.06] bg-white/[0.02] px-3 py-2.5 text-[9px] uppercase tracking-[0.22em] text-white/36">
                    <div>ID</div>
                    <div className="pl-2">文件名</div>
                    <div>所属知识库</div>
                    <div>类型</div>
                    <div>大小</div>
                    <div>分块数</div>
                    <div>状态</div>
                    <div>上传时间</div>
                    <div>操作</div>
                  </div>
                  <div className="h-[17rem] divide-y divide-white/[0.06]">
                    {pagedDocuments.length === 0 ? (
                      <div className="flex h-[17rem] items-center p-4">
                        <ManagementEmptyState
                          accent={chapter.accent}
                          icon={<FileText size={24} />}
                          title="No Document Matched"
                          description="当前筛选下没有匹配的文档。你可以清空联动筛选，或者直接上传新文档进入该知识库。"
                          action={
                            <ManagementActionButton accent={chapter.accent} icon={<UploadCloud size={14} />} onClick={openDocumentUpload} disabled={knowledgeBases.length === 0}>
                              Upload Document
                            </ManagementActionButton>
                          }
                        />
                      </div>
                    ) : (
                      pagedDocuments.map((item) => (
                        <div
                          key={item.id}
                          onMouseMove={handleFramePointerMove}
                          onMouseLeave={handleFramePointerLeave}
                          className="grid h-[4.25rem] grid-cols-[3rem_minmax(0,1.16fr)_minmax(0,0.88fr)_4.15rem_5rem_4rem_6.5rem_8.8rem_12rem] items-center gap-1.5 px-3 py-2 transition-colors duration-300 hover:bg-white/[0.03]"
                        >
                          <div className="text-[13px] text-white/64">{item.id}</div>
                          <div className="min-w-0 pl-2">
                            <div className="truncate text-[13px] font-medium text-white">
                              {item.fileName}.{item.type}
                            </div>
                            <div
                              className="mt-0.5 truncate text-[13px] leading-5 text-white/44"
                            >
                              {getDocumentSummaryText(item.summary)}
                            </div>
                          </div>
                          <div className="truncate text-[13px] leading-5 text-white/56">{knowledgeBaseNameMap.get(item.knowledgeBaseId) ?? "--"}</div>
                          <div className="text-[13px] text-white/72">{item.type}</div>
                          <div className="text-[13px] text-white/56">{item.size}</div>
                          <div className="text-[13px] text-white/56">{item.chunkCount}</div>
                          <div className="flex min-w-0 items-start">
                            <span
                              className="inline-flex max-w-full items-center border px-2 py-1 text-[9px] uppercase tracking-[0.2em] text-white/80"
                              style={{
                                borderColor: `${getDocumentStatusAccent(item.status, chapter)}80`,
                                background: `${getDocumentStatusAccent(item.status, chapter)}14`,
                              }}
                            >
                              {item.status}
                            </span>
                          </div>
                          <div className="whitespace-nowrap text-[12px] text-white/56">{item.uploadedAt}</div>
                          <div className="flex flex-nowrap items-center gap-1.5">
                            <ManagementActionButton accent={chapter.secondary} secondary onClick={() => setPreviewDocumentId(item.id)} className="h-8 justify-center px-2.5 text-[8px] tracking-[0.18em]">
                              <Eye size={12} />
                              Preview
                            </ManagementActionButton>
                            <ManagementActionButton accent="#ff6c86" secondary onClick={() => requestDeleteDocument(item.id)} className="h-8 justify-center px-2.5 text-[8px] tracking-[0.18em]">
                              <Trash2 size={12} />
                              Delete
                            </ManagementActionButton>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
            <ManagementPagination
              accent={chapter.accent}
              page={documentPage}
              pageSize={documentPageSize}
              totalItems={documentTotalItems}
              rangeStart={documentRangeStart}
              rangeEnd={documentRangeEnd}
              onPageChange={setDocumentPage}
            />
          </InteractiveCutPanel>
        </motion.div>
      </div>

      <ManagementModal
        open={knowledgeBaseModalOpen}
        accent={chapter.secondary}
        kicker={knowledgeBaseModalMode === "edit" ? "Edit Knowledge Base" : "Create Knowledge Base"}
        title={knowledgeBaseModalMode === "edit" ? "更新知识库" : "新建知识库"}
        onClose={() => setKnowledgeBaseModalOpen(false)}
      >
        <form
          className="grid gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            void handleSaveKnowledgeBase();
          }}
        >
          <div className="grid gap-4 md:grid-cols-2">
            <div className="grid gap-2">
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">知识库名称</div>
              <ManagementFieldShell accent={chapter.secondary} icon={<Database size={16} />}>
                <input
                  value={knowledgeBaseDraft.name}
                  onChange={(event) => setKnowledgeBaseDraft((current) => ({ ...current, name: event.target.value }))}
                  placeholder="请输入知识库名称"
                  className="w-full bg-transparent text-sm text-white outline-none placeholder:text-white/24"
                />
              </ManagementFieldShell>
            </div>
            <div className="grid gap-2">
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">创建者</div>
              <ManagementFieldShell accent={chapter.secondary} icon={<BrainCircuit size={16} />}>
                <div className="text-sm text-white/72">he114TnT</div>
              </ManagementFieldShell>
            </div>
          </div>

          <div className="grid gap-2">
            <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">描述</div>
            <div
              className="relative overflow-hidden border border-white/10 bg-white/[0.03] px-4 py-3"
              style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
            >
              <textarea
                value={knowledgeBaseDraft.description}
                onChange={(event) => setKnowledgeBaseDraft((current) => ({ ...current, description: event.target.value }))}
                placeholder="请输入知识库描述，用于说明该库收纳的文档范围与适用场景。"
                className="min-h-[7rem] w-full resize-none bg-transparent text-sm leading-6 text-white outline-none placeholder:text-white/24"
              />
              <span className="pointer-events-none absolute inset-x-3 bottom-0 h-px opacity-55" style={{ background: `linear-gradient(90deg, transparent, ${chapter.secondary}, transparent)` }} />
            </div>
          </div>

          <div className="flex flex-wrap justify-end gap-3 border-t border-white/[0.06] pt-4">
            <ManagementActionButton accent={chapter.secondary} secondary onClick={() => setKnowledgeBaseModalOpen(false)}>
              Cancel
            </ManagementActionButton>
            <ManagementActionButton accent={chapter.secondary} type="submit" icon={<Plus size={14} />}>
              {knowledgeBaseModalMode === "edit" ? "Save Changes" : "Create Knowledge Base"}
            </ManagementActionButton>
          </div>
        </form>
      </ManagementModal>

      <ManagementModal
        open={documentModalOpen}
        accent={chapter.accent}
        kicker="Upload Document"
        title="上传文档"
        onClose={() => setDocumentModalOpen(false)}
      >
        <motion.form
          className="grid gap-4"
          initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.28, ease: [0.16, 1, 0.3, 1] } }}
          onSubmit={(event) => {
            event.preventDefault();
            void handleSaveDocument();
          }}
        >
          {/* 文件选择区域 */}
          <div className="grid gap-2">
            <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">选择文件（PDF / DOCX）</div>
            <label
              className="relative flex cursor-pointer items-center gap-3 overflow-hidden border border-dashed border-white/16 bg-white/[0.03] px-4 py-3 transition-colors hover:border-white/28"
              style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
            >
              <UploadCloud size={20} className="shrink-0 text-white/40" />
              <span className="truncate text-sm text-white/60">
                {documentFile ? documentFile.name : "点击选择文件或拖放到此处"}
              </span>
              <input
                type="file"
                accept=".pdf,.docx"
                className="absolute inset-0 cursor-pointer opacity-0"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  setDocumentFile(file);
                  if (file) {
                    const name = file.name.replace(/\.[^.]+$/, "");
                    const ext = file.name.split(".").pop()?.toUpperCase() || "PDF";
                    const sizeMB = (file.size / (1024 * 1024)).toFixed(1) + " MB";
                    setDocumentDraft((current) => ({
                      ...current,
                      fileName: name,
                      type: ext as DocumentRecord["type"],
                      size: sizeMB,
                    }));
                    setUploadStage("idle");
                    setUploadProgress(0);
                    setUploadDetail("等待上传");
                    setDocumentSelectedKbId(null);
                    setAiClassifyResult(null);
                    void classifyFile(file.name);
                  }
                }}
              />
            </label>
          </div>

          {/* AI 知识库分类建议 */}
          {aiClassifying && (
            <div className="flex items-center gap-2 rounded border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-white/60">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white/60" />
              AI 正在分析文档归属知识库…
            </div>
          )}
          {aiClassifyResult && !aiClassifying && (
            <div className="rounded border border-white/10 bg-white/[0.03] px-4 py-3" onMouseMove={handleFramePointerMove} onMouseLeave={handleFramePointerLeave}>
              <div className="mb-1 text-[10px] uppercase tracking-[0.28em] text-white/36">AI 分类建议</div>
              {aiClassifyResult.suggestedKbId ? (
                <div className="text-sm text-white/80">
                  <span className="text-white/50">推荐归入：</span>
                  <span className="font-medium" style={{ color: chapter.accent }}>{aiClassifyResult.suggestedKbName}</span>
                  {aiClassifyResult.reason && <span className="ml-2 text-white/40">— {aiClassifyResult.reason}</span>}
                </div>
              ) : aiClassifyResult.newKbName ? (
                <div className="text-sm text-white/80">
                  <span className="text-white/50">建议创建新知识库：</span>
                  <span className="font-medium" style={{ color: chapter.secondary }}>{aiClassifyResult.newKbName}</span>
                  {aiClassifyResult.reason && <span className="ml-2 text-white/40">— {aiClassifyResult.reason}</span>}
                </div>
              ) : (
                <div className="text-sm text-white/50">{aiClassifyResult.reason || "无法确定分类"}</div>
              )}
            </div>
          )}

          {/* 知识库选择（可覆盖 AI 建议） */}
          <div className="grid gap-2">
            <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">
              目标知识库
              {documentSelectedKbId && aiClassifyResult?.suggestedKbId && documentSelectedKbId !== aiClassifyResult.suggestedKbId && (
                <span className="ml-2 normal-case tracking-normal text-white/30">（已手动覆盖 AI 建议）</span>
              )}
            </div>
            <ManagementFieldShell accent={chapter.secondary} icon={<Database size={16} />}>
              <select
                value={documentSelectedKbId ?? ""}
                onChange={(event) => setDocumentSelectedKbId(event.target.value ? Number(event.target.value) : null)}
                className="w-full appearance-none bg-transparent pr-6 text-sm text-white outline-none"
              >
                <option value="" className="bg-[#0a0c11] text-white">
                  {aiClassifyResult?.newKbName ? `AI 新建: ${aiClassifyResult.newKbName}` : "— 请选择知识库 —"}
                </option>
                {knowledgeBases.map((kb) => (
                  <option key={kb.id} value={kb.id} className="bg-[#0a0c11] text-white">
                    {kb.name}
                  </option>
                ))}
              </select>
              <ChevronDown size={14} className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-white/28" />
            </ManagementFieldShell>
          </div>

          {(documentFile || documentIndexing || uploadStage === "error") && (
            <div
              className="rounded border border-white/10 bg-white/[0.03] px-4 py-3"
              onMouseMove={handleFramePointerMove}
              onMouseLeave={handleFramePointerLeave}
            >
              <div className="mb-1 text-[10px] uppercase tracking-[0.28em] text-white/36">上传进度</div>
              <div className="mb-2 text-sm text-white/72">
                {getUploadStageLabel(uploadStage)}
                <span className="ml-2 text-white/44">{uploadDetail}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded bg-white/10">
                <div
                  className={`h-full transition-all duration-300 ${uploadStage === "error" ? "bg-[#ff6c86]" : "bg-white/70"}`}
                  style={{ width: `${Math.max(0, Math.min(100, uploadProgress))}%` }}
                />
              </div>
              <div className="mt-2 text-[11px] text-white/46">
                上传文件 → 语义分块与索引 → 写入文档记录
              </div>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2">
            <div className="grid gap-2 md:col-span-2">
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">文件名</div>
              <ManagementFieldShell accent={chapter.accent} icon={<FileText size={16} />}>
                <input
                  value={documentDraft.fileName}
                  onChange={(event) => setDocumentDraft((current) => ({ ...current, fileName: event.target.value }))}
                  placeholder="请输入文件名，不含后缀"
                  className="w-full bg-transparent text-sm text-white outline-none placeholder:text-white/24"
                />
              </ManagementFieldShell>
            </div>

            <div className="grid gap-2">
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">文件类型</div>
              <ManagementFieldShell accent={chapter.secondary} icon={<FileDigit size={16} />}>
                <div className="text-sm text-white/72">{documentDraft.type || "—"}</div>
              </ManagementFieldShell>
            </div>

            <div className="grid gap-2">
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">文件大小</div>
              <ManagementFieldShell accent={chapter.accent} icon={<Database size={16} />}>
                <div className="text-sm text-white/72">{documentDraft.size || "选择文件后自动检测"}</div>
              </ManagementFieldShell>
            </div>
          </div>

          <div className="grid gap-2">
            <div className="text-[10px] uppercase tracking-[0.28em] text-white/36">文档摘要（可选）</div>
            <div
              className="relative overflow-hidden border border-white/10 bg-white/[0.03] px-4 py-3"
              style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
            >
              <textarea
                value={documentDraft.summary}
                onChange={(event) => setDocumentDraft((current) => ({ ...current, summary: event.target.value }))}
                placeholder="请输入文档摘要，用于后续搜索、筛选和预览说明。"
                className="min-h-[5rem] w-full resize-none bg-transparent text-sm leading-6 text-white outline-none placeholder:text-white/24"
              />
              <span className="pointer-events-none absolute inset-x-3 bottom-0 h-px opacity-55" style={{ background: `linear-gradient(90deg, transparent, ${chapter.accent}, transparent)` }} />
            </div>
          </div>

          <div className="flex flex-wrap justify-end gap-3 border-t border-white/[0.06] pt-4">
            <ManagementActionButton accent={chapter.accent} secondary onClick={() => setDocumentModalOpen(false)}>
              Cancel
            </ManagementActionButton>
            <ManagementActionButton accent={chapter.accent} type="submit" icon={<UploadCloud size={14} />} disabled={(!documentFile && !documentDraft.fileName.trim()) || documentIndexing || aiClassifying}>
              {documentIndexing ? "索引中…" : "上传并索引"}
            </ManagementActionButton>
          </div>
        </motion.form>
      </ManagementModal>

      <ManagementModal
        open={previewDocument !== null}
        accent={previewDocument ? getDocumentStatusAccent(previewDocument.status, chapter) : chapter.secondary}
        kicker="Document Preview"
        title={previewDocument ? `${previewDocument.fileName}.${previewDocument.type}` : "文档预览"}
        onClose={() => setPreviewDocumentId(null)}
        widthClass="max-w-3xl"
      >
        {previewDocument ? (
          <div className="grid gap-5">
            <div className="grid gap-3 md:grid-cols-4">
              {[
                ["所属知识库", knowledgeBaseNameMap.get(previewDocument.knowledgeBaseId) ?? "--"],
                ["状态", previewDocument.status],
                ["大小", previewDocument.size],
                ["分块数", String(previewDocument.chunkCount)],
              ].map(([label, value], index) => (
                <div key={label} className="border border-white/8 bg-white/[0.02] px-4 py-4" onMouseMove={handleFramePointerMove} onMouseLeave={handleFramePointerLeave}>
                  <div className="text-[10px] uppercase tracking-[0.28em] text-white/34">{label}</div>
                  <div className="mt-2 font-display text-base uppercase tracking-[0.14em]" style={{ color: index === 1 ? getDocumentStatusAccent(previewDocument.status, chapter) : "#f5f7fa" }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            <div className="border border-white/8 bg-white/[0.02] px-4 py-4" onMouseMove={handleFramePointerMove} onMouseLeave={handleFramePointerLeave}>
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/34">摘要</div>
              <div className="mt-3 text-sm leading-7 text-white/62">{getDocumentSummaryText(previewDocument.summary)}</div>
            </div>

            <div className="border border-white/8 bg-white/[0.02] px-4 py-4" onMouseMove={handleFramePointerMove} onMouseLeave={handleFramePointerLeave}>
              <div className="text-[10px] uppercase tracking-[0.28em] text-white/34">上传时间</div>
              <div className="mt-3 text-sm leading-7 text-white/62">{previewDocument.uploadedAt}</div>
            </div>

            <div className="flex flex-wrap justify-end gap-3 border-t border-white/[0.06] pt-4">
              <ManagementActionButton accent={chapter.secondary} secondary onClick={() => setPreviewDocumentId(null)}>
                Close
              </ManagementActionButton>
            </div>
          </div>
        ) : null}
      </ManagementModal>

      <ManagementModal
        open={pendingDelete !== null}
        accent="#ff6c86"
        kicker="Confirm Delete"
        title={pendingDelete?.type === "knowledgeBase" ? "确认删除知识库" : "确认删除文档"}
        onClose={() => setPendingDelete(null)}
        widthClass="max-w-xl"
      >
        <div className="grid gap-5">
          <div
            className="relative overflow-hidden border border-white/10 bg-white/[0.03] px-4 py-4"
            style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
          >
            {pendingDelete?.type === "knowledgeBase" && pendingDeleteKnowledgeBase ? (
              <>
                <div className="text-[10px] uppercase tracking-[0.3em] text-white/34">Delete Target</div>
                <div className="mt-2 font-display text-lg uppercase tracking-[0.14em] text-white">
                  {pendingDeleteKnowledgeBase.name}
                </div>
                <div className="mt-3 text-sm leading-7 text-white/58">
                  将同时删除该知识库下的 {documentCountByKnowledgeBase.get(pendingDeleteKnowledgeBase.id) ?? 0} 份文档，这一步不可撤回。
                </div>
              </>
            ) : pendingDelete?.type === "document" && pendingDeleteDocument ? (
              <>
                <div className="text-[10px] uppercase tracking-[0.3em] text-white/34">Delete Target</div>
                <div className="mt-2 font-display text-lg uppercase tracking-[0.14em] text-white">
                  {pendingDeleteDocument.fileName}.{pendingDeleteDocument.type}
                </div>
                <div className="mt-3 text-sm leading-7 text-white/58">删除后文档将从当前知识库移除，这一步不可撤回。</div>
              </>
            ) : (
              <div className="text-sm leading-7 text-white/58">删除目标不存在，可能已经被移除。</div>
            )}
            <span className="pointer-events-none absolute inset-x-3 bottom-0 h-px opacity-55" style={{ background: "linear-gradient(90deg, transparent, #ff6c86, transparent)" }} />
          </div>

          <div className="flex flex-wrap justify-end gap-3 border-t border-white/[0.06] pt-4">
            <ManagementActionButton accent="#ff6c86" secondary onClick={() => setPendingDelete(null)}>
              Cancel
            </ManagementActionButton>
            <ManagementActionButton accent="#ff6c86" icon={<Trash2 size={14} />} onClick={() => void handleConfirmDelete()}>
              Confirm Delete
            </ManagementActionButton>
          </div>
        </div>
      </ManagementModal>
    </div>
  );
}

function RetrievalPage({ chapter }: { chapter: Chapter }) {
  return (
    <div className="relative min-h-full">
      <StageHeader chapter={chapter} />
      <motion.div
        initial={{ opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1, transition: { duration: 0.7 } }}
        className="mb-5 grid gap-4"
      >
        <InteractiveCutPanel
          accent={chapter.secondary}
          secondary={chapter.accent}
          kicker="Observation Scope"
          title="This Page Explains The Backend"
        >
          <div className="grid gap-4 md:grid-cols-[auto_1fr]">
            <div className="flex h-12 w-12 items-center justify-center border border-white/10 bg-white/[0.03]">
              <Radar size={20} color={chapter.secondary} />
            </div>
            <div>
              <div className="font-display text-lg uppercase tracking-[0.15em] text-white">
                液压系统压力下降 / ATA 29 / 回油过滤器 / 执行机构泄漏
              </div>
              <div className="mt-2 text-sm leading-6 text-white/58">
                前端只提交统一查询并读取 trace record。Elasticsearch、向量检索、融合重排和证据绑定都在后端自动完成，
                这个页面只负责把那条后台链路可视化，而不是让用户直接操作检索引擎。
              </div>
            </div>
          </div>
        </InteractiveCutPanel>
      </motion.div>
      <div className="grid gap-5 xl:grid-cols-[0.98fr_1.04fr_0.98fr]">
        <motion.div initial={{ opacity: 0, x: -60 }} animate={{ opacity: 1, x: 0, transition: { duration: 0.55 } }}>
          <InteractiveCutPanel accent={chapter.accent} secondary={chapter.secondary} kicker="Backend Service" title="Recall Stream" className="h-full min-h-[28rem]">
            <div className="space-y-4">
              <div className="flex items-center justify-between border border-white/10 bg-white/[0.008] px-4 py-4">
                <div>
                  <div className="text-[10px] uppercase tracking-[0.3em] text-white/35">Elasticsearch</div>
                  <div className="mt-1 font-display text-lg uppercase tracking-[0.16em] text-white">Broad Recall</div>
                </div>
                <FileSearch size={20} color={chapter.accent} />
              </div>
              {["ATA 章节编号匹配", "故障码精确命中", "缩略语扩展召回", "航空术语词典增强"].map((item) => (
                <motion.div
                  key={item}
                  whileHover={{ x: 8, scale: 1.015, borderColor: `${chapter.accent}55` }}
                  className="border border-white/8 bg-white/[0.006] px-4 py-3"
                >
                  <div className="flex items-center justify-between text-sm text-white/75">
                    <span>{item}</span>
                    <span className="font-display text-xs uppercase tracking-[0.22em]" style={{ color: chapter.accent }}>
                      OBSERVE
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </InteractiveCutPanel>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.78 }}
          animate={{ opacity: 1, scale: 1, transition: { duration: 0.7, delay: 0.06 } }}
        >
          <InteractiveCutPanel accent={chapter.secondary} secondary={chapter.accent} kicker="Fusion Core" title="Automatic Merge / Rerank / Ground" className="h-full min-h-[28rem]">
            <div className="relative flex h-full min-h-[23rem] flex-col items-center justify-center overflow-hidden">
              <motion.div
                className="absolute h-[17rem] w-[17rem] rounded-full border"
                style={{ borderColor: `${chapter.accent}80`, boxShadow: `0 0 40px ${chapter.accent}35` }}
                animate={{ rotate: 360 }}
                transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
              />
              <motion.div
                className="absolute h-[12rem] w-[12rem] border"
                style={{ borderColor: `${chapter.secondary}80`, clipPath: "polygon(50% 0, 100% 50%, 50% 100%, 0 50%)" }}
                animate={{ rotate: -360 }}
                transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
              />
              <motion.div
                className="relative z-10 h-[8rem] w-[8rem] rounded-full"
                style={{
                  background: `radial-gradient(circle at 35% 35%, ${chapter.secondary}, ${chapter.accent} 60%)`,
                  boxShadow: `0 0 80px ${chapter.accent}66`,
                }}
                animate={{ scale: [0.92, 1.05, 0.92] }}
                transition={{ duration: 2.8, repeat: Infinity }}
              />
              <div className="relative z-10 mt-8 text-center">
                <div className="text-[10px] uppercase tracking-[0.34em] text-white/35">Fusion Layer</div>
                <div className="mt-2 font-display text-2xl uppercase tracking-[0.18em] text-white">No Direct User Control</div>
                <div className="mt-3 max-w-sm text-sm leading-6 text-white/58">
                  这一层由后端服务自动执行。前端只展示融合后的证据质量、来源绑定状态和检索链路，不暴露引擎内部操作。
                </div>
              </div>
            </div>
          </InteractiveCutPanel>
        </motion.div>
        <motion.div initial={{ opacity: 0, x: 60 }} animate={{ opacity: 1, x: 0, transition: { duration: 0.55, delay: 0.05 } }}>
          <InteractiveCutPanel accent={chapter.secondary} secondary={chapter.accent} kicker="Backend Service" title="Semantic Rank Stream" className="h-full min-h-[28rem]">
            <div className="space-y-4">
              <div className="flex items-center justify-between border border-white/10 bg-white/[0.008] px-4 py-4">
                <div>
                  <div className="text-[10px] uppercase tracking-[0.3em] text-white/35">Vector Store</div>
                  <div className="mt-1 font-display text-lg uppercase tracking-[0.16em] text-white">Precision Ranking</div>
                </div>
                <BrainCircuit size={20} color={chapter.secondary} />
              </div>
              {["BGE-M3 / m3e 嵌入选型", "段落级语义相似度", "相邻 chunk 连贯修复", "跨文档关联分析"].map((item) => (
                <motion.div
                  key={item}
                  whileHover={{ x: 8, scale: 1.015, borderColor: `${chapter.secondary}55` }}
                  className="border border-white/8 bg-white/[0.006] px-4 py-3"
                >
                  <div className="flex items-center justify-between text-sm text-white/75">
                    <span>{item}</span>
                    <span className="font-display text-xs uppercase tracking-[0.22em]" style={{ color: chapter.secondary }}>
                      TRACE
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </InteractiveCutPanel>
        </motion.div>
      </div>
      <div className="mt-5 grid gap-5 xl:grid-cols-[1.08fr_0.92fr]">
        <InteractiveCutPanel accent={chapter.accent} secondary={chapter.secondary} kicker="Trace Timeline" title="Backend Execution Record">
          <div className="space-y-3">
            {retrievalTraceSteps.map((step, index) => (
              <motion.div
                key={step.stage}
                className="grid gap-3 border border-white/8 bg-white/[0.008] px-4 py-4 md:grid-cols-[9rem_1fr_5rem]"
                style={{ clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 16px), calc(100% - 16px) 100%, 0 100%)" }}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0, transition: { duration: 0.4, delay: 0.08 * index } }}
                whileHover={{ y: -5, scale: 1.01, borderColor: `${chapter.accent}55` }}
              >
                <div>
                  <div className="font-display text-sm uppercase tracking-[0.18em]" style={{ color: chapter.accent }}>
                    {step.stage}
                  </div>
                  <div className="mt-2 text-[10px] uppercase tracking-[0.3em] text-white/35">{step.system}</div>
                </div>
                <div className="text-sm leading-6 text-white/58">{step.detail}</div>
                <div className="flex items-center justify-end font-display text-sm uppercase tracking-[0.18em]" style={{ color: chapter.secondary }}>
                  {step.score}
                </div>
              </motion.div>
            ))}
          </div>
        </InteractiveCutPanel>

        <div className="grid gap-5">
          <InteractiveCutPanel accent={chapter.secondary} secondary={chapter.accent} kicker="Observation Summary" title="What The Frontend Shows">
            <div className="space-y-3">
              {retrievalObservations.map((result, index) => (
                <motion.div
                  key={result.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0, transition: { duration: 0.4, delay: 0.1 * index } }}
                  whileHover={{ y: -5, scale: 1.012, borderColor: `${chapter.secondary}55` }}
                  className="border border-white/8 bg-white/[0.008] px-4 py-4"
                  style={{ clipPath: "polygon(0 0, 100% 0, 100% calc(100% - 16px), calc(100% - 16px) 100%, 0 100%)" }}
                >
                  <div className="flex items-center justify-between gap-4">
                    <div className="text-[10px] uppercase tracking-[0.34em] text-white/35">{result.label}</div>
                    <div className="font-display text-xs uppercase tracking-[0.22em]" style={{ color: index === 1 ? chapter.secondary : chapter.accent }}>
                      {result.score}
                    </div>
                  </div>
                  <div className="mt-2 text-sm leading-6 text-white/62">{result.title}</div>
                </motion.div>
              ))}
            </div>
          </InteractiveCutPanel>
          <InteractiveCutPanel accent={chapter.accent} secondary={chapter.secondary} kicker="Current Evidence" title="Fused Context Output">
            <div className="space-y-3">
              {evidenceCards.map((item, index) => (
                <motion.div
                  key={item.doc}
                  className="border border-white/8 bg-white/[0.008] px-4 py-4"
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0, transition: { delay: 0.1 * index } }}
                  whileHover={{ x: 6, scale: 1.01, borderColor: `${chapter.accent}55` }}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-medium text-white">{item.doc}</div>
                      <div className="mt-2 text-sm leading-6 text-white/55">{item.section}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-display text-base uppercase tracking-[0.18em]" style={{ color: chapter.secondary }}>
                        {item.score}
                      </div>
                      <div className="text-[10px] uppercase tracking-[0.28em] text-white/35">{item.page}</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </InteractiveCutPanel>
        </div>
      </div>
    </div>
  );
}

function AviationAssistantPage({
  chapter,
  goSystem,
}: {
  chapter: Chapter;
  goSystem: () => void;
}) {
  const [conversation, setConversation] = useState<AssistantConversation | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [documentTypeFilter, setDocumentTypeFilter] = useState<AssistantDocumentType | "ALL">("ALL");
  const [aircraftModelFilter, setAircraftModelFilter] = useState("ALL");
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [evidencePanelOpen, setEvidencePanelOpen] = useState(false);
  const [evidenceMode, setEvidenceMode] = useState<"all" | "active">("active");
  const [previewPage, setPreviewPage] = useState(1);
  const [historyItems, setHistoryItems] = useState<AssistantHistoryItem[]>(() => createInitialAssistantHistory());
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [viewportWidth, setViewportWidth] = useState(() => (typeof window === "undefined" ? 1440 : window.innerWidth));
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleResize = () => {
      setViewportWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    const scrollNode = chatScrollRef.current;
    if (!scrollNode) return;

    const timer = window.setTimeout(() => {
      scrollNode.scrollTo({ top: scrollNode.scrollHeight, behavior: "smooth" });
    }, 36);

    return () => window.clearTimeout(timer);
  }, [conversation]);

  const isDrawerMode = viewportWidth < 1440;

  const citationsById = useMemo(
    () => new Map((conversation?.citations ?? []).map((citation) => [citation.id, citation])),
    [conversation],
  );
  const sourcesById = useMemo(
    () => new Map((conversation?.sources ?? []).map((source) => [source.id, source])),
    [conversation],
  );

  const filteredSources = useMemo(
    () => (conversation?.sources ?? []).filter((source) => matchesAssistantFilters(source, documentTypeFilter, aircraftModelFilter)),
    [conversation, documentTypeFilter, aircraftModelFilter],
  );
  const filteredSourceIds = useMemo(() => new Set(filteredSources.map((source) => source.id)), [filteredSources]);

  const selectedCitation = selectedCitationId ? citationsById.get(selectedCitationId) ?? null : null;
  const selectedSource = selectedCitation ? sourcesById.get(selectedCitation.sourceId) ?? null : null;

  const visibleCitations = useMemo(() => {
    if (!conversation) return [];

    const filtered = conversation.citations.filter((citation) => filteredSourceIds.has(citation.sourceId));

    if (selectedCitation && !filtered.some((item) => item.id === selectedCitation.id)) {
      return [selectedCitation, ...filtered];
    }

    return filtered;
  }, [conversation, filteredSourceIds, selectedCitation]);

  const hydrateConversation = (nextConversation: AssistantConversation) => {
    const defaultCitation =
      nextConversation.citations.find((citation) => citation.id === nextConversation.defaultCitationId) ??
      nextConversation.citations[0] ??
      null;

    setConversation(nextConversation);
    setSelectedCitationId(defaultCitation?.id ?? null);
    setPreviewPage(defaultCitation?.page ?? 1);
    setEvidenceMode("active");
    setEvidencePanelOpen(false);
  };

  const runConversation = async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || loading) return;

    setDraft("");
    setLoading(true);

    // 先显示用户消息（loading 状态）
    const pendingConversation: AssistantConversation = {
      id: `pending-${Date.now()}`,
      quickPrompt: trimmedQuestion,
      userQuestion: trimmedQuestion,
      aircraftModel: "ALL",
      documentTypeFocus: "ALL",
      defaultCitationId: "",
      messages: [{ id: `pending-user`, role: "user", content: trimmedQuestion }],
      citations: [],
      sources: [],
    };
    setConversation(pendingConversation);

    try {
      // 调用真实 API，失败则 fallback 到 mock
      const apiConversation = await callQAApi(trimmedQuestion);
      const resolvedConversation = apiConversation ?? resolveAssistantConversation(trimmedQuestion);

      const createdAt = Date.now();
      const historyEntry: AssistantHistoryItem = {
        id: `${resolvedConversation.id}-${createdAt}`,
        title: trimmedQuestion,
        question: trimmedQuestion,
        createdAt,
        preview: buildAssistantHistoryPreview(resolvedConversation),
        conversation: cloneAssistantConversation(resolvedConversation, trimmedQuestion),
      };

      hydrateConversation(resolvedConversation);
      setHistoryItems((current) => [
        historyEntry,
        ...current.filter((item) => item.question !== trimmedQuestion).slice(0, 19),
      ]);
      setActiveHistoryId(historyEntry.id);
      setDocumentTypeFilter(resolvedConversation.documentTypeFocus);
      setAircraftModelFilter(resolvedConversation.aircraftModel);

      // 异步生成对话标题（不阻塞 UI）
      const answerText = resolvedConversation.messages
        .find((m) => m.role === "assistant")
        ?.segments?.map((s) => s.text)
        .join(" ") ?? "";
      void summarizeConversationTitle(trimmedQuestion, answerText).then((title) => {
        setHistoryItems((current) =>
          current.map((h) => (h.id === historyEntry.id ? { ...h, title } : h)),
        );
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRestoreHistory = (item: AssistantHistoryItem) => {
    hydrateConversation(cloneAssistantConversation(item.conversation, item.question));
    setDocumentTypeFilter(item.conversation.documentTypeFocus);
    setAircraftModelFilter(item.conversation.aircraftModel);
    setActiveHistoryId(item.id);
    setDraft("");
  };

  const handleCitationClick = (citationId: string) => {
    const nextCitation = citationsById.get(citationId);
    if (!nextCitation) return;

    setSelectedCitationId(citationId);
    setPreviewPage(nextCitation.page);
    setEvidenceMode("active");
    setEvidencePanelOpen(true);
  };

  const handleToggleEvidencePanel = () => {
    if (!conversation) return;

    if (!selectedCitationId) {
      const fallbackCitation = conversation.citations[0] ?? null;
      if (fallbackCitation) {
        setSelectedCitationId(fallbackCitation.id);
        setPreviewPage(fallbackCitation.page);
      }
    }

    setEvidencePanelOpen((current) => !current);
  };

  const resetConversationShell = () => {
    setConversation(null);
    setDraft("");
    setSelectedCitationId(null);
    setEvidencePanelOpen(false);
    setEvidenceMode("active");
    setPreviewPage(1);
    setActiveHistoryId(null);
  };

  const emptyState = (
    <motion.div
      initial={{ opacity: 0, y: 18, filter: "blur(10px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.42, ease: [0.16, 1, 0.3, 1] } }}
      className="mx-auto max-w-[46rem] px-4 py-8 text-center"
    >
      <div
        className="mx-auto flex h-16 w-16 items-center justify-center border border-white/10 bg-white/[0.03]"
        style={{ color: chapter.accent, clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
      >
        <Bot size={24} />
      </div>
      <div className="mt-5 font-display text-2xl uppercase tracking-[0.16em] text-white md:text-[2.05rem]">
        请输入航空维修问题
      </div>
      <div className="mt-3 text-sm leading-7 text-white/54">
        系统会基于维修手册、AMM、FIM、TSM 等文档给出带依据的回答，并支持点击引用直接查看来源证据。
      </div>
    </motion.div>
  );

  const evidencePanel = (
    <EvidencePanel
      chapter={chapter}
      citations={visibleCitations}
      selectedCitationId={selectedCitationId}
      onSelectCitation={handleCitationClick}
      onClose={() => setEvidencePanelOpen(false)}
      mode={evidenceMode}
      onModeChange={setEvidenceMode}
      source={selectedSource}
      citation={selectedCitation}
      previewPage={previewPage}
      onPreviewPageChange={setPreviewPage}
    />
  );

  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden">
      <div className={`grid min-h-0 flex-1 gap-2 ${evidencePanelOpen && !isDrawerMode ? "xl:grid-cols-[18rem_minmax(0,1fr)_34rem]" : "xl:grid-cols-[18rem_minmax(0,1fr)]"}`}>
        <motion.div
          initial={{ opacity: 0, x: -42, filter: "blur(14px)" }}
          animate={{ opacity: 1, x: 0, filter: "blur(0px)", transition: { duration: 0.52, ease: [0.16, 1, 0.3, 1] } }}
          className="min-w-0 min-h-0"
        >
          <div className="flex h-full min-h-0 flex-col overflow-hidden">
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
              <ManagementActionButton accent={chapter.accent} hoverOutlineOnly icon={<Plus size={14} />} onClick={resetConversationShell} className="h-8 w-full justify-start px-3 text-[8px] tracking-[0.22em]">
                新建会话
              </ManagementActionButton>
              <ManagementActionButton
                accent={chapter.secondary}
                secondary
                hoverOutlineOnly
                icon={<Trash2 size={14} />}
                onClick={() => {
                  resetConversationShell();
                  setHistoryItems([]);
                  setDocumentTypeFilter("ALL");
                  setAircraftModelFilter("ALL");
                }}
                className="h-8 w-full justify-start px-3 text-[8px] tracking-[0.22em]"
              >
                清空上下文
              </ManagementActionButton>
              <ManagementActionButton
                accent={chapter.secondary}
                hoverOutlineOnly
                secondary={!evidencePanelOpen}
                icon={<Eye size={14} />}
                onClick={handleToggleEvidencePanel}
                disabled={!conversation}
                className="h-8 w-full justify-start px-3 text-[8px] tracking-[0.22em] sm:col-span-2 xl:col-span-1"
              >
                {evidencePanelOpen ? "隐藏证据" : "显示证据"}
              </ManagementActionButton>
            </div>

            <div className="panel-scroll assistant-console-scroll mt-4 min-h-0 flex-1 overflow-y-auto">
              <div className="mb-3 text-[10px] uppercase tracking-[0.3em] text-white/34">历史记录</div>
              <div className="grid gap-1">
                {historyItems.length === 0 ? (
                  <div className="px-3 py-3 text-sm leading-6 text-white/42">暂无历史记录</div>
                ) : (
                  historyItems.map((item) => {
                    const active = activeHistoryId === item.id;

                    return (
                      <AssistantHistoryRow
                        key={item.id}
                        chapter={chapter}
                        item={item}
                        active={active}
                        onSelect={handleRestoreHistory}
                      />
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 28, filter: "blur(12px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.56, delay: 0.05, ease: [0.16, 1, 0.3, 1] } }}
          className="min-w-0 min-h-0 border-l border-white/[0.08] pl-4"
        >
          <CutPanel
            accent={chapter.accent}
            kicker={`System Chapter ${chapter.code}`}
            title={chapter.title}
            className="assistant-main-panel flex h-full min-h-0 flex-col overflow-hidden pt-1"
            headerClassName="mb-1"
            actions={
              <ManagementActionButton accent={chapter.secondary} secondary onClick={goSystem} className="h-11 px-4 text-[10px] tracking-[0.24em]">
                <ChevronLeft size={13} />
                Return to System
              </ManagementActionButton>
            }
          >
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <div ref={chatScrollRef} className="panel-scroll min-h-0 flex-1 overflow-y-auto pr-2">
                <ChatMessageList
                  chapter={chapter}
                  messages={conversation?.messages ?? []}
                  citationsById={citationsById}
                  selectedCitationId={selectedCitationId}
                  onCitationClick={handleCitationClick}
                  emptyState={emptyState}
                />
                {loading ? (
                  <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mr-auto mt-3 flex items-center gap-3 border border-white/10 bg-white/[0.03] px-5 py-4"
                    style={{ clipPath: "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%)" }}
                  >
                    <Bot size={16} color={chapter.accent} className="animate-pulse" />
                    <span className="text-sm text-white/50 animate-pulse">正在检索文档并生成回答...</span>
                  </motion.div>
                ) : null}
              </div>
              <ChatComposer
                chapter={chapter}
                value={draft}
                onChange={setDraft}
                onSubmit={() => runConversation(draft)}
              />
            </div>
          </CutPanel>
        </motion.div>

        <AnimatePresence initial={false}>
          {!isDrawerMode && evidencePanelOpen ? (
            <motion.div
              key="assistant-evidence-docked"
              initial={{ opacity: 0, x: 36, filter: "blur(12px)" }}
              animate={{ opacity: 1, x: 0, filter: "blur(0px)", transition: { duration: 0.42, ease: [0.16, 1, 0.3, 1] } }}
              exit={{ opacity: 0, transition: { duration: 0.001 } }}
              className="min-w-0 min-h-0 border-l border-white/[0.08] pl-0.5"
            >
              {evidencePanel}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      <AnimatePresence initial={false}>
        {isDrawerMode && evidencePanelOpen ? (
          <motion.div
            key="assistant-evidence-drawer"
            className="fixed inset-0 z-[70] bg-black/70 px-3 py-3 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.2 } }}
            onClick={() => setEvidencePanelOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, x: 36, filter: "blur(10px)" }}
              animate={{ opacity: 1, x: 0, filter: "blur(0px)", transition: { duration: 0.36, ease: [0.16, 1, 0.3, 1] } }}
              exit={{ opacity: 0, x: 36, filter: "blur(8px)", transition: { duration: 0.18 } }}
              className="ml-auto h-full w-full max-w-[42rem]"
              onClick={(event) => event.stopPropagation()}
            >
              {evidencePanel}
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

function DonutChart({ data, size = 140, strokeWidth = 18 }: { data: { label: string; value: number; color: string }[]; size?: number; strokeWidth?: number }) {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  let accumulated = 0;

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="drop-shadow-[0_0_12px_rgba(255,255,255,0.06)]">
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth} />
        {data.map((item) => {
          const ratio = item.value / total;
          const dashLength = circumference * ratio;
          const gap = circumference - dashLength;
          const offset = -circumference * accumulated + circumference * 0.25;
          accumulated += ratio;
          return (
            <circle
              key={item.label}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={item.color}
              strokeWidth={strokeWidth}
              strokeDasharray={`${dashLength} ${gap}`}
              strokeDashoffset={offset}
              strokeLinecap="butt"
              className="transition-all duration-700"
            />
          );
        })}
        <text x={size / 2} y={size / 2 - 6} textAnchor="middle" fill="white" fontSize="22" fontWeight="600" fontFamily="inherit">
          {total}
        </text>
        <text x={size / 2} y={size / 2 + 14} textAnchor="middle" fill="rgba(255,255,255,0.45)" fontSize="10" letterSpacing="0.12em" fontFamily="inherit">
          TOTAL
        </text>
      </svg>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {data.map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-[11px] text-white/60">
            <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: item.color }} />
            <span className="truncate">{item.label}</span>
            <span className="ml-auto tabular-nums text-white/40">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RagPage({ chapter, onAssistantLaunch, onUpload }: { chapter: Chapter; onAssistantLaunch: (rect: DOMRect | null) => void; onUpload: () => void }) {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseRecord[]>([]);

  useEffect(() => {
    let active = true;
    void Promise.all([
      knowledgeBaseDocumentService.listDocuments(),
      knowledgeBaseDocumentService.listKnowledgeBases(),
    ]).then(([docs, kbs]) => {
      if (!active) return;
      setDocuments(docs);
      setKnowledgeBases(kbs);
    });
    return () => { active = false; };
  }, []);

  const latestDocs = useMemo(() => {
    return documents.slice(0, 4).map((d) => ({
      type: d.type,
      state: d.status,
      title: d.fileName,
      description: d.summary || `${d.type} · ${d.size} · ${d.chunkCount} 个语义块`,
      footer: d.uploadedAt || "—",
    }));
  }, [documents]);

  const docTypeDist = useMemo(() => {
    const colors: Record<string, string> = { PDF: "#ff5e7e", DOCX: "#7fe0ff", PPTX: "#d6ff3f", XLSX: "#ffcb5b" };
    const counts = new Map<string, number>();
    documents.forEach((d) => counts.set(d.type, (counts.get(d.type) ?? 0) + 1));
    return Array.from(counts.entries()).map(([label, value]) => ({
      label,
      value,
      color: colors[label] || "#aaa",
    }));
  }, [documents]);

  const kbDist = useMemo(() => {
    const kbMap = new Map(knowledgeBases.map((kb) => [kb.id, kb.name]));
    const colors = ["#ff78c8", "#7fe0ff", "#d9ff52", "#ff9a3c", "#ff5e7e", "#d6ff3f", "#ffcb5b"];
    const counts = new Map<number, number>();
    documents.forEach((d) => counts.set(d.knowledgeBaseId, (counts.get(d.knowledgeBaseId) ?? 0) + 1));
    return Array.from(counts.entries()).map(([id, value], i) => ({
      label: kbMap.get(id) ?? `KB-${id}`,
      value,
      color: colors[i % colors.length],
    }));
  }, [documents, knowledgeBases]);

  const systemTabs = [
    { key: "history", label: "历史浏览", icon: <Eye size={14} /> },
    { key: "assistant", label: "智能检索问答", icon: <FileSearch size={14} /> },
    { key: "upload", label: "文档管理", icon: <FolderOpen size={14} /> },
  ] as const;

  const emptyDocHint = (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0, transition: { duration: 0.4, delay: 0.15 } }}
      className="flex flex-col items-center justify-center py-16 text-center"
    >
      <div className="flex h-14 w-14 items-center justify-center border border-white/10 bg-white/[0.03]" style={{ color: chapter.accent }}>
        <FileText size={22} />
      </div>
      <div className="mt-4 font-display text-base uppercase tracking-[0.14em] text-white/60">暂无文档</div>
      <div className="mt-2 max-w-xs text-[12px] leading-5 text-white/36">请先在「知识库和文档管理」中上传文档，数据将自动呈现在此处。</div>
    </motion.div>
  );

  return (
    <div className="relative flex h-full min-h-full flex-col">
      <div className="mb-4 flex flex-wrap gap-2">
        {systemTabs.map((tab) => (
          <GlowTileButton
            key={tab.key}
            accent={tab.key === "history" ? chapter.accent : chapter.secondary}
            active={tab.key === "history"}
            onClick={() => {
              if (tab.key === "assistant") onAssistantLaunch(null);
              else if (tab.key === "upload") onUpload();
            }}
            className="min-w-[10rem] px-4 py-2.5"
          >
            <div className="flex items-center gap-2">
              <span style={{ color: tab.key === "history" ? chapter.accent : "rgba(255,255,255,0.55)" }}>{tab.icon}</span>
              <span className="font-display text-sm uppercase tracking-[0.14em]" style={{ color: tab.key === "history" ? chapter.accent : "#f8f9fb" }}>
                {tab.label}
              </span>
            </div>
          </GlowTileButton>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 16 }}
        animate={{
          opacity: 1,
          scale: 1,
          y: 0,
          transition: { duration: 0.52, delay: 0.08, ease: [0.16, 1, 0.3, 1] },
        }}
        className="relative flex-1 min-h-0 bg-transparent px-0 pb-11 pt-0 md:pb-12"
      >
        <div className="relative z-10 grid h-full min-h-0 gap-5 xl:grid-cols-[1fr_1fr_1fr]">
          <SketchSection
            title="猜你想看"
            subtitle="最近上传的文档，优先展示最新入库内容。"
            description=""
            accent={chapter.accent}
            eyebrow="Dock 02"
            signal="Latest Documents"
          >
            {latestDocs.length === 0 ? (
              emptyDocHint
            ) : (
              <div className="grid content-start gap-3">
                {latestDocs.map((item, index) => (
                  <SketchDocCard
                    key={item.title}
                    accent={index % 2 === 0 ? chapter.accent : chapter.secondary}
                    type={item.type}
                    state={item.state}
                    title={item.title}
                    description={item.description}
                    footer={item.footer}
                    delay={0.08 * index}
                  />
                ))}
              </div>
            )}
          </SketchSection>
          <SketchSection
            title="最近看过"
            subtitle="保留最近浏览时间与文档线索，方便回到上次位置继续问答。"
            description=""
            accent={chapter.secondary}
            eyebrow="Cache 03"
            signal="History Recall"
            delay={0.08}
          >
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.4, delay: 0.2 } }}
              className="flex flex-col items-center justify-center py-16 text-center"
            >
              <div className="flex h-14 w-14 items-center justify-center border border-white/10 bg-white/[0.03]" style={{ color: chapter.secondary }}>
                <Eye size={22} />
              </div>
              <div className="mt-4 font-display text-base uppercase tracking-[0.14em] text-white/60">暂无浏览记录</div>
              <div className="mt-2 max-w-xs text-[12px] leading-5 text-white/36">使用智能问答检索后，浏览记录将自动出现在此处。</div>
            </motion.div>
          </SketchSection>
          <SketchSection
            title="分布情况"
            subtitle="文档类型与知识库的分布统计概览。"
            description=""
            accent={chapter.accent}
            eyebrow="Stats 04"
            signal="Distribution Overview"
            delay={0.16}
          >
            {documents.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0, transition: { duration: 0.4, delay: 0.25 } }}
                className="flex flex-col items-center justify-center py-16 text-center"
              >
                <div className="flex h-14 w-14 items-center justify-center border border-white/10 bg-white/[0.03]" style={{ color: chapter.accent }}>
                  <Database size={22} />
                </div>
                <div className="mt-4 font-display text-base uppercase tracking-[0.14em] text-white/60">暂无统计数据</div>
                <div className="mt-2 max-w-xs text-[12px] leading-5 text-white/36">上传文档后，分布统计将自动生成。</div>
              </motion.div>
            ) : (
              <div className="grid content-start gap-6 py-2">
                <div>
                  <div className="mb-3 text-[10px] uppercase tracking-[0.28em] text-white/40">文档类型分布</div>
                  <DonutChart data={docTypeDist} />
                </div>
                <div>
                  <div className="mb-3 text-[10px] uppercase tracking-[0.28em] text-white/40">知识库分布</div>
                  <DonutChart data={kbDist} />
                </div>
              </div>
            )}
          </SketchSection>
        </div>
      </motion.div>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState<AppRoute>(() => resolveRoute(window.location.pathname));
  const [clock, setClock] = useState("");
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const cursorX = useSpring(mouseX, { stiffness: 240, damping: 20, mass: 0.3 });
  const cursorY = useSpring(mouseY, { stiffness: 240, damping: 20, mass: 0.3 });

  useEffect(() => {
    const updateClock = () => {
      setClock(
        new Intl.DateTimeFormat("zh-CN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }).format(new Date()),
      );
    };

    updateClock();
    const timer = window.setInterval(updateClock, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const handleMove = (event: MouseEvent) => {
      mouseX.set(event.clientX);
      mouseY.set(event.clientY);
    };

    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, [mouseX, mouseY]);

  useEffect(() => {
    const handlePopState = () => {
      setRoute(resolveRoute(window.location.pathname));
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    const currentPath = normalizePathname(window.location.pathname);
    const canonicalPath = routePathMap[route];

    if (currentPath !== canonicalPath) {
      window.history.replaceState(window.history.state, "", canonicalPath);
    }
  }, [route]);

  const navigateToRoute = (nextRoute: AppRoute, options?: { replace?: boolean }) => {
    const nextPath = routePathMap[nextRoute];
    const currentPath = normalizePathname(window.location.pathname);

    if (currentPath !== nextPath) {
      window.history[options?.replace ? "replaceState" : "pushState"](window.history.state, "", nextPath);
    }

    setRoute(nextRoute);
  };

  const navigateToChapter = (chapterIndex: number) => {
    navigateToRoute(chapterRoutes[chapterIndex]);
  };

  const launchAssistantFromSystem = (_rect: DOMRect | null) => {
    navigateToRoute("assistant");
  };

  const enteredSystem = route !== "launch";
  const isAssistantWorkbench = route === "assistant";
  const activeChapter = getChapterIndexForRoute(route);
  const chapter = enteredSystem ? chapters[activeChapter] : introTheme;

  const stage = useMemo(() => {
    if (route === "launch") {
      return (
        <HomePage
          chapter={chapter}
          enterQA={() => navigateToRoute("system")}
          openTrace={() => navigateToRoute("trace")}
        />
      );
    }

    switch (route) {
      case "system":
        return <RagPage chapter={chapter} onAssistantLaunch={launchAssistantFromSystem} onUpload={() => navigateToRoute("upload")} />;
      case "trace":
        return <RetrievalPage chapter={chapter} />;
      case "upload":
        return <IntakePage chapter={chapter} goSystem={() => navigateToRoute("system")} />;
      case "assistant":
        return <AviationAssistantPage chapter={chapter} goSystem={() => navigateToRoute("system")} />;
      default:
        return null;
    }
  }, [chapter, route]);

  return (
    <div
      className="relative min-h-dvh overflow-hidden bg-[#07080b] font-body text-white"
      style={
        {
          "--accent": chapter.accent,
          "--accent-soft": chapter.accentSoft,
          "--secondary": chapter.secondary,
        } as CSSProperties
      }
    >
      <motion.div
        className="pointer-events-none absolute inset-0 opacity-90"
        animate={{
          background: `linear-gradient(135deg, rgba(7,8,11,1), rgba(11,13,16,1) 48%, rgba(8,9,12,1))`,
        }}
        transition={{ duration: 0.55 }}
      />
      <DigitalWaveField mouseX={mouseX} mouseY={mouseY} chapter={chapter} />
      <div className="pointer-events-none absolute inset-0 bg-grid bg-[size:72px_72px] opacity-[0.05]" />
      <div className="pointer-events-none absolute inset-0 [background:linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px)] [background-size:100%_3px] opacity-[0.08]" />

      <motion.div
        className="pointer-events-none fixed left-0 top-0 z-50 hidden h-4 w-4 rounded-full border border-white/60 mix-blend-difference md:block"
        style={{ x: cursorX, y: cursorY, translateX: "-50%", translateY: "-50%" }}
      />
      <motion.div
        className="pointer-events-none fixed left-0 top-0 z-50 hidden h-12 w-12 rounded-full border md:block"
        style={{
          x: cursorX,
          y: cursorY,
          translateX: "-50%",
          translateY: "-50%",
          borderColor: `${chapter.accent}55`,
          boxShadow: `0 0 18px ${chapter.accent}30`,
        }}
      />
      <motion.div
        className="pointer-events-none fixed left-0 top-0 z-40 hidden h-20 w-20 border md:block"
        style={{
          x: cursorX,
          y: cursorY,
          translateX: "-50%",
          translateY: "-50%",
          rotate: 45,
          borderColor: `${chapter.secondary}22`,
          background: `linear-gradient(135deg, ${chapter.secondary}12, transparent 55%)`,
          boxShadow: `0 0 26px ${chapter.secondary}14`,
        }}
      />

      <header className="relative px-5 py-5 md:px-8 xl:px-10 z-20">
        <motion.div
          animate={{ minHeight: 64 }}
          transition={headerMorphTransition}
          className="flex items-start gap-4 md:gap-6"
        >
          <motion.button
            type="button"
            onClick={() => {
              if (enteredSystem) navigateToRoute("launch");
            }}
            animate={{
              width: 64,
              height: 64,
            }}
            transition={headerMorphTransition}
            whileHover={{ y: -4, scale: 1.035 }}
            whileTap={{ scale: 0.98 }}
            className={`group relative flex shrink-0 items-center justify-center overflow-hidden border font-display tracking-[0.3em] transition-[border-color,box-shadow,background] duration-300 ${
              enteredSystem ? "cursor-pointer" : "cursor-default"
            }`}
            style={{
              borderColor: `${chapter.accent}70`,
              background: chapter.accentSoft,
            }}
          >
            <span
              className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
              style={{ background: `radial-gradient(circle at 35% 30%, ${chapter.accent}55 0%, transparent 72%)` }}
            />
            <span className="pointer-events-none absolute inset-x-2 bottom-0 h-px origin-left scale-x-0 bg-white/70 transition-transform duration-300 group-hover:scale-x-100" />
            <span
              className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
              style={{ boxShadow: `inset 0 0 0 1px ${chapter.accent}66, 0 0 24px ${chapter.accent}28` }}
            />
            <motion.span
              animate={{ fontSize: "1rem" }}
              transition={headerMorphTransition}
              className="relative z-10 flex items-center justify-center"
            >
              <svg viewBox="0 0 200 200" width="42" height="42" fill="none" xmlns="http://www.w3.org/2000/svg">
                {/* Outer ring */}
                <circle cx="100" cy="100" r="94" stroke="currentColor" strokeWidth="3" opacity="0.7" />
                <circle cx="100" cy="100" r="86" stroke="currentColor" strokeWidth="1.5" opacity="0.35" />
                {/* Radar dish / antenna base */}
                <path d="M100 28 L100 100" stroke="currentColor" strokeWidth="2.5" opacity="0.8" />
                <path d="M100 100 L60 160" stroke="currentColor" strokeWidth="2.5" opacity="0.6" />
                <path d="M100 100 L140 160" stroke="currentColor" strokeWidth="2.5" opacity="0.6" />
                {/* Radar waves */}
                <path d="M68 48 A60 60 0 0 1 132 48" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.55" />
                <path d="M78 38 A45 45 0 0 1 122 38" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.4" />
                <path d="M58 58 A75 75 0 0 1 142 58" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.7" />
                {/* Center point */}
                <circle cx="100" cy="100" r="5" fill="currentColor" opacity="0.9" />
                <circle cx="100" cy="100" r="9" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
                {/* Base platform */}
                <path d="M55 162 L145 162" stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity="0.7" />
                <path d="M65 170 L135 170" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.45" />
                {/* Text arc - 西电 */}
                <text x="100" y="188" textAnchor="middle" fill="currentColor" fontSize="18" fontWeight="bold" opacity="0.7" fontFamily="Microsoft YaHei UI, sans-serif" letterSpacing="0.15em">西电</text>
              </svg>
            </motion.span>
          </motion.button>

          <motion.div
            animate={{ height: "auto" }}
            transition={headerMorphTransition}
            className="relative min-w-0 flex-1"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key="default-header-center"
                variants={headerContentVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="flex min-w-0 items-start justify-between gap-6 pt-1"
              >
                <motion.div variants={headerContentVariants} className="min-w-0">
                  <div className="text-[10px] uppercase tracking-[0.42em] text-white/40">Xidian University Innovation Training Project</div>
                  <div className="mt-1 font-display text-lg uppercase tracking-[0.18em] text-white">Aerospace Knowledge Command Core</div>
                </motion.div>
                <motion.div variants={headerContentVariants} className="hidden shrink-0 items-center gap-8 self-center lg:mr-12 lg:flex xl:mr-16">
                  {headerTelemetryItems.map((item) => (
                    <motion.div key={item.label} variants={headerContentVariants} className="flex items-center gap-2 text-xs uppercase tracking-[0.24em] text-white/45">
                      <span>{item.icon}</span>
                      <span>{item.label}</span>
                    </motion.div>
                  ))}
                </motion.div>
              </motion.div>
            </AnimatePresence>
          </motion.div>

          <motion.div
            className="shrink-0 text-right"
            animate={{ paddingTop: 0 }}
            transition={headerMorphTransition}
          >
            <motion.div
              animate={{
                fontSize: 11,
                letterSpacing: "0.34em",
              }}
              transition={headerMorphTransition}
              className="uppercase text-white/35"
            >
              System Clock
            </motion.div>
            <motion.div
              animate={{
                marginTop: 4,
                fontSize: "2.35rem",
                letterSpacing: "0.2em",
              }}
              transition={headerMorphTransition}
              className="font-display uppercase leading-none text-white"
            >
              {clock}
            </motion.div>
          </motion.div>
        </motion.div>
      </header>

      <main
        className={`relative z-10 px-5 md:px-8 xl:px-10 ${
          isAssistantWorkbench ? "pt-0 pb-0" : "pt-2 pb-24"
        }`}
      >
        <AnimatePresence mode="wait">
          <motion.section
            key={route}
            {...(enteredSystem ? pageVariants[(activeChapter + 1) % pageVariants.length] : pageVariants[0])}
            className={
              isAssistantWorkbench
                ? "relative h-[calc(100dvh-8rem)] overflow-hidden rounded-[2rem] border border-white/8 bg-black/[0.015] pb-0 pl-2 pr-0.5 pt-2 shadow-panel md:pb-0 md:pl-3 md:pr-1 md:pt-3 xl:pb-0 xl:pl-4 xl:pr-1 xl:pt-4"
                : "relative h-[calc(100dvh-11rem)] overflow-hidden rounded-[2rem] border border-white/8 bg-black/[0.015] p-4 shadow-panel md:p-6 xl:p-7"
            }
          >
            <div className="h-full">
              <div className={`min-w-0 h-full overflow-x-hidden pr-0 ${isAssistantWorkbench ? "overflow-y-hidden" : "overflow-y-auto stage-scroll"}`}>
                {enteredSystem ? (
                  <ChapterRail
                    activeChapter={activeChapter}
                    setActiveChapter={navigateToChapter}
                    chapter={chapter}
                    mode="mobile"
                    onBackToLaunch={() => navigateToRoute("launch")}
                  />
                ) : null}
                {stage}
              </div>
            </div>
          </motion.section>
        </AnimatePresence>
      </main>

      {!isAssistantWorkbench ? (
        <div className="relative z-20 flex items-center justify-between gap-4 px-5 pb-6 md:px-8 xl:px-10">
          <div className="flex flex-wrap gap-2">
            {(enteredSystem
              ? ["Grounded QA", "Retrieval Record", "Traceable Evidence", "Intake Dock", "Hybrid Retrieval"]
              : ["Launch Deck", "Our Advantage", "A800 Support", "Traceable QA", "System Narrative"]
            ).map((item, index) => (
              <TacticalTag key={item} accent={index % 2 === 0 ? chapter.accent : chapter.secondary}>
                {item}
              </TacticalTag>
            ))}
          </div>
          <div className="flex items-center gap-3">
            {enteredSystem ? (
              <>
                <GlowTileButton
                  onClick={() => navigateToChapter(activeChapter === 0 ? chapters.length - 1 : activeChapter - 1)}
                  accent={chapter.accent}
                  className="flex h-12 w-12 items-center justify-center"
                >
                  <ChevronLeft size={18} />
                </GlowTileButton>
                <GlowTileButton
                  onClick={() => navigateToChapter(activeChapter === chapters.length - 1 ? 0 : activeChapter + 1)}
                  accent={chapter.secondary}
                  className="flex h-12 w-12 items-center justify-center"
                >
                  <ChevronRight size={18} />
                </GlowTileButton>
              </>
            ) : null}
          </div>
        </div>
      ) : null}

    </div>
  );
}

