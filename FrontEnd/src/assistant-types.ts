export type EvidenceLocation = {
  page: number;
  chapter: string;
  textAnchor: string;
  bbox?: { x: number; y: number; width: number; height: number };
  bboxPages?: Array<{
    page_number: number;
    bbox: { x: number; y: number; width: number; height: number };
    highlight_text?: string;
    matched_line_count?: number;
    semantic_line_count?: number;
  }>;
};

export type AssistantDocumentType = "AMM" | "FIM" | "TSM" | "IPC" | "SRM";

export type AssistantCitation = {
  id: string;
  label: string;
  sourceId: string;
  documentTitle: string;
  documentType: AssistantDocumentType;
  chapter: string;
  page: number;
  quotedText: string;
  highlightText: string;
  snippet: string;
  evidenceLocation: EvidenceLocation;
};

export type AssistantSourcePage = {
  pageNumber: number;
  sectionTitle: string;
  lines: string[];
  previewImageUrl?: string;
};

export type AssistantSource = {
  id: string;
  documentTitle: string;
  documentType: AssistantDocumentType;
  aircraftModel: string;
  chapter: string;
  pageCount: number;
  snippet: string;
  pages: AssistantSourcePage[];
  previewMode?: "image" | "text";
  previewImageTemplate?: string;
};

export type AssistantMessageSegment = {
  id: string;
  kind: "step" | "paragraph" | "note";
  text: string;
  citationIds: string[];
};

export type AssistantMessage = {
  id: string;
  role: "user" | "assistant";
  content?: string;
  segments?: AssistantMessageSegment[];
};

export type AssistantConversation = {
  id: string;
  quickPrompt: string;
  userQuestion: string;
  aircraftModel: string;
  documentTypeFocus: AssistantDocumentType | "ALL";
  messages: AssistantMessage[];
  citations: AssistantCitation[];
  sources: AssistantSource[];
  defaultCitationId: string;
};
