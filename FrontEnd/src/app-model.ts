export const supportedDocumentFormats = [
  "PDF",
  "DOCX",
  "PPTX",
  "XLSX",
  "HTML",
  "MD",
  "TXT",
  "CSV",
  "JSON",
  "XML",
] as const;
export const documentAccept = supportedDocumentFormats
  .map((type) => `.${type.toLowerCase()}`)
  .concat(".htm")
  .join(",");

export type KnowledgeBaseRecord = {
  id: number;
  name: string;
  description: string;
  creator: string;
  createdAt: string;
};

export type DocumentStatus =
  "READY" | "INDEXING" | "CHUNKING" | "SYNCED" | "FAILED";

export type DocumentRecord = {
  id: number;
  fileName: string;
  knowledgeBaseId: number;
  type: (typeof supportedDocumentFormats)[number];
  title?: string;
  sourceUrl?: string;
  publisher?: string;
  size: string;
  chunkCount: number;
  status: DocumentStatus;
  uploadedAt: string;
  summary: string;
};
