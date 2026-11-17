import { request, type Citation, type Evidence } from "./api";

export type GraphScope = "hierarchy" | "semantic" | "all";

export type GraphNode = {
  id: string;
  label: string;
  type: string;
  kind?: "structure" | "semantic";
  level?: number;
  parentId?: string;
  docId?: string;
  locator?: { label?: string; [key: string]: unknown };
  sourceIds: string[];
};
export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  edgeKind?: "contains" | "mentions" | "references" | "semantic";
  relationType?: string;
  directed?: boolean;
  provenance?: {
    docId: string;
    method: string;
    sourceId?: string;
    sourceUrl?: string;
    locator?: { label?: string };
    quote?: string;
  }[];
  sourceIds: string[];
};
export type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  paths: {
    nodeIds: string[];
    edgeIds: string[];
    hops: number;
    semanticHops?: number;
    hierarchyDepth?: number;
  }[];
  truncated?: boolean;
};
export type GraphDocument = {
  id: string;
  name: string;
  sourceUrl: string;
  chunks: number;
  indexed: boolean;
  format?: string;
  publisher?: string;
};
export type GraphStatus = {
  status: string;
  documents: number;
  nodes: number;
  edges: number;
  chunks: number;
  building: boolean;
  structureNodes?: number;
  semanticNodes?: number;
  structuredDocuments?: number;
  semanticDocuments?: number;
  completedDocuments: number;
  totalDocuments: number;
  error?: string;
};
export type GraphAnswer = {
  answer: string;
  sources: Evidence[];
  citations: Citation[];
  graph: GraphData;
  trace: {
    step: number;
    action: string;
    query: string;
    summary: string;
    nodeIds: string[];
    sourceIds: string[];
  }[];
  stopReason: string;
};
export type GraphEvidence = Evidence & { source_url?: string };

export function graphRequest<T>(path: string, body?: unknown) {
  return request<T>(
    `/graph/${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          body: JSON.stringify(body),
        },
  );
}
