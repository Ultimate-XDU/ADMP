import { useCallback, useEffect, useRef, useState } from "react";
import cytoscape, { type Core } from "cytoscape";
import {
  ArrowUpRight,
  Check,
  ChevronDown,
  Expand,
  FileText,
  GitBranch,
  Loader2,
  Network,
  RefreshCw,
  Search,
  Send,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { errorMessage, evidenceLocationLabel } from "../api";
import { AnswerText } from "./AnswerText";
import {
  graphRequest,
  type GraphAnswer,
  type GraphData,
  type GraphDocument,
  type GraphEdge,
  type GraphEvidence,
  type GraphNode,
  type GraphStatus,
  type GraphScope,
} from "../graph-api";
import "./graph.css";

const emptyGraph: GraphData = { nodes: [], edges: [], paths: [] };
const actions: Record<string, string> = {
  graph_search: "定位实体",
  graph_expand: "扩展关系",
  document_retrieve: "检索文档",
  inspect_evidence: "核对证据",
  finish: "完成检索",
  decide: "选择下一步",
};

const entityTypes: Record<string, string> = {
  source: "机构来源",
  record: "数据记录",
  document: "文档",
  section: "章节",
  article: "条款",
  paragraph: "段落",
  chunk: "证据片段",
  passage: "证据片段",
  table: "表格",
  figure: "图示",
  publisher: "发布机构",
  person: "人物",
  organization: "机构",
  location: "地点",
  event: "事件",
  concept: "概念",
  method: "方法",
  content: "资料",
  data: "数据",
  artifact: "产品与设备",
  naturalobject: "自然对象",
  condition: "条件",
  part: "部件",
};
const stopReasons: Record<string, string> = {
  answered: "回答已生成，可回查原文",
  tool_budget: "已达到本次检索步数上限",
  insufficient_evidence: "当前资料不足以支持可靠结论",
  partial_evidence: "已回答有依据的部分，其余信息不足",
  uncited_answer: "未获得有原文引用支持的回答",
  graph_unavailable: "知识图谱尚未就绪",
  error: "本次生成未完成",
};

function GraphCanvas({
  graph,
  scope,
  onSelect,
  onExpand,
}: {
  graph: GraphData;
  scope: GraphScope;
  onSelect: (item: GraphNode | GraphEdge) => void;
  onExpand: (id: string) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<Core | null>(null);
  const callbacks = useRef({ onSelect, onExpand });
  callbacks.current = { onSelect, onExpand };
  useEffect(() => {
    const cy = cytoscape({
      container: container.current,
      minZoom: 0.15,
      maxZoom: 3,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "background-color": "#76a99c",
            "border-color": "#fff",
            "border-width": 3,
            width: 22,
            height: 22,
            color: "#36544b",
            "font-size": 10,
            "text-valign": "bottom",
            "text-margin-y": 7,
            "text-wrap": "ellipsis",
            "text-max-width": "120px",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.2,
            "line-color": "#c8d6cf",
            "curve-style": "bezier",
            "font-size": 8,
            color: "#7d9186",
            label: "data(label)",
            "text-rotation": "autorotate",
            "text-background-color": "#f8faf7",
            "text-background-opacity": 0.9,
            "text-background-padding": "2px",
          },
        },
        {
          selector: 'node[kind="structure"]',
          style: {
            shape: "round-rectangle",
            width: 70,
            height: 25,
            "background-color": "#d6e1d5",
            "border-width": 1,
            "border-color": "#98af97",
            "font-size": 10,
            "text-max-width": "145px",
          },
        },
        {
          selector: 'node[type="document"]',
          style: {
            width: 110,
            height: 34,
            "background-color": "#607c72",
            color: "#334c42",
            "font-size": 12,
          },
        },
        {
          selector: 'node[type="chunk"],node[type="passage"]',
          style: {
            width: 16,
            height: 16,
            shape: "ellipse",
            "background-color": "#c5d8c4",
            "text-max-width": "100px",
          },
        },
        {
          selector: "edge.directed",
          style: {
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#b1c4b8",
          },
        },
        {
          selector: 'edge[edgeKind="contains"]',
          style: {
            "line-color": "#b8cbb8",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#b8cbb8",
            "curve-style": "taxi",
            "taxi-direction": "downward",
            label: "",
          },
        },
        {
          selector: 'edge[edgeKind="mentions"]',
          style: {
            "line-style": "dashed",
            "line-color": "#b5c7d1",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#b5c7d1",
          },
        },
        {
          selector: 'edge[edgeKind="references"]',
          style: {
            "line-color": "#ad9365",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#ad9365",
            width: 1.8,
          },
        },
        {
          selector: ".path",
          style: {
            "line-color": "#b68b49",
            "background-color": "#b68b49",
            "border-color": "#f2e5cf",
          },
        },
        { selector: "edge.path", style: { width: 3 } },
        { selector: "node.path", style: { "border-width": 3 } },
        {
          selector: ":selected",
          style: {
            "border-color": "#39796b",
            "border-width": 4,
            "line-color": "#39796b",
            "background-color": "#39796b",
          },
        },
        { selector: ".muted", style: { opacity: 0.15 } },
      ],
    });
    cy.on("tap", "node, edge", (event) =>
      callbacks.current.onSelect(event.target.data()),
    );
    cy.on("dbltap", "node", (event) =>
      callbacks.current.onExpand(event.target.id()),
    );
    cy.on("mouseover", "node", (event) => {
      const neighbours = event.target.closedNeighborhood();
      cy.elements().difference(neighbours).addClass("muted");
    });
    cy.on("mouseout", "node", () => cy.elements().removeClass("muted"));
    instance.current = cy;
    const resize = new ResizeObserver(() => cy.resize());
    resize.observe(container.current!);
    return () => {
      resize.disconnect();
      cy.destroy();
      instance.current = null;
    };
  }, []);
  useEffect(() => {
    const cy = instance.current!;
    const pathNodes = new Set(graph.paths.flatMap((path) => path.nodeIds));
    const pathEdges = new Set(graph.paths.flatMap((path) => path.edgeIds));
    cy.elements().remove();
    cy.add([
      ...graph.nodes.map((node) => ({
        data: node,
        classes: pathNodes.has(node.id) ? "path" : "",
      })),
      ...graph.edges.map((edge) => ({
        data: edge,
        classes: [
          pathEdges.has(edge.id) ? "path" : "",
          edge.directed ? "directed" : "",
        ]
          .filter(Boolean)
          .join(" "),
      })),
    ]);
    if (
      scope === "hierarchy" &&
      graph.nodes.some((node) => node.kind === "structure")
    ) {
      const hierarchy = cy.nodes().union(cy.edges('[edgeKind="contains"]'));
      const roots = cy
        .nodes()
        .filter(
          (node) => node.incomers('edge[edgeKind="contains"]').length === 0,
        );
      // Its default bounds use the previous zoom's world extent.
      hierarchy
        .layout({
          name: "breadthfirst",
          directed: true,
          roots: roots.map((node) => node.id()),
          boundingBox: {
            x1: 0,
            y1: 0,
            w: container.current!.clientWidth,
            h: container.current!.clientHeight,
          },
          circle: false,
          padding: 45,
          spacingFactor: 1.35,
          animate: false,
          fit: true,
        })
        .run();
    } else {
      cy.layout({
        name: "cose",
        animate: false,
        fit: true,
        padding: 50,
        nodeRepulsion: () => 12000,
        idealEdgeLength: () => 100,
        componentSpacing: 70,
      }).run();
    }
  }, [graph, scope]);
  return (
    <div className="kg-canvas-wrap">
      <div
        className="kg-canvas"
        ref={container}
        aria-label={`知识图谱：${graph.nodes.length}个节点，${graph.edges.length}条关系`}
      />
      <div className="kg-zoom">
        <button
          aria-label="放大图谱"
          onClick={() => {
            const cy = instance.current!;
            cy.zoom(cy.zoom() * 1.25);
          }}
        >
          <ZoomIn size={16} />
        </button>
        <button
          aria-label="缩小图谱"
          onClick={() => {
            const cy = instance.current!;
            cy.zoom(cy.zoom() / 1.25);
          }}
        >
          <ZoomOut size={16} />
        </button>
        <button
          aria-label="适应画布"
          onClick={() => instance.current!.fit(undefined, 45)}
        >
          <Expand size={16} />
        </button>
      </div>
      <span className="kg-canvas-hint">
        拖动平移 · 滚轮缩放 · 双击节点展开下一层
      </span>
    </div>
  );
}

export function GraphView() {
  const [status, setStatus] = useState<GraphStatus | null>(null);
  const [documents, setDocuments] = useState<GraphDocument[]>([]);
  const [docId, setDocId] = useState("");
  const [scope, setScope] = useState<GraphScope>("hierarchy");
  const [graph, setGraph] = useState<GraphData>(emptyGraph);
  const [query, setQuery] = useState("");
  const [entityQuery, setEntityQuery] = useState("");
  const [mode, setMode] = useState<"graph" | "agent">("agent");
  const [answer, setAnswer] = useState<GraphAnswer | null>(null);
  const [selected, setSelected] = useState<GraphNode | GraphEdge | null>(null);
  const [evidence, setEvidence] = useState<GraphEvidence | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [highlighted, setHighlighted] = useState<string[]>([]);
  const [evidenceBusy, setEvidenceBusy] = useState(false);

  const refresh = useCallback(async () => {
    const [nextStatus, nextDocs] = await Promise.all([
      graphRequest<GraphStatus>("status"),
      graphRequest<GraphDocument[]>("documents"),
    ]);
    setStatus(nextStatus);
    setDocuments(nextDocs);
    return nextStatus;
  }, []);
  const browse = useCallback(
    async (selectedDoc: string, selectedScope: GraphScope) => {
      setBusy("正在载入图谱");
      setError("");
      try {
        setGraph(
          await graphRequest<GraphData>("search", {
            query: "",
            scope: selectedScope,
            docIds: selectedDoc ? [selectedDoc] : [],
            limit: selectedScope === "hierarchy" ? 16 : 100,
          }),
        );
      } catch (err) {
        setError(errorMessage(err));
      } finally {
        setBusy("");
      }
    },
    [],
  );
  useEffect(() => {
    void refresh()
      .then((next) => {
        if (next.nodes > 0 || (next.structureNodes ?? 0) > 0)
          void browse("", "hierarchy");
      })
      .catch((err) => setError(errorMessage(err)));
  }, [refresh, browse]);
  useEffect(() => {
    if (!status?.building) return;
    const timer = window.setInterval(() => {
      void refresh()
        .then((next) => {
          if (!next.building && (next.nodes || next.structureNodes) && !answer)
            void browse(docId, scope);
        })
        .catch((err) => setError(errorMessage(err)));
    }, 4000);
    return () => window.clearInterval(timer);
  }, [status?.building, refresh, browse, docId, answer, scope]);

  async function searchEntities() {
    setBusy("正在查找实体");
    setError("");
    setAnswer(null);
    setSelected(null);
    setEvidence(null);
    setHighlighted([]);
    try {
      setGraph(
        await graphRequest<GraphData>("search", {
          query: entityQuery.trim(),
          scope,
          docIds: docId ? [docId] : [],
          limit: 100,
        }),
      );
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy("");
    }
  }
  async function ask() {
    if (!query.trim()) return;
    setBusy("正在沿关系检索与核对证据");
    setError("");
    setAnswer(null);
    setSelected(null);
    setEvidence(null);
    setHighlighted([]);
    try {
      const result = await graphRequest<GraphAnswer>("ask", {
        question: query.trim(),
        mode,
        docIds: docId ? [docId] : [],
      });
      setAnswer(result);
      setGraph(result.graph);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy("");
    }
  }
  async function expand(nodeId: string) {
    setBusy("正在扩展一跳关系");
    setError("");
    try {
      const expanded = await graphRequest<GraphData>("expand", {
        nodeIds: [nodeId],
        scope,
        hops: 1,
        docIds: docId ? [docId] : [],
      });
      setGraph((previous) => ({
        nodes: [
          ...new Map(
            [...previous.nodes, ...expanded.nodes].map((item) => [
              item.id,
              item,
            ]),
          ).values(),
        ],
        edges: [
          ...new Map(
            [...previous.edges, ...expanded.edges].map((item) => [
              item.id,
              item,
            ]),
          ).values(),
        ],
        paths: previous.paths,
        truncated: previous.truncated || expanded.truncated,
      }));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy("");
    }
  }
  async function openEvidence(chunkId: string) {
    setEvidenceBusy(true);
    setError("");
    try {
      setEvidence(
        await graphRequest<GraphEvidence>(
          `evidence?${new URLSearchParams({ chunkId })}`,
        ),
      );
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setEvidenceBusy(false);
    }
  }
  async function build() {
    const ids = docId ? [docId] : documents.map((document) => document.id);
    setBusy("正在提交图谱构建");
    setError("");
    try {
      setStatus(await graphRequest<GraphStatus>("build", { docIds: ids }));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy("");
    }
  }
  function changeDocument(value: string) {
    setDocId(value);
    setAnswer(null);
    setSelected(null);
    setEvidence(null);
    setHighlighted([]);
    void browse(value, scope);
  }
  const displayedGraph = highlighted.length
    ? {
        ...graph,
        paths: [
          {
            nodeIds: highlighted,
            edgeIds: graph.edges
              .filter(
                (edge) =>
                  highlighted.includes(edge.source) &&
                  highlighted.includes(edge.target),
              )
              .map((edge) => edge.id),
            hops: 0,
          },
        ],
      }
    : answer
      ? graph
      : { ...graph, paths: [] };
  const sourceIds = selected?.sourceIds ?? [];
  const lineage: GraphNode[] = [];
  if (selected && !("source" in selected) && selected.kind === "structure") {
    let node: GraphNode | undefined = selected;
    while (node) {
      lineage.unshift(node);
      node = graph.nodes.find((item) => item.id === node!.parentId);
    }
  }
  function changeScope(value: GraphScope) {
    setScope(value);
    setAnswer(null);
    setSelected(null);
    setEvidence(null);
    setHighlighted([]);
    void browse(docId, value);
  }
  return (
    <section className="kg-page page-view">
      <header className="page-heading kg-heading">
        <div>
          <span className="eyebrow">CONNECTED KNOWLEDGE</span>
          <h1>让知识，彼此连接。</h1>
          <p>从文档层级定位证据，沿跨文档关系发现下一条线索。</p>
        </div>
        <div className="kg-status">
          <span className={`status-dot ${status?.nodes ? "online" : ""}`} />
          {status?.building
            ? `语义抽取 ${status.completedDocuments}/${status.totalDocuments}`
            : status?.structuredDocuments !== undefined
              ? `${status.structuredDocuments} 份文档结构 · ${status.semanticDocuments ?? 0} 份语义抽取`
              : `${status?.nodes ?? 0} 个实体 · ${status?.edges ?? 0} 条关系`}
        </div>
      </header>
      <div className="kg-question">
        <Network size={21} />
        <textarea
          aria-label="图谱检索问题"
          placeholder="提出一个需要串联多份资料的问题…"
          value={query}
          rows={1}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              void ask();
            }
          }}
        />
        <button
          className="kg-mode"
          title="选择检索方式"
          onClick={() => setMode(mode === "agent" ? "graph" : "agent")}
        >
          {mode === "agent" ? "智能体检索" : "图谱检索"}
          <ChevronDown size={12} />
        </button>
        <button
          className="kg-send"
          disabled={
            !!busy ||
            !query.trim() ||
            !(status?.nodes || status?.structureNodes)
          }
          onClick={() => void ask()}
          aria-label="提交图谱问题"
        >
          {busy ? (
            <Loader2 className="kg-spin" size={18} />
          ) : (
            <Send size={18} />
          )}
        </button>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      {status?.error && (
        <div className="error-banner" role="alert">
          构建状态：{status.error}
        </div>
      )}
      <div className="kg-workbench">
        <div className="kg-map">
          <header className="kg-toolbar">
            <div>
              <GitBranch size={15} />
              <strong>
                {answer
                  ? "本次检索子图"
                  : scope === "hierarchy"
                    ? "文档层级"
                    : "知识关系"}
              </strong>
              <span>
                {graph.nodes.length} 节点 / {graph.edges.length} 关系
              </span>
            </div>
            <div>
              <select
                className="kg-scope-select"
                aria-label="图谱层次视图"
                value={scope}
                onChange={(event) =>
                  changeScope(event.target.value as GraphScope)
                }
                disabled={!!busy}
              >
                <option value="hierarchy">文档层级</option>
                <option value="semantic">语义关系</option>
                <option value="all">全部连接</option>
              </select>
              <label className="kg-entity-search">
                <Search size={13} />
                <input
                  aria-label="查找图谱节点"
                  placeholder="查找章节或实体…"
                  value={entityQuery}
                  onChange={(event) => setEntityQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void searchEntities();
                  }}
                />
              </label>
              <button
                className="kg-filter"
                aria-expanded={filterOpen}
                onClick={() => setFilterOpen(!filterOpen)}
              >
                <FileText size={14} />
                {docId ? "已筛选文档" : "全部文档"}
                <ChevronDown size={12} />
              </button>
              <button
                className="icon-button"
                aria-label="刷新图谱"
                disabled={!!busy}
                onClick={() => {
                  setAnswer(null);
                  setHighlighted([]);
                  void refresh().then(() => browse(docId, scope));
                }}
              >
                <RefreshCw size={15} />
              </button>
            </div>
          </header>
          {filterOpen && (
            <div className="kg-filter-panel">
              <label>
                检索范围
                <select
                  aria-label="图谱文档范围"
                  value={docId}
                  onChange={(event) => changeDocument(event.target.value)}
                >
                  <option value="">全部文档</option>
                  {documents.map((document) => (
                    <option key={document.id} value={document.id}>
                      {document.name}
                    </option>
                  ))}
                </select>
              </label>
              <p>文档层级随索引更新；语义关系抽取会调用本地模型。</p>
              <button
                className="secondary-button"
                disabled={!!busy || status?.building || !documents.length}
                onClick={() => void build()}
              >
                <RefreshCw size={13} />
                抽取所选文档语义关系
              </button>
            </div>
          )}
          <GraphCanvas
            graph={displayedGraph}
            scope={scope}
            onSelect={(item) => {
              setSelected(item);
              setEvidence(null);
            }}
            onExpand={(id) => void expand(id)}
          />
          {!graph.nodes.length && (
            <div className="kg-empty">
              <Network size={40} strokeWidth={1} />
              <h3>
                {busy
                  ? busy
                  : status?.building
                    ? "正在建立知识连接"
                    : scope === "hierarchy"
                      ? "从文档开始，定位知识"
                      : "从文档开始，建立关系"}
              </h3>
              <p>
                {status?.building
                  ? "实体与关系抽取需要一些时间，进度将自动更新。"
                  : answer
                    ? "这次问题没有找到可展示的关系，请查看回答与证据。"
                    : scope === "hierarchy"
                      ? "文档索引完成后，可按来源、章节和原文片段逐层浏览。"
                      : "选择文档并抽取语义关系后，可沿关系继续检索。"}
              </p>
              {!busy && !status?.building && !answer && (
                <button
                  className="secondary-button"
                  disabled={!documents.length}
                  onClick={() => setFilterOpen(true)}
                >
                  选择文档 <ArrowUpRight size={14} />
                </button>
              )}
            </div>
          )}
          <footer className="kg-map-footer">
            <span>
              <i className="kg-structure-dot" /> 文档层级{" "}
              <i className="kg-dot" /> 语义实体 <i className="kg-dot kg-gold" />
              关系路径
            </span>
            <span>
              {graph.truncated
                ? "已显示局部图谱，可搜索或扩展"
                : scope === "hierarchy"
                  ? "层次来自原文结构，可双击节点逐层展开"
                  : "关系来自文档抽取，请结合原文判断"}
            </span>
          </footer>
          {busy && (
            <div className="kg-working" role="status">
              <Loader2 className="kg-spin" size={13} />
              {busy}
            </div>
          )}
        </div>
        <aside className="kg-inspector" aria-label="图谱详情">
          <header>
            <span className="eyebrow">EVIDENCE & CONNECTIONS</span>
            {(selected || evidence) && (
              <button
                className="icon-button"
                aria-label="清除图谱选择"
                onClick={() => {
                  setSelected(null);
                  setEvidence(null);
                }}
              >
                <X size={14} />
              </button>
            )}
          </header>
          {selected ? (
            <>
              <span className="kg-tag">
                {"source" in selected
                  ? {
                      contains: "层级包含",
                      mentions: "原文提及",
                      references: "显式引用",
                      semantic: "抽取的语义关系",
                    }[selected.edgeKind || "semantic"]
                  : entityTypes[selected.type.toLowerCase()] || "实体"}
              </span>
              <h3>{selected.label}</h3>
              {!("source" in selected) && selected.locator?.label && (
                <p className="kg-location">{selected.locator.label}</p>
              )}
              {lineage.length > 1 && (
                <nav className="kg-breadcrumb" aria-label="节点所在层级">
                  {lineage.map((node) => (
                    <button
                      key={node.id}
                      onClick={() => {
                        setSelected(node);
                        setEvidence(null);
                      }}
                    >
                      {node.label}
                    </button>
                  ))}
                </nav>
              )}
              {"source" in selected ? (
                <p className="kg-connection">
                  {
                    graph.nodes.find((node) => node.id === selected.source)
                      ?.label
                  }{" "}
                  <span>{selected.directed ? "→" : "—"}</span>{" "}
                  {
                    graph.nodes.find((node) => node.id === selected.target)
                      ?.label
                  }
                </p>
              ) : (
                <button
                  className="text-button kg-expand-button"
                  disabled={!!busy}
                  onClick={() => void expand(selected.id)}
                >
                  {selected.kind === "structure"
                    ? "展开下一级"
                    : "扩展相邻关系"}{" "}
                  <ArrowUpRight size={14} />
                </button>
              )}
              {"source" in selected && selected.provenance?.length ? (
                <div className="kg-provenance">
                  {selected.provenance.slice(0, 3).map((item, index) => (
                    <div key={`${item.docId}-${item.sourceId || index}`}>
                      <small>
                        {(
                          {
                            structure: "原文结构",
                            explicit_metadata: "文档引用元数据",
                            explicit_url: "原文链接",
                            explicit_identifier: "原文文号引用",
                            quoted_title: "原文引文",
                            extracted_sources: "原文提及",
                            lightrag: "模型抽取",
                          } as Record<string, string>
                        )[item.method] || item.method}
                      </small>
                      {item.locator?.label && <p>{item.locator.label}</p>}
                      {item.quote && <blockquote>{item.quote}</blockquote>}
                      {item.sourceUrl && (
                        <a
                          href={item.sourceUrl}
                          target="_blank"
                          rel="noreferrer"
                        >
                          查看官方来源 <ArrowUpRight size={11} />
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              ) : null}
              <h4>
                关联原文 <span>{sourceIds.length}</span>
              </h4>
              <div className="kg-source-list">
                {sourceIds.map((id, index) => (
                  <button
                    key={id}
                    onClick={() => void openEvidence(id)}
                    className={evidence?.chunk_id === id ? "active" : ""}
                  >
                    <FileText size={14} />
                    <span>
                      来源片段 {index + 1}
                      <small>{id}</small>
                    </span>
                    <ArrowUpRight size={13} />
                  </button>
                ))}
              </div>
            </>
          ) : (
            !evidence && (
              <div className="kg-inspector-intro">
                <Search size={26} strokeWidth={1.2} />
                <h3>每条连接，都有出处。</h3>
                <p>
                  点击实体或关系，查看它在文档中的依据。双击节点，展开文档层级或相邻知识。
                </p>
                <div>
                  <span>01</span> 定位问题中的关键实体
                </div>
                <div>
                  <span>02</span> 沿关系发现桥接线索
                </div>
                <div>
                  <span>03</span> 串联原文，核对答案
                </div>
                {graph.nodes.length > 0 && (
                  <div className="kg-entity-index">
                    <h4>节点索引</h4>
                    {graph.nodes.slice(0, 12).map((node) => (
                      <button
                        key={node.id}
                        onClick={() => {
                          setSelected(node);
                          setEvidence(null);
                        }}
                      >
                        {node.label}
                        <ArrowUpRight size={11} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          )}
          {evidenceBusy && (
            <p className="kg-evidence-loading" role="status">
              正在读取来源…
            </p>
          )}
          {evidence && !evidenceBusy && (
            <article className="kg-evidence">
              <div>
                <FileText size={15} />
                <strong>{evidence.document_title || evidence.doc_name}</strong>
              </div>
              <small>
                {`${evidenceLocationLabel(evidence)} · `}
                {evidence.section_title}
              </small>
              <p>{evidence.content}</p>
              {evidence.source_url && (
                <a href={evidence.source_url} target="_blank" rel="noreferrer">
                  访问官方来源 <ArrowUpRight size={12} />
                </a>
              )}
            </article>
          )}
        </aside>
      </div>
      {answer && (
        <div className="kg-results">
          <section className="kg-answer">
            <span className="eyebrow">GROUNDED ANSWER</span>
            <h2>沿证据得到的回答</h2>
            <AnswerText
              text={answer.answer}
              sources={answer.sources}
              citations={answer.citations}
              onSelect={(source) => {
                setEvidence(source);
                setSelected(null);
              }}
            />
            <div className="kg-answer-sources">
              {answer.sources.map((source, index) => (
                <button
                  key={source.chunk_id}
                  onClick={() => void openEvidence(source.chunk_id)}
                >
                  <span>
                    {answer.citations.find(
                      (citation) => citation.sourceIndex === index,
                    )?.id ?? index + 1}
                  </span>
                  {source.document_title || source.doc_name}
                  <ArrowUpRight size={12} />
                </button>
              ))}
            </div>
          </section>
          <section className="kg-trace">
            <span className="eyebrow">RETRIEVAL JOURNEY</span>
            <h2>实际检索过程</h2>
            <ol>
              {answer.trace.map((step, index) => (
                <li key={`${step.step}-${index}`}>
                  <button
                    onClick={() => setHighlighted(step.nodeIds)}
                    disabled={!step.nodeIds.length}
                  >
                    <span className="kg-step-number">{index + 1}</span>
                    <div>
                      <strong>{actions[step.action] || step.action}</strong>
                      <p>
                        {step.summary.split("抽取到的关系路径")[0].trim() ||
                          step.query}
                      </p>
                      <small>{step.sourceIds.length} 个原文片段</small>
                    </div>
                  </button>
                </li>
              ))}
            </ol>
            <p className="kg-stop">
              <Check size={13} />
              {stopReasons[answer.stopReason] || answer.stopReason}
            </p>
          </section>
        </div>
      )}
    </section>
  );
}
