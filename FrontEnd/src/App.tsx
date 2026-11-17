import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { MotionConfig } from "framer-motion";
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  CircleHelp,
  Compass,
  GitBranch,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import {
  type ChatSummary,
  type Evidence,
  type Health,
  type Library,
  errorMessage,
  loadLibrary,
  request,
} from "./api";
import { EvidencePanel } from "./components/EvidencePanel";
import { LibraryView } from "./components/LibraryView";
import { Workspace } from "./components/Workspace";
const GraphView = lazy(() =>
  import("./components/GraphView").then((module) => ({
    default: module.GraphView,
  })),
);

type View = "chat" | "library" | "search" | "graph" | "status";
const routes: Record<View, string> = {
  chat: "/aviation-assistant",
  library: "/upload",
  search: "/trace",
  status: "/system",
  graph: "/graph",
};
const labels: Record<View, string> = {
  chat: "智能问答",
  library: "知识库",
  search: "文档检索",
  status: "服务状态",
  graph: "知识图谱",
};

function initialView(): View {
  if (location.pathname === "/graph") return "graph";
  if (["/upload", "/intake"].includes(location.pathname)) return "library";
  if (location.pathname === "/trace") return "search";
  if (location.pathname === "/system") return "status";
  return "chat";
}

export default function App() {
  const [view, setView] = useState<View>(initialView);
  const [library, setLibrary] = useState<Library>({ bases: [], documents: [] });
  const [history, setHistory] = useState<ChatSummary[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState("");
  const [dataError, setDataError] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [historyRequest, setHistoryRequest] = useState<{ id: string } | null>(
    null,
  );
  const [resetKey, setResetKey] = useState(0);
  const [source, setSource] = useState<Evidence | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [historyEditor, setHistoryEditor] = useState<{
    item: ChatSummary;
    action: "rename" | "delete";
  } | null>(null);
  const [historyTitle, setHistoryTitle] = useState("");
  const [historyBusy, setHistoryBusy] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const historyDialog = useRef<HTMLDialogElement>(null);

  const refreshLibrary = useCallback(async () => {
    const data = await loadLibrary();
    setLibrary(data);
    setDataError("");
  }, []);
  const refreshHistory = useCallback(async () => {
    const data = await request<ChatSummary[]>("/chat/conversations");
    setHistory(data);
  }, []);
  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await request<Health>("/health"));
      setHealthError("");
    } catch (error) {
      setHealth(null);
      setHealthError(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    void refreshLibrary().catch((error) => setDataError(errorMessage(error)));
    void refreshHistory().catch((error) => setDataError(errorMessage(error)));
    void refreshHealth();
    const onPop = () => {
      setView(initialView());
      setSource(null);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [refreshLibrary, refreshHistory, refreshHealth]);

  function navigate(next: View) {
    setView(next);
    setMenuOpen(false);
    setSource(null);
    if (location.pathname !== routes[next])
      window.history.pushState({}, "", routes[next]);
  }

  function newConversation() {
    setConversationId(null);
    setHistoryRequest(null);
    setResetKey((value) => value + 1);
    navigate("chat");
  }

  function openHistory(item: ChatSummary) {
    setConversationId(item.id);
    setHistoryRequest({ id: item.id });
    navigate("chat");
  }

  function editHistory(item: ChatSummary, action: "rename" | "delete") {
    setHistoryEditor({ item, action });
    setHistoryTitle(item.title);
    setHistoryError("");
    historyDialog.current?.showModal();
  }

  async function saveHistory() {
    if (!historyEditor) return;
    const { item, action } = historyEditor;
    setHistoryBusy(true);
    setHistoryError("");
    try {
      await request(
        `/chat/conversations/${encodeURIComponent(item.id)}`,
        action === "rename"
          ? {
              method: "PATCH",
              body: JSON.stringify({ title: historyTitle.trim() }),
            }
          : { method: "DELETE" },
      );
      await refreshHistory();
      historyDialog.current?.close();
      setHistoryEditor(null);
      if (action === "delete" && conversationId === item.id) newConversation();
    } catch (error) {
      setHistoryError(errorMessage(error));
    } finally {
      setHistoryBusy(false);
    }
  }

  const online = health?.status === "ok";
  return (
    <MotionConfig reducedMotion="user">
      <div className={`app-shell ${source ? "has-evidence" : ""}`}>
        {menuOpen && (
          <button
            className="sidebar-backdrop"
            onClick={() => setMenuOpen(false)}
            aria-label="关闭导航"
          />
        )}
        <aside
          id="app-navigation"
          className={`sidebar ${menuOpen ? "mobile-open" : ""}`}
          onKeyDown={(event) => {
            if (event.key === "Escape") setMenuOpen(false);
          }}
        >
          <a
            className="brand"
            href="/"
            onClick={(event) => {
              event.preventDefault();
              newConversation();
            }}
          >
            <span className="brand-mark">
              <Compass size={25} strokeWidth={1.4} />
            </span>
            <span>
              <strong>
                ADMP<span className="brand-dot">.</span>
              </strong>
              <small>航空文档智能平台</small>
            </span>
          </a>
          <button className="new-chat" onClick={newConversation}>
            <Plus size={17} /> 开始新对话 <span>↗</span>
          </button>
          <div className="nav-caption">工作空间</div>
          <nav aria-label="主导航">
            {(
              [
                { id: "chat", icon: MessageSquare },
                { id: "search", icon: Search },
                { id: "graph", icon: GitBranch },
                { id: "library", icon: BookOpen },
              ] as const
            ).map((item) => (
              <button
                key={item.id}
                className={`nav-item ${view === item.id ? "active" : ""}`}
                onClick={() => navigate(item.id)}
                aria-current={view === item.id ? "page" : undefined}
              >
                <item.icon size={18} />
                <span>{labels[item.id]}</span>
                {view === item.id && <span className="active-dot" />}
              </button>
            ))}
          </nav>
          <div className="sidebar-history">
            <div className="nav-caption">
              最近会话 <span>{history.length}</span>
            </div>
            {history.length ? (
              <ul>
                {history.map((item) => (
                  <li
                    key={item.id}
                    className={
                      conversationId === item.id ? "current-history" : ""
                    }
                  >
                    <button
                      className="history-link"
                      onClick={() => openHistory(item)}
                      title={item.title}
                    >
                      <MessageSquare size={13} />
                      <span>{item.title || item.question}</span>
                    </button>
                    <div className="history-actions">
                      <button
                        aria-label={`重命名会话 ${item.title}`}
                        onClick={() => editHistory(item, "rename")}
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        aria-label={`删除会话 ${item.title}`}
                        onClick={() => editHistory(item, "delete")}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="sidebar-empty">你的探索将从这里延续</p>
            )}
          </div>
          <div className="sidebar-bottom">
            <div className="local-workspace">
              <span className="local-icon">
                <BookOpen size={18} />
              </span>
              <div>
                <strong>我的航空知识空间</strong>
                <small>
                  {library.bases.length} 个知识库 · {library.documents.length}{" "}
                  份文档
                </small>
              </div>
            </div>
            <button
              className="sidebar-status"
              onClick={() => {
                navigate("status");
                void refreshHealth();
              }}
            >
              <span className={`status-dot ${online ? "online" : ""}`} />
              <span>{online ? "服务已连接" : "检查服务连接"}</span>
              <Activity size={14} />
            </button>
          </div>
        </aside>
        <div className="main-shell">
          <header className="topbar">
            <div className="breadcrumbs">
              <button
                className="icon-button mobile-toggle"
                aria-controls="app-navigation"
                aria-expanded={menuOpen}
                onClick={() => setMenuOpen(true)}
                aria-label="打开导航"
              >
                <Menu size={20} />
              </button>
              <span>工作空间</span>
              <ChevronRight size={13} />
              <strong>{labels[view]}</strong>
            </div>
            <div className="topbar-right">
              <span className="private-badge">
                <span className="status-dot online" /> 本地工作空间
              </span>
              <button
                className="icon-button"
                aria-label="查看使用说明"
                onClick={() => navigate("status")}
              >
                <CircleHelp size={18} />
              </button>
              <span className="user-monogram">A</span>
            </div>
          </header>
          {dataError && (
            <div className="data-error" role="alert">
              暂时无法载入部分数据：{dataError}
              <button
                className="text-button"
                onClick={() => {
                  void refreshHealth();
                  void refreshLibrary().catch((error) =>
                    setDataError(errorMessage(error)),
                  );
                  void refreshHistory().catch((error) =>
                    setDataError(errorMessage(error)),
                  );
                }}
              >
                重试
              </button>
            </div>
          )}
          <main className="main-content">
            <div hidden={view !== "chat" && view !== "search"}>
              <Workspace
                mode={view === "search" ? "search" : "chat"}
                library={library}
                historyRequest={historyRequest}
                resetKey={resetKey}
                onConversation={(id) => {
                  setConversationId(id);
                  void refreshHistory().catch((error) =>
                    setDataError(errorMessage(error)),
                  );
                }}
                onLibrary={() => navigate("library")}
                onSource={setSource}
              />
            </div>
            {view === "graph" && (
              <Suspense
                fallback={
                  <p className="page-view muted" role="status">
                    正在载入知识图谱…
                  </p>
                }
              >
                <GraphView />
              </Suspense>
            )}
            {view === "library" && (
              <LibraryView
                library={library}
                onRefresh={refreshLibrary}
                onSource={setSource}
              />
            )}
            {view === "status" && (
              <section className="status-view page-view">
                <header className="page-heading">
                  <div>
                    <span className="eyebrow">WORKSPACE STATUS</span>
                    <h1>服务状态</h1>
                    <p>查看当前连接，以及文档问答的使用方式。</p>
                  </div>
                  <button
                    className="secondary-button"
                    onClick={() => void refreshHealth()}
                  >
                    <RefreshCw size={15} />
                    刷新状态
                  </button>
                </header>
                {healthError && (
                  <div className="error-banner">{healthError}</div>
                )}
                <div className="service-list">
                  {[
                    ["api", "问答接口", "文档管理与问答服务"],
                    ["elasticsearch", "关键词检索", "匹配法规条款与专业术语"],
                    ["milvus", "语义检索", "查找语义相关的文档片段"],
                  ].map(([key, title, description]) => (
                    <div className="service-row" key={key}>
                      <span className="service-icon">
                        {key === "api" ? (
                          <MessageSquare size={20} />
                        ) : key === "milvus" ? (
                          <GitBranch size={20} />
                        ) : (
                          <Search size={20} />
                        )}
                      </span>
                      <div>
                        <h3>{title}</h3>
                        <p>{description}</p>
                      </div>
                      <span
                        className={`status-pill ${health?.details[key] ? "ready" : "pending"}`}
                      >
                        <i />
                        {health?.details[key] ? "已连接" : "未连接"}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="library-note">
                  连接状态只说明服务可达。文档是否可检索，还需检查实际索引与检索结果。
                </p>
                <div className="how-to">
                  <h2>从一个好问题开始</h2>
                  <ol>
                    <li>
                      <span>01</span>
                      <div>
                        <h3>准备你的资料</h3>
                        <p>
                          在知识库中导入 PDF、Word、PowerPoint 或
                          Excel，等待解析与索引完成。
                        </p>
                      </div>
                    </li>
                    <li>
                      <span>02</span>
                      <div>
                        <h3>明确你的问题</h3>
                        <p>
                          加入法规编号、机型或专业术语。多跳检索可用于需要关联多段证据的问题。
                        </p>
                      </div>
                    </li>
                    <li>
                      <span>03</span>
                      <div>
                        <h3>回到原文核验</h3>
                        <p>
                          点击回答中的引用编号或来源卡片，查看文档原页与证据上下文。
                        </p>
                      </div>
                    </li>
                  </ol>
                  <a
                    href="https://github.com/Ultimate-XDU/ADMP"
                    target="_blank"
                    rel="noreferrer"
                    className="text-button"
                  >
                    项目主页 <ArrowUpRight size={14} />
                  </a>
                </div>
              </section>
            )}
          </main>
          <footer className="app-footer">
            <span>ADMP</span>
            <span>专业知识，清晰可见。</span>
            <span>AVIATION KNOWLEDGE WORKSPACE</span>
          </footer>
        </div>
        {source && (
          <EvidencePanel source={source} onClose={() => setSource(null)} />
        )}
        <dialog
          className="modal"
          ref={historyDialog}
          onCancel={() => setHistoryEditor(null)}
        >
          <header>
            <h2>
              {historyEditor?.action === "rename" ? "重命名会话" : "删除会话"}
            </h2>
            <button
              className="icon-button"
              onClick={() => {
                historyDialog.current?.close();
                setHistoryEditor(null);
              }}
              aria-label="关闭会话对话框"
            >
              <X size={18} />
            </button>
          </header>
          {historyError && <p className="error-banner">{historyError}</p>}
          {historyEditor?.action === "rename" ? (
            <label className="field">
              会话名称
              <input
                autoFocus
                maxLength={100}
                value={historyTitle}
                onChange={(event) => setHistoryTitle(event.target.value)}
              />
            </label>
          ) : (
            <p>
              将删除「{historyEditor?.item.title}」及其消息。此操作无法撤销。
            </p>
          )}
          <footer>
            <button
              className="secondary-button"
              onClick={() => {
                historyDialog.current?.close();
                setHistoryEditor(null);
              }}
            >
              取消
            </button>
            <button
              className={
                historyEditor?.action === "delete"
                  ? "danger-button"
                  : "primary-button"
              }
              onClick={() => void saveHistory()}
              disabled={
                historyBusy ||
                (historyEditor?.action === "rename" && !historyTitle.trim())
              }
            >
              {historyBusy ? (
                "处理中…"
              ) : historyEditor?.action === "rename" ? (
                <>
                  <Check size={15} />
                  保存
                </>
              ) : (
                "确认删除"
              )}
            </button>
          </footer>
        </dialog>
      </div>
    </MotionConfig>
  );
}
