import { useEffect, useRef, useState, type FormEvent } from "react";
import { motion } from "framer-motion";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  Compass,
  Copy,
  FileText,
  GitBranch,
  LoaderCircle,
  ScanEye,
  Search,
  Sparkles,
  Square,
} from "lucide-react";
import {
  type ChatDetail,
  type ChatMessage,
  type Evidence,
  type Library,
  type QAResult,
  type RetrievalTrace,
  errorMessage,
  evidenceLocationLabel,
  request,
} from "../api";
import { AnswerText } from "./AnswerText";
import { SourceCards } from "./EvidencePanel";
import { RetrievalSteps } from "./RetrievalSteps";

const abstentionLabels: Record<string, string> = {
  no_evidence: "没有检索到可用于回答的文档证据。",
  search_error: "检索服务未能完成请求，请稍后重试。",
  llm_unavailable: "模型暂时不可用，以下提供已检索到的资料。",
};

type Props = {
  mode: "chat" | "search";
  library: Library;
  historyRequest: { id: string } | null;
  resetKey: number;
  onConversation: (id: string) => void;
  onLibrary: () => void;
  onSource: (source: Evidence) => void;
};

function FlightLines() {
  return (
    <svg
      className="flight-lines"
      viewBox="0 0 440 230"
      fill="none"
      aria-hidden="true"
    >
      <g stroke="currentColor" strokeWidth="0.7">
        <ellipse
          cx="273"
          cy="126"
          rx="143"
          ry="77"
          transform="rotate(-24 273 126)"
        />
        <ellipse
          cx="273"
          cy="126"
          rx="105"
          ry="56"
          transform="rotate(-24 273 126)"
        />
        <ellipse
          cx="273"
          cy="126"
          rx="65"
          ry="35"
          transform="rotate(-24 273 126)"
        />
        <path
          d="M70 189 402 44M201 20 343 216M95 82 429 163"
          strokeDasharray="3 7"
        />
        <path d="M69 180C155 218 178 44 336 72" strokeWidth="1.2" />
      </g>
      <circle
        cx="204"
        cy="116"
        r="5"
        fill="var(--accent)"
        stroke="var(--paper)"
        strokeWidth="3"
      />
      <circle cx="336" cy="72" r="3" fill="currentColor" />
      <path
        d="m267 123 29-23-12 34-6-12-11 1Z"
        fill="var(--accent)"
        opacity=".85"
      />
      <g fill="currentColor" fontSize="9" fontFamily="monospace">
        <text x="82" y="204">
          KNOWLEDGE IN SIGHT
        </text>
        <text x="344" y="66">
          REF. 01
        </text>
        <text x="212" y="107">
          ADMP
        </text>
      </g>
    </svg>
  );
}

export function Workspace({
  mode,
  library,
  historyRequest,
  resetKey,
  onConversation,
  onLibrary,
  onSource,
}: Props) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [results, setResults] = useState<Evidence[] | null>(null);
  const [searchTrace, setSearchTrace] = useState<RetrievalTrace | null>(null);
  const [searchQuestion, setSearchQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState("");
  const [retrievalMode, setRetrievalMode] = useState("auto");
  const [includeVisualContext, setIncludeVisualContext] = useState(false);
  const [copied, setCopied] = useState("");
  const [seconds, setSeconds] = useState(0);
  const activeRequest = useRef<AbortController | null>(null);
  const currentConversation = useRef<string | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    activeRequest.current?.abort();
    setMessages([]);
    setResults(null);
    setSearchTrace(null);
    setSearchQuestion("");
    setError("");
    setQuestion("");
    setBusy(false);
    setLoadingHistory(false);
    currentConversation.current = null;
  }, [resetKey]);

  useEffect(() => {
    if (!historyRequest) return;
    const controller = new AbortController();
    activeRequest.current?.abort();
    setBusy(false);
    setLoadingHistory(true);
    setError("");
    request<ChatDetail>(
      `/chat/conversations/${encodeURIComponent(historyRequest.id)}`,
      { signal: controller.signal },
    )
      .then((detail) => {
        setMessages(detail.messages);
        currentConversation.current = detail.id;
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(errorMessage(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingHistory(false);
      });
    return () => controller.abort();
  }, [historyRequest]);

  useEffect(() => {
    if (!busy) return;
    setSeconds(0);
    const timer = window.setInterval(
      () => setSeconds((value) => value + 1),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [busy]);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages, busy]);
  useEffect(() => () => activeRequest.current?.abort(), []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const query = question.trim();
    if (!query || busy || loadingHistory) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setBusy(true);
    setError("");
    setQuestion("");
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    if (mode === "chat") setMessages((previous) => [...previous, userMessage]);
    else {
      setSearchQuestion(query);
      setResults(null);
      setSearchTrace(null);
    }
    try {
      const body = JSON.stringify({
        question: query,
        retrievalMode,
        includeVisualContext: mode === "chat" && includeVisualContext,
        ...(currentConversation.current && mode === "chat"
          ? { conversationId: currentConversation.current }
          : {}),
      });
      if (mode === "search") {
        const result = await request<{
          results: Evidence[];
          retrievalTrace?: RetrievalTrace;
        }>("/search", {
          method: "POST",
          body,
          signal: controller.signal,
        });
        setResults(result.results);
        setSearchTrace(result.retrievalTrace || null);
      } else {
        const result = await request<QAResult>("/qa", {
          method: "POST",
          body,
          signal: controller.signal,
        });
        currentConversation.current = result.conversationId;
        setMessages((previous) => [
          ...previous,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: result.answer,
            payload: result,
          },
        ]);
        onConversation(result.conversationId);
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(errorMessage(error));
        setQuestion(query);
      }
    } finally {
      if (activeRequest.current === controller) {
        setBusy(false);
        activeRequest.current = null;
      }
    }
  }

  function stop() {
    activeRequest.current?.abort();
    setBusy(false);
    setError(
      "已停止等待。服务端可能仍在完成本次请求，可稍后从历史会话查看结果。",
    );
  }

  async function copyAnswer(message: ChatMessage) {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(message.id);
    } catch {
      setError("未能复制，请选中答案文字后复制。");
    }
  }

  const isEmpty = mode === "chat" ? messages.length === 0 : !searchQuestion;
  const indexed = library.documents.filter(
    (doc) =>
      doc.chunkCount > 0 && (doc.status === "READY" || doc.status === "SYNCED"),
  ).length;
  const prompts =
    mode === "search"
      ? [
          "航空发动机适航要求",
          "结冰对机翼升力的影响",
          "飞行中风切变的识别与处置",
        ]
      : [
          "概述航空发动机适航规定的主要要求",
          "结合不同文档，解释结冰如何影响飞机性能",
          "承运人的运行合格审定有哪些要求？",
        ];

  return (
    <div
      className={`workspace ${isEmpty ? "workspace-welcome" : "workspace-active"}`}
    >
      {isEmpty && (
        <motion.section
          className="welcome"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          onPointerMove={(event) => {
            const bounds = event.currentTarget.getBoundingClientRect();
            event.currentTarget.style.setProperty(
              "--flight-x",
              `${(event.clientX - bounds.left - bounds.width / 2) / 60}px`,
            );
            event.currentTarget.style.setProperty(
              "--flight-y",
              `${(event.clientY - bounds.top - bounds.height / 2) / 40}px`,
            );
          }}
          onPointerLeave={(event) => {
            event.currentTarget.style.setProperty("--flight-x", "0px");
            event.currentTarget.style.setProperty("--flight-y", "0px");
          }}
        >
          <div className="welcome-heading">
            <span className="eyebrow">
              <span className="tiny-line" /> AVIATION DOCUMENT INTELLIGENCE
            </span>
            <h1>
              {mode === "search" ? (
                <>
                  发现线索，
                  <br />
                  <span>回到知识的源头。</span>
                </>
              ) : (
                <>
                  每一次探索，
                  <br />
                  <span>都有据可循。</span>
                </>
              )}
            </h1>
            <p>
              {mode === "search"
                ? "跨越关键词与语义，直接查找航空资料中的原始证据。"
                : "连接航空规章、技术手册与研究资料，让复杂问题清晰可解。"}
            </p>
          </div>
          <FlightLines />
        </motion.section>
      )}

      {!isEmpty && (
        <div className="conversation-body">
          {mode === "chat" ? (
            messages.map((message) =>
              message.role === "user" ? (
                <div className="user-turn" key={message.id}>
                  <span className="avatar user-avatar">你</span>
                  <h2>{message.content}</h2>
                </div>
              ) : (
                <article className="assistant-turn" key={message.id}>
                  <div className="answer-byline">
                    <span className="avatar assistant-avatar">
                      <Compass size={19} />
                    </span>
                    <strong>ADMP</strong>
                    <span>文档问答</span>
                  </div>
                  {message.payload?.abstainReason && (
                    <div className="notice">
                      {abstentionLabels[message.payload.abstainReason] ||
                        "当前资料不足以确认完整答案，请结合原文核对。"}
                    </div>
                  )}
                  <AnswerText
                    text={message.content}
                    sources={message.payload?.sources || []}
                    citations={message.payload?.citations || []}
                    onSelect={onSource}
                  />
                  {!!message.payload?.sources.length && (
                    <section className="answer-sources">
                      <div className="section-label">
                        <BookOpen size={14} /> 参考来源{" "}
                        <span>{message.payload.sources.length}</span>
                      </div>
                      <SourceCards
                        sources={message.payload.sources}
                        citations={message.payload.citations || []}
                        onSelect={onSource}
                      />
                    </section>
                  )}
                  <RetrievalSteps trace={message.payload?.retrievalTrace} />
                  {message.payload?.visualContext?.used && (
                    <div className="visual-note">
                      <ScanEye size={14} />
                      <span>
                        已读取 {message.payload.visualContext.pages.length}{" "}
                        页原图
                      </span>
                      {message.payload.visualContext.pages.map(
                        (page, index) => (
                          <button
                            key={index}
                            onClick={() =>
                              onSource(
                                message.payload!.sources[page.sourceIndex],
                              )
                            }
                          >
                            第 {page.page} 页
                          </button>
                        ),
                      )}
                    </div>
                  )}
                  {!message.payload?.visualContext?.used &&
                    message.payload?.visualContext?.reason && (
                      <p className="visual-fallback">
                        {message.payload.visualContext.reason === "no_pdf_page"
                          ? "召回的来源没有可用的 PDF 原页，本次按文本作答。"
                          : "原页图像暂不可用，本次按文本作答。"}
                      </p>
                    )}
                  <button
                    className="text-button answer-copy"
                    onClick={() => void copyAnswer(message)}
                  >
                    {copied === message.id ? (
                      <Check size={14} />
                    ) : (
                      <Copy size={14} />
                    )}
                    {copied === message.id ? "已复制" : "复制回答"}
                  </button>
                </article>
              ),
            )
          ) : (
            <section className="search-results">
              <div className="eyebrow">RETRIEVAL RESULTS</div>
              <h2>{searchQuestion}</h2>
              {results && (
                <p className="muted">
                  找到 {results.length} 条相关片段，点击可核对原文。
                </p>
              )}
              <RetrievalSteps trace={searchTrace} />
              {results?.map((source, index) => (
                <button
                  className="search-result"
                  key={`${source.chunk_id}-${index}`}
                  onClick={() => onSource(source)}
                >
                  <div className="result-top">
                    <span className="source-number">{index + 1}</span>
                    <strong>{source.document_title || source.doc_name}</strong>
                    <span className="type-pill">
                      {source.source_type === "image"
                        ? "图像"
                        : source.source_type === "table"
                          ? "表格"
                          : "文本"}
                    </span>
                    <ChevronRight size={16} />
                  </div>
                  <p className="result-location">
                    {evidenceLocationLabel(source)} ·{" "}
                    {source.article_label ||
                      source.section_title ||
                      source.heading}
                  </p>
                  <p className="result-text">{source.content}</p>
                </button>
              ))}
              {results?.length === 0 && (
                <div className="empty-state">
                  <Search size={28} />
                  <h3>暂未找到匹配的原文</h3>
                  <p>试试具体的法规编号、专业术语，或先导入相关文档。</p>
                </div>
              )}
            </section>
          )}
          {busy && (
            <div className="thinking" role="status">
              <LoaderCircle size={18} className="spin" />
              <div>
                <strong>
                  {mode === "search"
                    ? "正在检索文档"
                    : "正在检索证据并组织回答"}
                </strong>
                <span>
                  已等待 {seconds} 秒 · 本地模型处理长文档可能需要更多时间
                </span>
              </div>
            </div>
          )}
          <div ref={end} />
        </div>
      )}
      {loadingHistory && (
        <div className="thinking" role="status">
          <LoaderCircle className="spin" size={18} />
          正在读取历史会话…
        </div>
      )}
      <div className="composer-area">
        {error && (
          <div className="error-banner" role="alert">
            {error}
            <button onClick={() => setError("")} aria-label="关闭提示">
              ×
            </button>
          </div>
        )}
        <form className="composer" onSubmit={(event) => void submit(event)}>
          <label className="sr-only" htmlFor={`question-${mode}`}>
            {mode === "search" ? "检索关键词" : "你的航空文档问题"}
          </label>
          <textarea
            id={`question-${mode}`}
            ref={input}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={
              mode === "search"
                ? "输入关键词、条款编号或描述你要找的内容…"
                : "提出你的问题，从航空知识中寻找答案…"
            }
            rows={isEmpty ? 3 : 2}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <div className="composer-bottom">
            <span className="scope">
              <BookOpen size={15} /> 全部资料
            </span>
            <span className="composer-divider" />
            <label className="mode-select">
              <GitBranch size={14} />
              <select
                aria-label="检索策略"
                value={retrievalMode}
                onChange={(event) => setRetrievalMode(event.target.value)}
                disabled={busy}
              >
                <option value="auto">自动检索</option>
                <option value="single">单跳检索</option>
                <option value="multi">多跳检索</option>
              </select>
            </label>
            {mode === "chat" && (
              <label
                className={`visual-toggle ${includeVisualContext ? "selected" : ""}`}
                title="让模型同时读取召回的 PDF 原页，适合图表问题，处理时间会更长。"
              >
                <input
                  type="checkbox"
                  aria-label="图文增强"
                  checked={includeVisualContext}
                  disabled={busy}
                  onChange={(event) =>
                    setIncludeVisualContext(event.target.checked)
                  }
                />
                <ScanEye size={14} />
                图文增强
              </label>
            )}
            <span className="composer-spacer" />
            <span className="keyboard-hint">Shift + Enter 换行</span>
            {busy ? (
              <button
                className="send-button"
                type="button"
                onClick={stop}
                aria-label="停止等待"
              >
                <Square size={15} />
              </button>
            ) : (
              <button
                className="send-button"
                type="submit"
                disabled={!question.trim() || loadingHistory}
                aria-label={mode === "search" ? "开始检索" : "发送问题"}
              >
                <ArrowUp size={19} />
              </button>
            )}
          </div>
        </form>
        {isEmpty ? (
          <div className="suggestions">
            {prompts.map((prompt, index) => (
              <button
                key={prompt}
                onClick={() => {
                  setQuestion(prompt);
                  input.current?.focus();
                }}
              >
                <span>
                  {
                    [
                      <FileText size={14} />,
                      <GitBranch size={14} />,
                      <Search size={14} />,
                    ][index]
                  }
                </span>
                {prompt}
                <ArrowRight size={13} />
              </button>
            ))}
          </div>
        ) : (
          <p className="composer-footnote">
            {mode === "chat"
              ? "答案基于文档证据生成，请结合原文核对。"
              : "检索结果为原文片段，点击可查看对应页与上下文。"}
          </p>
        )}
      </div>
      {isEmpty && (
        <section className="workspace-overview">
          <div className="overview-heading">
            <div>
              <span className="eyebrow">YOUR KNOWLEDGE, CONNECTED</span>
              <h2>从资料，到洞见</h2>
            </div>
            <button className="text-button" onClick={onLibrary}>
              管理知识库 <ArrowRight size={15} />
            </button>
          </div>
          <div className="overview-content">
            <div className="recent-documents">
              <div className="section-label">最近文档</div>
              {library.documents.length ? (
                library.documents.slice(0, 3).map((doc) => (
                  <button
                    key={doc.id}
                    onClick={onLibrary}
                    className="recent-document"
                  >
                    <span className="file-icon">
                      <FileText size={18} />
                    </span>
                    <span>
                      <strong>{doc.fileName}</strong>
                      <small>
                        {doc.type} · {doc.uploadedAt.split(" ")[0]}
                      </small>
                    </span>
                    <ChevronRight size={15} />
                  </button>
                ))
              ) : (
                <button className="empty-library" onClick={onLibrary}>
                  <BookOpen size={23} />
                  <span>导入第一份文档，开始构建航空知识库</span>
                  <ArrowRight size={16} />
                </button>
              )}
            </div>
            <div className="collection-summary">
              <div className="summary-symbol">
                <Sparkles size={19} />
              </div>
              <h3>
                有来源的知识，
                <br />
                有依据的回答。
              </h3>
              <p>让分散的专业资料成为随时可查、可核验的工作参考。</p>
              <div className="summary-stats">
                <div>
                  <strong>{library.bases.length}</strong>
                  <span>知识库</span>
                </div>
                <div>
                  <strong>{indexed}</strong>
                  <span>就绪文档</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
      {isEmpty && (
        <div className="welcome-footer">
          <span>ADMP · 航空文档智能管理平台</span>
          <ArrowDown size={13} />
          <span>让知识落地</span>
        </div>
      )}
    </div>
  );
}
