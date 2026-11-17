import { useRef, useState, type FormEvent } from "react";
import {
  BookOpen,
  CheckCircle2,
  FileText,
  Folder,
  LoaderCircle,
  Pencil,
  Plus,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  type Document,
  type Evidence,
  type Library,
  errorMessage,
  importDocument,
  request,
} from "../api";
import {
  documentAccept,
  supportedDocumentFormats,
  type KnowledgeBaseRecord,
} from "../app-model";

type Props = {
  library: Library;
  onRefresh: () => Promise<void>;
  onSource: (source: Evidence) => void;
};
type EditingBase = { id?: number; name: string; description: string };
type Removal = { kind: "document" | "base"; id: number; name: string };

export function LibraryView({ library, onRefresh, onSource }: Props) {
  const [filter, setFilter] = useState("");
  const [baseId, setBaseId] = useState<number | "all">("all");
  const [editor, setEditor] = useState<EditingBase | null>(null);
  const [removal, setRemoval] = useState<Removal | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadBase, setUploadBase] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const modal = useRef<HTMLDialogElement>(null);
  const visibleDocs = library.documents.filter(
    (doc) =>
      (baseId === "all" || doc.knowledgeBaseId === baseId) &&
      `${doc.title || ""} ${doc.fileName}`
        .toLowerCase()
        .includes(filter.toLowerCase()),
  );
  const selectedBase = library.bases.find((base) => base.id === baseId);

  function openImport() {
    setFile(null);
    setStage("");
    setError("");
    setSuccess("");
    setUploadBase(
      String(baseId === "all" ? library.bases[0]?.id || "" : baseId),
    );
    setImportOpen(true);
    modal.current?.showModal();
  }

  function openBase(base?: KnowledgeBaseRecord) {
    setEditor(
      base
        ? { id: base.id, name: base.name, description: base.description }
        : { name: "", description: "" },
    );
    setError("");
    modal.current?.showModal();
  }

  function closeModal() {
    modal.current?.close();
    setEditor(null);
    setRemoval(null);
    setImportOpen(false);
    setError("");
  }

  async function saveBase(event: FormEvent) {
    event.preventDefault();
    if (!editor) return;
    setBusy(true);
    setError("");
    try {
      await request(`/knowledge-bases${editor.id ? `/${editor.id}` : ""}`, {
        method: editor.id ? "PUT" : "POST",
        body: JSON.stringify({
          name: editor.name.trim(),
          description: editor.description.trim(),
        }),
      });
      await onRefresh();
      closeModal();
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: FormEvent) {
    event.preventDefault();
    if (!file || !uploadBase) return;
    setBusy(true);
    setError("");
    try {
      const document = await importDocument(file, Number(uploadBase), setStage);
      await onRefresh();
      closeModal();
      setSuccess(
        `已导入「${document.fileName}」，生成 ${document.chunkCount} 个片段。`,
      );
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!removal) return;
    setBusy(true);
    setError("");
    try {
      await request(
        `/${removal.kind === "base" ? "knowledge-bases" : "documents"}/${removal.id}`,
        { method: "DELETE" },
      );
      if (removal.kind === "base" && removal.id === baseId) setBaseId("all");
      await onRefresh();
      closeModal();
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  function preview(doc: Document) {
    onSource({
      doc_name:
        doc.storageFileName || `${doc.fileName}.${doc.type.toLowerCase()}`,
      document_title: doc.title || doc.fileName,
      content:
        doc.summary || "此文档尚未提供摘要，可在图谱中查看已解析的证据片段。",
      heading: doc.title || doc.fileName,
      parent_heading: "",
      score: 0,
      chunk_id: "",
      page_number: doc.type === "PDF" ? 1 : 0,
      page_count: 0,
      section_title: doc.type === "PDF" ? "文档首页预览" : "文档说明",
      source_url: doc.sourceUrl,
      highlight_text: "",
      source_type: doc.type.toLowerCase(),
    });
  }

  return (
    <div className="library-view page-view">
      <header className="page-heading">
        <div>
          <span className="eyebrow">KNOWLEDGE LIBRARY</span>
          <h1>让资料井然有序</h1>
          <p>统一管理航空文档，为每一次检索建立可靠的知识基础。</p>
        </div>
        <button className="primary-button" onClick={openImport}>
          <Upload size={16} />
          导入文档
        </button>
      </header>
      {success && (
        <div className="success-banner" role="status">
          <CheckCircle2 size={16} />
          {success}
          <button onClick={() => setSuccess("")} aria-label="关闭成功提示">
            <X size={15} />
          </button>
        </div>
      )}
      <div className="library-layout">
        <aside className="library-folders">
          <div className="section-label">
            知识库{" "}
            <button
              className="icon-button"
              onClick={() => openBase()}
              aria-label="新建知识库"
            >
              <Plus size={17} />
            </button>
          </div>
          <button
            className={`folder-item ${baseId === "all" ? "selected" : ""}`}
            onClick={() => setBaseId("all")}
          >
            <BookOpen size={16} />
            <span>全部文档</span>
            <small>{library.documents.length}</small>
          </button>
          {library.bases.map((base) => (
            <button
              key={base.id}
              className={`folder-item ${baseId === base.id ? "selected" : ""}`}
              onClick={() => setBaseId(base.id)}
            >
              <Folder size={16} />
              <span>{base.name}</span>
              <small>
                {
                  library.documents.filter(
                    (doc) => doc.knowledgeBaseId === base.id,
                  ).length
                }
              </small>
            </button>
          ))}
          <button className="text-button new-base" onClick={() => openBase()}>
            <Plus size={15} />
            新建知识库
          </button>
        </aside>
        <section className="document-section">
          <div className="document-heading">
            <div>
              <h2>
                {selectedBase?.name || "全部文档"}{" "}
                <span>{visibleDocs.length}</span>
              </h2>
              {selectedBase && (
                <p className="muted small">{selectedBase.description}</p>
              )}
            </div>
            {selectedBase && (
              <div className="row-actions">
                <button
                  className="icon-button"
                  onClick={() => openBase(selectedBase)}
                  aria-label="编辑知识库"
                >
                  <Pencil size={15} />
                </button>
                <button
                  className="icon-button"
                  onClick={() => {
                    setRemoval({
                      kind: "base",
                      id: selectedBase.id,
                      name: selectedBase.name,
                    });
                    modal.current?.showModal();
                  }}
                  aria-label="删除知识库"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            )}
          </div>
          <label className="document-search">
            <Search size={16} />
            <input
              aria-label="搜索文档"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="搜索文档名称…"
            />
            <span>{visibleDocs.length} 份文档</span>
          </label>
          <div className="table-scroll">
            <table className="documents-table">
              <thead>
                <tr>
                  <th>文档名称</th>
                  <th>状态</th>
                  <th>片段</th>
                  <th>导入日期</th>
                  <th>
                    <span className="sr-only">操作</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {visibleDocs.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <button
                        className="document-name"
                        onClick={() => preview(doc)}
                      >
                        <span className="file-icon">
                          <FileText size={18} />
                        </span>
                        <span>
                          <strong title={doc.title || doc.fileName}>
                            {doc.title || doc.fileName}
                          </strong>
                          <small>
                            {doc.type} · {doc.size}
                          </small>
                        </span>
                      </button>
                    </td>
                    <td>
                      <span
                        className={`status-pill ${doc.status === "FAILED" ? "failed" : doc.chunkCount > 0 ? "ready" : "pending"}`}
                      >
                        <i />
                        {doc.status === "FAILED"
                          ? "失败"
                          : doc.status === "INDEXING" ||
                              doc.status === "CHUNKING"
                            ? "正在索引"
                            : doc.chunkCount > 0
                              ? "已索引"
                              : "待核验"}
                      </span>
                    </td>
                    <td className="tabular">{doc.chunkCount || "—"}</td>
                    <td className="muted small">
                      {doc.uploadedAt.split(" ")[0]}
                    </td>
                    <td>
                      <button
                        className="icon-button row-delete"
                        aria-label={`删除 ${doc.fileName}`}
                        onClick={() => {
                          setRemoval({
                            kind: "document",
                            id: doc.id,
                            name: doc.fileName,
                          });
                          modal.current?.showModal();
                        }}
                      >
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!visibleDocs.length && (
            <div className="empty-state">
              <Folder size={30} />
              <h3>{filter ? "未找到匹配文档" : "这里还没有文档"}</h3>
              <p>
                {filter
                  ? "换一个名称或切换知识库试试。"
                  : "支持办公文档、网页、Markdown、文本和结构化数据。"}
              </p>
              {!filter && (
                <button className="secondary-button" onClick={openImport}>
                  <Upload size={15} />
                  导入文档
                </button>
              )}
            </div>
          )}
          <p className="library-note">
            “待核验”表示管理记录尚未提供有效片段数，实际检索结果以索引为准。
          </p>
        </section>
      </div>
      <dialog
        ref={modal}
        className="modal"
        onCancel={(event) => {
          if (busy) event.preventDefault();
          else closeModal();
        }}
        onClick={(event) => {
          if (event.target === modal.current && !busy) closeModal();
        }}
      >
        <header>
          <h2>
            {editor
              ? editor.id
                ? "编辑知识库"
                : "新建知识库"
              : removal
                ? "确认删除"
                : "导入文档"}
          </h2>
          <button
            className="icon-button"
            onClick={closeModal}
            disabled={busy}
            aria-label="关闭对话框"
          >
            <X size={19} />
          </button>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}
        {editor && (
          <form onSubmit={(event) => void saveBase(event)}>
            <label className="field">
              名称
              <input
                autoFocus
                required
                maxLength={80}
                value={editor.name}
                onChange={(event) =>
                  setEditor({ ...editor, name: event.target.value })
                }
                placeholder="例如：飞行安全与适航资料"
              />
            </label>
            <label className="field">
              说明
              <textarea
                rows={3}
                value={editor.description}
                onChange={(event) =>
                  setEditor({ ...editor, description: event.target.value })
                }
                placeholder="描述这个知识库的资料范围"
              />
            </label>
            <footer>
              <button
                type="button"
                className="secondary-button"
                onClick={closeModal}
                disabled={busy}
              >
                取消
              </button>
              <button
                className="primary-button"
                disabled={busy || !editor.name.trim()}
              >
                {busy && <LoaderCircle size={15} className="spin" />}保存
              </button>
            </footer>
          </form>
        )}
        {importOpen && (
          <form onSubmit={(event) => void upload(event)}>
            <p className="muted">文档将经过解析、切片与索引后加入知识库。</p>
            <label className="field">
              目标知识库
              <select
                required
                value={uploadBase}
                onChange={(event) => setUploadBase(event.target.value)}
                disabled={busy}
              >
                <option value="" disabled>
                  请选择知识库
                </option>
                {library.bases.map((base) => (
                  <option key={base.id} value={base.id}>
                    {base.name}
                  </option>
                ))}
              </select>
            </label>
            <input
              className="sr-only"
              ref={fileInput}
              type="file"
              accept={documentAccept}
              disabled={busy}
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <button
              className="upload-zone"
              type="button"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
            >
              <Upload size={26} />
              <strong>{file?.name || "点击选择文档"}</strong>
              <span>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
                  : supportedDocumentFormats.join(" · ")}
              </span>
            </button>
            {stage && (
              <p className="upload-stage" role="status">
                {busy && <LoaderCircle size={16} className="spin" />}
                {stage}
              </p>
            )}
            <footer>
              <button
                type="button"
                className="secondary-button"
                onClick={closeModal}
                disabled={busy}
              >
                取消
              </button>
              <button
                className="primary-button"
                disabled={!file || !uploadBase || busy}
              >
                {busy ? "正在处理…" : "导入并建立索引"}
              </button>
            </footer>
          </form>
        )}
        {removal && (
          <div>
            <p>删除「{removal.name}」？</p>
            <p className="muted">
              {removal.kind === "base"
                ? "该知识库及其原始文件、文档记录和检索索引将一并移除。"
                : "该文档的原始文件、管理记录和检索索引将被移除。"}
              此操作无法从页面撤销。
            </p>
            <footer>
              <button
                className="secondary-button"
                onClick={closeModal}
                disabled={busy}
              >
                取消
              </button>
              <button
                className="danger-button"
                onClick={() => void remove()}
                disabled={busy}
              >
                {busy ? "删除中…" : "确认删除"}
              </button>
            </footer>
          </div>
        )}
      </dialog>
    </div>
  );
}
