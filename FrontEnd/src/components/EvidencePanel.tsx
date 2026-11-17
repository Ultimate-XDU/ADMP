import { useEffect, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  FileText,
  X,
} from "lucide-react";
import {
  type Citation,
  type Evidence,
  pageImageUrl,
  evidenceLocationLabel,
} from "../api";

export function EvidencePanel({
  source,
  onClose,
}: {
  source: Evidence;
  onClose: () => void;
}) {
  const [page, setPage] = useState(source.page_number);
  const [imageFailed, setImageFailed] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  useEffect(() => {
    setPage(source.page_number);
  }, [source]);
  useEffect(() => {
    setImageFailed(false);
    setImageLoaded(false);
  }, [source.doc_name, page]);
  const pdfPreview = source.doc_name.toLowerCase().endsWith(".pdf");
  const highlight =
    source.bbox_pages?.find((item) => item.page_number === page)?.bbox ||
    (source.page_number === page ? source.bbox : undefined);
  return (
    <aside className="evidence-panel" aria-label="来源详情">
      <header>
        <span className="eyebrow">SOURCE VIEWER</span>
        <button className="icon-button" onClick={onClose} aria-label="关闭来源">
          <X size={18} />
        </button>
      </header>
      <div className="evidence-title">
        <span className="file-icon">
          <FileText size={20} />
        </span>
        <h3>{source.document_title || source.doc_name}</h3>
      </div>
      <p className="muted small">
        {source.article_label ||
          source.section_title ||
          source.heading ||
          "原文证据"}
      </p>
      {pdfPreview && (
        <>
          <div className="page-toolbar">
            <button
              className="icon-button"
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              aria-label="上一页"
            >
              <ChevronLeft size={17} />
            </button>
            <span>
              第 {page} 页{" "}
              {source.page_count > 0 && (
                <span className="muted">/ {source.page_count}</span>
              )}
            </span>
            <button
              className="icon-button"
              disabled={page >= source.page_count}
              onClick={() => setPage(page + 1)}
              aria-label="下一页"
            >
              <ChevronRight size={17} />
            </button>
            <a
              className="icon-button"
              href={pageImageUrl(source.doc_name, page)}
              target="_blank"
              rel="noreferrer"
              aria-label="打开原页大图"
            >
              <ExternalLink size={15} />
            </a>
          </div>
          <div className="page-preview">
            {!imageLoaded && !imageFailed && (
              <div className="preview-loading">正在加载原页…</div>
            )}
            {imageFailed ? (
              <div className="preview-unavailable">
                <FileText size={30} />
                <p>此来源暂无页图预览</p>
                <small>可在下方核对检索原文。</small>
              </div>
            ) : (
              <img
                src={pageImageUrl(source.doc_name, page)}
                alt={`${source.doc_name} 第 ${page} 页`}
                onLoad={() => setImageLoaded(true)}
                onError={() => setImageFailed(true)}
              />
            )}
            {imageLoaded && !imageFailed && highlight && (
              <span
                className="evidence-highlight"
                style={{
                  left: `${highlight.x * 100}%`,
                  top: `${highlight.y * 100}%`,
                  width: `${highlight.width * 100}%`,
                  height: `${highlight.height * 100}%`,
                }}
              />
            )}
          </div>
        </>
      )}
      {!pdfPreview && (
        <p className="muted small source-location">
          {evidenceLocationLabel(source)}
        </p>
      )}
      {source.source_url && (
        <a
          className="text-button"
          href={source.source_url}
          target="_blank"
          rel="noreferrer"
        >
          访问原始来源 <ExternalLink size={13} />
        </a>
      )}
      <section className="source-excerpt">
        <div className="section-label">
          {source.chunk_id ? "检索原文" : "文档说明"}{" "}
          <span>{evidenceLocationLabel(source)}</span>
        </div>
        <p>{source.content}</p>
      </section>
    </aside>
  );
}

export function SourceCards({
  sources,
  citations,
  onSelect,
}: {
  sources: Evidence[];
  citations: Citation[];
  onSelect: (source: Evidence) => void;
}) {
  return (
    <div className="source-cards">
      {sources.map((source, index) => {
        const citation = citations.find((item) => item.sourceIndex === index);
        return (
          <button
            className="source-card"
            key={`${source.chunk_id}-${index}`}
            onClick={() => onSelect(source)}
          >
            <span className="source-number">
              {citation?.id || <FileText size={12} />}
            </span>
            <div>
              <strong>{source.document_title || source.doc_name}</strong>
              <span>
                {evidenceLocationLabel(source)} ·{" "}
                {source.article_label ||
                  source.section_title ||
                  source.heading ||
                  "原文片段"}
              </span>
            </div>
            <ChevronRight size={15} />
          </button>
        );
      })}
    </div>
  );
}
