import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";
import type { Root, PhrasingContent } from "mdast";
import type { Citation, Evidence } from "../api";

export function AnswerText({
  text,
  sources,
  citations,
  onSelect,
}: {
  text: string;
  sources: Evidence[];
  citations: Citation[];
  onSelect: (source: Evidence) => void;
}) {
  const citationMap = new Map(
    citations.map((citation) => [citation.id, citation]),
  );

  function remarkCitations() {
    return (tree: Root) => {
      visit(tree, "text", (node, index, parent) => {
        if (
          !parent ||
          index === undefined ||
          parent.type === "link" ||
          parent.type === "linkReference"
        )
          return;
        const parts: PhrasingContent[] = [];
        let offset = 0;
        for (const match of node.value.matchAll(/\[(\d+)\]/g)) {
          const id = Number(match[1]);
          if (!citationMap.has(id)) continue;
          if (match.index! > offset)
            parts.push({
              type: "text",
              value: node.value.slice(offset, match.index),
            });
          parts.push({
            type: "link",
            url: `#admp-citation-${id}`,
            children: [{ type: "text", value: String(id) }],
          });
          offset = match.index! + match[0].length;
        }
        if (!parts.length) return;
        if (offset < node.value.length)
          parts.push({ type: "text", value: node.value.slice(offset) });
        parent.children.splice(index, 1, ...parts);
        return index + parts.length;
      });
    };
  }

  return (
    <div className="answer-text">
      <Markdown
        remarkPlugins={[remarkGfm, remarkCitations]}
        skipHtml
        components={{
          a: ({ href, children }) => {
            const id = href?.startsWith("#admp-citation-")
              ? Number(href.slice(15))
              : null;
            const citation = id === null ? undefined : citationMap.get(id);
            return citation ? (
              <button
                className="citation"
                onClick={() => onSelect(sources[citation.sourceIndex])}
                aria-label={`查看引用 ${citation.id}`}
              >
                {children}
              </button>
            ) : (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
          table: ({ children }) => (
            <div className="answer-table">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
