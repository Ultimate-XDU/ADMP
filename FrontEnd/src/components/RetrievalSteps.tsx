import { GitBranch } from "lucide-react";
import type { RetrievalTrace } from "../api";

export function RetrievalSteps({ trace }: { trace?: RetrievalTrace | null }) {
  if (!trace?.steps.length) return null;
  return (
    <details className="retrieval-trace">
      <summary>
        <GitBranch size={14} />
        查看检索过程 · {trace.steps.length} 跳
      </summary>
      {trace.planner === "heuristic_fallback" && (
        <p className="trace-note">
          补检规划暂不可用，已改为依据首轮证据中的关键词补检。
        </p>
      )}
      {trace.reason === "no_gap" && (
        <p className="trace-note">首轮后未提出需要继续检索的子问题。</p>
      )}
      <ol>
        {trace.steps.map((step, index) => (
          <li key={index}>
            <span>
              第 {step.hop} 跳 · {step.resultCount} 个候选片段
            </span>
            <p>{step.query}</p>
            {!!step.basisChunkIds?.length && (
              <small>
                基于上一步 {step.basisChunkIds.length} 个片段继续检索
              </small>
            )}
          </li>
        ))}
      </ol>
    </details>
  );
}
