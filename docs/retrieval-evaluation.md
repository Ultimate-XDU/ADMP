# Retrieval evaluation

This interface returns the passages a run actually selected. It does not compute quality scores and it does not read gold labels. Score the JSONL outside this repository, against the private labels, on the same corpus and the same candidate and context budgets.

## Query file

JSONL objects. Only these fields are used:

- `query_id` (string; a line number is used when it is absent)
- `query` (required string)
- `doc_ids` (optional list of document ids)

Any other key, including gold evidence, is ignored and is not sent to a model. A line without `query` is rejected. `doc_ids` must be a list when it is present. An empty list means every document. A non-empty list is applied to every retrieval tool and overrides a title mentioned in the query.

## Command

From `BackEnd`, with the project virtualenv:

```bash
PYTHONPATH=. python -m scripts.eval_retrieval \
  --queries queries.jsonl \
  --mode hybrid \
  --output hits.jsonl \
  --candidate-limit 8 \
  --context-chars 5600 \
  --passage-chars 900 \
  --rerank-pool 16
```

`--tool-calls` replaces the facet rule with a fixed cap from 1 to 8. Leave it unset to keep the rule below. The process uses `GRAPH_DATA_DIR` and `GRAPH_CORPUS_DIR`. The script does not download a corpus and does not start a server.

Library entry points are `load_queries`, `run_query`, and `run_query_file` in `knowledge.graph_rag.eval_retrieval`. `run_query` accepts an in-memory snapshot so a comparison can reuse one loaded graph.

## Modes

| mode | what runs | answer model |
| --- | --- | --- |
| `baseline` | Frozen package `work/eval20/baseline-code` in a new interpreter. Its files are not edited. | no |
| `hybrid` | Window BM25 and dense ranks, fused by reciprocal rank, then one rerank batch. | no |
| `hybrid_graph` | The same hybrid seeds, plus graph bridges that cover a facet those seeds miss. | no |
| `agent` | Planner and answer generation on the current graph service. | yes |

`hybrid` and `hybrid_graph` set `generated` to false and `answer` to null. `agent` sets `generated` to true. Do not compare an agent answer's confidence with a retrieval score.

The frozen ranker still embeds `content[:2000]`, mixes lexical and vector magnitudes (`0.45` / `0.55`), and keeps eight hits. The harness then truncates that output to the shared candidate and context budgets. Improved retrieval embeds every window of the stored text into `window_vectors.npz`. `vectors.npz` stays the legacy prefix index so the frozen package can read the same data directory. Neither index rewrites the LightRAG graph.

## Result row

Each output line has:

- `query_id`, `mode`, `generated`, `answer`
- `candidates`: ordered spans, each with `unit_id`, `doc_id`, `start`, `end`, `text`, `score`, `ranks`, and source fields. `text` is `content[start:end]` of the original unit.
- `packed_spans`: the windows that fit the context budget, one relevant window per question aspect when the text supports it
- `trace`: tool step, action, query, summary, `nodeIds`, `sourceIds`. No hidden reasoning.
- `timings.seconds`: wall time for that query
- `budgets`: `candidates`, `context_chars`, `passage_chars`, `rerank_pool`, `tool_calls`, `tool_call_base`, `note`
- `tool_calls_made`: how many tool steps ran. This is not a quality score.
- `error`: present only when that query failed

`hybrid` and `hybrid_graph` are given the same budget object. Hybrid seeds stay ahead of graph hits. A bridge may take a free candidate slot, or replace a later seed whose aspect is already covered. It does not push out a unique seed. There is no `recall`, `precision`, `mrr`, or `ndcg` field.

## How to read scores

Improved `score` is reciprocal rank fusion with k = 60: the sum of `1 / (60 + rank)` over the lexical, dense, alias, and rerank orderings. Cosine magnitude and rerank logits are not added. `ranks` records the 1-based place on each list that contained the span.

Baseline `score` is whatever the frozen ranker stored (lexical overlap, the old magnitude mix, or a rerank logit). A larger baseline number is not on the same scale as an improved score. Compare the ordered ids and spans, then compute metrics externally.

The tool cap starts at 4. A question with two or more aspects may add one call per extra aspect, at most two, so the cap is 4, 5, or 6. `GRAPH_MAX_TOOL_CALLS` replaces that rule with a value from 1 to 8. `budgets.note` states which rule was used. The frozen agent does not read this cap; it stays at 4.

## Metrics that belong outside this interface

On the held-out labels, compute Recall@5, Recall@10, Precision@5, Precision@10, MRR, nDCG@10, and all-required-evidence coverage from the returned spans. A hit is a selected span that contains a required evidence quote, or that cites the annotated unit and offsets when the label has them. Precision uses the annotated relevant set only. Unannotated passages are not counted as relevant and are not counted as irrelevant; with sparse labels, precision is a lower bound on judged precision. Do not treat graph size, source-id validity, model confidence, or the number of passages as a metric.

Answer status, when generation is enabled, is separate from those ranks:

- `answered`: the packed spans support every aspect and the answer cites them
- `partial_evidence`: at least one aspect is supported and cited; unsupported aspects stay unsupported. The cited text is kept.
- `insufficient_evidence` or `uncited_answer`: nothing usable was cited. The answer is the abstain sentence.

## Corpus and graph coverage

`scope=all` indexes every non-empty parsed unit. `scope=pilot` drops bibliography units and, only when more than 12 units remain, samples 12 of them. `scope=smoke` keeps two units. `fullGraph` is true only when the scope is `all` and every parsed unit was extracted. Otherwise the status note is `partial graph coverage; not a full knowledge graph`. `availableChunks` remains the number of stored chunks, not a claim that the graph is complete. Window rebuilds do not rebuild the graph, and a rerank change does not either.
