# ADMP 航空知识工作台

面向航空法规、适航与维修资料的本地检索和问答系统。以原文证据为中心，将文档管理、混合检索、层次化知识图谱、问答与引用核验放在同一工作台。

## 当前能力

- React、TypeScript 与 Vite 前端，提供问答、检索轨迹、知识库和图谱页面，支持响应式布局与减少动态效果设置。
- 多源文档解析，支持 PDF、DOCX、PPTX、XLSX 及 HTML、Markdown、TXT、JSON、CSV 等原生文本与表格格式。扫描件和复杂版面仍取决于解析器与 OCR 配置。
- Elasticsearch 关键词检索、BGE-M3 向量召回与 BGE Reranker 重排序。
- 文档、章节、条款和片段的结构层次，以及保留原文出处的实体、关系与跨文档引用。
- 单轮混合检索、确定性图检索与 Agent 检索。工具轨迹、候选来源、实际送入模型的上下文和最终引用分别保留。
- PDF 原页、页码与高亮核验；图文增强使用文本召回后的 PDF 页图，不等同于视觉向量检索。
- 本地 Ollama 模型接入；当前 Ubuntu 部署使用 Qwen3.5:9b。

多跳检索与图谱功能已经接入。召回率、精度及真正依赖前一跳线索的检索能力需要通过独立评测确认，不以图谱规模、工具调用次数或引用编号有效代替质量结论。

## 在现有 Ubuntu 环境运行

当前部署目录中执行：

```bash
./scripts/admp-round2
```

| 服务 | 地址 |
| --- | --- |
| 前端 | http://localhost:5174 |
| API | http://127.0.0.1:8788 |
| 健康检查 | http://127.0.0.1:8788/api/health |
| API 文档 | http://127.0.0.1:8788/docs |

这个入口面向已配置的 Ubuntu 工作目录，复用现有模型和检索服务，不启动 Docker。脚本中的安装路径与端口对应当前部署，不能直接当成任意新机器的一键安装器。运行细节见 [Ubuntu 部署说明](docs/round2-runtime.md)。

原有 `./scripts/admp` 保留为基础启动入口，默认端口为 8787 / 5173。新机器应先准备 Python 依赖、本地模型与 Elasticsearch 配置，再按自己的环境配置路径；原始文档与索引需要单独导入。

## 代码结构

```text
BackEnd/
  api/                    FastAPI 产品接口与图谱接口
  search/                 解析、索引与混合检索
  knowledge/graph_rag/     结构层次、来源登记、图扩展与 Agent 流程
  qa/                     证据选择、回答与引用
  tests/                  后端回归测试
FrontEnd/
  src/                    工作台、知识库、图谱与原文核验
  tests/                  引用组件测试
scripts/                  现有环境启动入口
docs/                     图谱、运行与评测接口说明
requirements.txt          后端依赖版本
```

## 验证

在当前 Ubuntu 环境中：

```bash
PYTHONPATH=BackEnd work/venv/bin/python -m pytest -q BackEnd/tests
cd FrontEnd
npm run build
npm run test:citations
```

后端测试验证实际实现，前端检查覆盖引用编号与来源映射。完整后端回归还依赖本机航空样本与 `work/eval20/baseline-code` 历史基线；这些测试依赖不包含在源码仓库中，缺少时相关检查无法通过。它们不能代替同语料、同预算下的召回与回答评测。检索输出接口见 [评测说明](docs/retrieval-evaluation.md)。

## 本地数据

仓库只包含源码、依赖清单与配置说明。模型权重、数据库、文档原件、会话记录、`.env`、运行日志及未公开评测题保存在部署环境，不随代码发布。公开航空资料仍应保留各来源机构的出处与使用条件。

配置中的模型地址、目录和密钥应从本机环境或 `BackEnd/api/.env` 提供。前端不得包含模型服务密钥。现有数据库与知识库不会因切换 Git 远端而重新创建。

## 技术基础

图谱与检索流程使用 LightRAG、LangGraph、NetworkX 和本地持久化，前端图谱使用 Cytoscape.js。结构关系帮助发现线索，最终结论仍需原文证据。实现与限制见 [图谱说明](docs/graph-rag.md)。
