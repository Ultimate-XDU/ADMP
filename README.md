# Aviation Knowledge Command

本项目是一个面向航空法规、适航资料、维修资料的本地知识库检索与问答系统。前端负责知识库管理、文档上传、问答会话和证据展示；后端负责文档入库、结构化解析、混合检索、证据定位和大模型回答生成。

## 目录结构

```text
D:\innovat
  BackEnd\
    aerospace_api.py                 FastAPI 服务入口，给前端提供统一 API
    aerospace_search_main.py         文档解析、切块、索引、混合检索、重建索引
    aerospace_structure.py           法规/适航文档结构树构建与结构化问答
    aerospace_qa.py                  QA prompt、证据筛选、PDF 上下文补强、大模型调用
    import_caac_documents.py         批量导入 CAAC 文档
    import_public_reference_documents.py
    index_caac_imported_documents.py 批量索引导入文档
    reindex_documents_one_by_one.py  清库后逐个重建索引
    aerospace_docs\                 本地知识库原文目录，不提交 Git
    structural_records\             本地结构化索引缓存，不提交 Git
  FrontEnd\
    src\App.tsx                     主界面、知识库管理、问答、证据预览
    src\aviationAssistantMock.ts    前端类型和兜底数据
    scripts\*.mjs                   依赖检查、构建、预览脚本
    start.bat                       Windows 前端启动脚本
```

## 核心业务流程

1. 用户在前端上传 PDF/DOCX 或手工登记文档。
2. 前端调用后端 `/api/upload` 保存文件到 `BackEnd/aerospace_docs`。
3. 前端调用 `/api/index` 或 `/api/rebuild-index` 触发索引。
4. 后端解析文档内容，提取页码、文本块、表格、标题、条款、款项、证据定位信息。
5. `aerospace_structure.py` 构建结构树，把文档组织成 `doc/chapter/section/article/clause/item/toc` 等结构记录。
6. `aerospace_search_main.py` 生成检索文本和向量，写入 Elasticsearch 与 Milvus。
7. 用户提问时，前端调用 `/api/qa`。
8. 后端先分析问题意图，再执行 BM25、向量、结构化记录、标题/条款等多路召回。
9. 候选证据经过 rerank、结构化优先级、质量门和 PDF 页面上下文补强。
10. `aerospace_qa.py` 组装 prompt，调用通义千问兼容 OpenAI 接口生成答案。
11. 后端返回答案、引用来源、页码、bbox、高亮文本和会话记录。
12. 前端展示回答，并可在证据面板中打开 PDF 页图和高亮定位。

## 后端逻辑

`aerospace_api.py` 是服务入口，主要负责：

- 加载 `.env` 中的环境变量。
- 初始化 FastAPI 和 CORS。
- 懒加载 `HybridSearch`，避免服务启动时就占用模型资源。
- 提供问答、检索、证据页图、聊天历史、知识库管理、文档上传、索引、分类、标题摘要等 API。
- 维护 `management_state.json` 文档管理状态和 `chat_history.sqlite3` 本地会话库。

`aerospace_search_main.py` 是检索主链路，主要负责：

- 解析 PDF/DOCX，抽取正文、标题、表格、页码和布局信息。
- 按法规/适航资料特点进行语义切块，并保留 `prev_id/next_id/parent_id/heading_path/clause_path` 等上下文链路。
- 调用 `aerospace_structure.py` 生成结构化记录，并将结构字段回填到 chunk。
- 写入 Milvus 向量库和 Elasticsearch 关键词索引。
- 查询时融合向量召回、关键词召回、结构化召回、条款定位、标题匹配和 rerank。

`aerospace_structure.py` 是结构化增强层，主要负责：

- 识别章节、条、款、项、目录页、编号标题等结构。
- 构建父子关系、直接子节点数、后代数、路径、页码范围。
- 支持结构计数、定义类、分类类问题的直接回答。
- 避免普通 QA 被目录页或弱语义证据干扰。

`aerospace_qa.py` 是答案生成层，主要负责：

- 识别追问和历史上下文，必要时把最近用户问题拼入检索 query。
- 选择高质量证据进入 prompt，过滤明显弱相关来源。
- 对非结构证据补充 PDF 页上下文。
- 调用 DashScope 兼容 OpenAI 接口的 `qwen-turbo`。
- 返回可被前端定位的 sources、highlight、bbox、bbox_pages。

## 前端逻辑

前端是 Vite + React + TypeScript 应用，主要逻辑集中在 `FrontEnd/src/App.tsx`：

- 知识库管理：调用 `/api/knowledge-bases` 和 `/api/documents` 维护分类、文档列表和状态。
- 文档上传：调用 `/api/upload`，上传完成后调用 `/api/index` 建索引。
- 问答会话：调用 `/api/qa`，保存和恢复 `/api/chat/conversations`。
- 证据展示：根据后端返回的 `sources`、`page_number`、`bbox`、`bbox_pages` 展示引用卡片和 PDF 高亮。
- 删除会话：调用 `/api/chat/conversations/{conversation_id}`。

前端默认 API 前缀是 `/api`。开发环境下 `FrontEnd/vite.config.ts` 会把 `/api` 代理到：

```text
http://127.0.0.1:8787
```

## 本地运行

后端使用用户本机 conda 环境：

```powershell
D:\miniconda3\envs\PYcode\python.exe BackEnd\aerospace_api.py
```

后端默认监听：

```text
http://127.0.0.1:8787
```

前端启动：

```powershell
cd FrontEnd
.\start.bat
```

也可以手动执行：

```powershell
cd FrontEnd
npm run dev -- --open
```

## 运行依赖

后端运行前需要确认：

- Elasticsearch 在 `http://localhost:9200` 可访问。
- Milvus 在 `localhost:19530` 可访问。
- `BackEnd/.env` 中配置了 `QWEN_API_KEY`。
- 本地可加载 embedding 模型 `BAAI/bge-m3`。
- 本地可加载 rerank 模型 `BAAI/bge-reranker-v2-m3`。
- 如果没有 GPU，可通过 `AEROSPACE_RERANK_DEVICE` 调整 rerank 设备。

后端常用环境变量：

```text
QWEN_API_KEY
LOG_LEVEL
AEROSPACE_EMBEDDING_MODEL
AEROSPACE_RERANK_MODEL
AEROSPACE_RERANK_DEVICE
AEROSPACE_VECTOR_DIM
AEROSPACE_SKIP_UPLOAD_DIR_SYNC
AEROSPACE_HF_ENABLE_PARALLEL_LOADING
```

## 数据和 Git 边界

本仓库提交代码、脚本和说明，不提交本地知识库原文和运行状态。

以下内容被 `.gitignore` 忽略：

- `BackEnd/aerospace_docs/`
- `BackEnd/structural_records/`
- `BackEnd/chat_history.sqlite3`
- 后端日志、模型下载日志、索引日志
- 本地导入/健康检查报告 JSON
- 前端 `node_modules/`、`dist/`、`*.tsbuildinfo`

因此，对方拉取代码后不会自动获得本机 226 个知识库文档，也不会自动拥有本机索引库内容。要复现完整问答效果，需要单独同步文档目录，并在同样的 Elasticsearch/Milvus 环境中重新索引，或连接同一套索引服务。

## 常用 API

```text
GET    /api/health
POST   /api/qa
POST   /api/search
GET    /api/evidence/page-image
GET    /api/chat/conversations
GET    /api/chat/conversations/{conversation_id}
PATCH  /api/chat/conversations/{conversation_id}
DELETE /api/chat/conversations
DELETE /api/chat/conversations/{conversation_id}
GET    /api/knowledge-bases
POST   /api/knowledge-bases
PUT    /api/knowledge-bases/{kb_id}
DELETE /api/knowledge-bases/{kb_id}
GET    /api/documents
POST   /api/documents
DELETE /api/documents/{doc_id}
POST   /api/upload
POST   /api/index
POST   /api/rebuild-index
DELETE /api/index/{doc_name}
POST   /api/classify
POST   /api/summarize-title
POST   /api/reload-modules
```

## 批量导入和重建索引

导入 CAAC 文档：

```powershell
D:\miniconda3\envs\PYcode\python.exe BackEnd\import_caac_documents.py
```

导入公共参考资料：

```powershell
D:\miniconda3\envs\PYcode\python.exe BackEnd\import_public_reference_documents.py
```

批量索引已导入文档：

```powershell
cd BackEnd
D:\miniconda3\envs\PYcode\python.exe index_caac_imported_documents.py --ready --start 1 --limit 5 --sleep 2
```

逐个重建索引：

```powershell
D:\miniconda3\envs\PYcode\python.exe BackEnd\reindex_documents_one_by_one.py
```

## 开发注意事项

- 不要提交 `.env`、日志、SQLite、本地知识库原文、结构化缓存和前端构建产物。
- 后端索引状态和前端展示数量取决于运行机器上的 `BackEnd/aerospace_docs` 和 `management_state.json`。
- 问答质量同时依赖文档原文、结构化记录、ES/Milvus 索引和 Qwen API。
- 修改检索或结构化逻辑后，需要对相关文档重新索引，已有索引不会自动包含新字段。
