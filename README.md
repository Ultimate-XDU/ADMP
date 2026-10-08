# ADMP 航空文档智能检索平台

ADMP 是一个面向航空法规、适航资料和维修资料的本地知识库检索与问答系统。系统将文档解析、结构化记录、Elasticsearch 关键词检索、Milvus 向量检索、BGE 重排序和基于证据的大模型问答连接在一起。

当前部署方案面向 Windows + Docker Desktop：数据库服务运行在 Docker 中，后端与前端运行在本机。所有服务端口仅绑定到本机回环地址，不对局域网开放。

## 功能概览

- 上传并索引 PDF、DOCX 文档。
- 提取标题、条款、款项、页码、表格和文档结构，生成可追溯的结构化证据。
- 融合向量、关键词、标题、条款、结构记录和目标文档等检索通道。
- 使用 BGE-M3 生成向量，并使用 BGE Reranker v2-m3 对候选证据重排序。
- 通过 DeepSeek 的 OpenAI 兼容接口生成基于证据的答案。
- 在前端显示会话、答案、页码和证据卡片，可定位到原文 PDF 页面。

## 项目结构

```text
D:\ADMP
├─ BackEnd\
│  ├─ api\aerospace_api.py        FastAPI 服务入口
│  ├─ aerospace_search_main.py    文档解析、索引与混合检索入口
│  ├─ aerospace_qa.py             问答兼容入口
│  ├─ qa\                         Prompt、证据选择和问答流程
│  ├─ search\                     向量、ES、重排序和检索策略实现
│  ├─ utils\structure.py          法规结构树与结构记录
│  ├─ aerospace_docs\             本地原始文档目录
│  └─ api\.env                    本机模型与 API 配置，不提交 Git
├─ FrontEnd\                      Vite + React + TypeScript 前端
├─ docker\elasticsearch\          带 IK 分词器的 Elasticsearch 镜像
└─ docker-compose.yml             Elasticsearch、Milvus、etcd、MinIO
```

## 架构与数据流

```text
浏览器前端 (5173)
        │ /api
        ▼
FastAPI 后端 (8787)
 ├─ Elasticsearch (9200)  关键词、标题、条款检索，IK 分词
 ├─ Milvus (19530)        BGE-M3 向量检索
 ├─ 本地 BGE-M3           Query embedding
 ├─ 本地 BGE Reranker     候选证据重排序
 └─ DeepSeek API          基于证据的最终回答与检索策略代理
```

文档上传至 `BackEnd/aerospace_docs/` 后，后端会生成文本块和法规结构记录，并分别写入 Elasticsearch、Milvus 和 `BackEnd/utils/structural_records/`。提问时，系统先多路召回和融合，再执行重排序、证据质量筛选和答案生成。

## 环境要求

- Windows 10/11，Docker Desktop 已启动并使用 Linux containers。
- Conda 或 Miniconda。当前已验证的环境名称为 `admp`，Python 为 3.11.15。
- Node.js 18+ 与 npm。当前前端已验证 Node.js 24.18.0、npm 11.16.0。
- NVIDIA GPU 为推荐配置。没有 GPU 时可将重排序设备设为 `cpu`，但索引和问答检索速度会降低。

当前 `admp` 环境已验证以下关键 Python 包：FastAPI 0.141.1、Uvicorn 0.52.4、Elasticsearch 8.19.3、PyMilvus 2.6.17、Sentence Transformers 5.7.0、PyMuPDF 1.28.2、python-docx 1.2.0、OpenAI 2.54.0。

仓库当前未提供 Python 依赖锁定文件。若在新机器重建环境，请基于以上已验证版本安装依赖，并在启动前执行 `python -m pip check`。

## 1. 启动 Docker 检索服务

在 Docker Desktop 已运行的前提下，于项目根目录执行：

```powershell
Set-Location D:\ADMP
docker compose up -d --build
docker compose ps
```

Compose 服务及端口：

| 服务 | 版本或镜像 | 本机地址 | 用途 |
| --- | --- | --- | --- |
| Elasticsearch + IK | Elasticsearch 8.4.1 + 匹配 IK 插件 | `http://127.0.0.1:9200` | 中文关键词、标题和条款检索 |
| Milvus | `milvusdb/milvus:v2.4.23` | `127.0.0.1:19530` | 向量检索 |
| Milvus Web UI | Milvus 内置 | `http://127.0.0.1:9091/webui/` | Milvus 状态查看 |
| etcd | `quay.io/coreos/etcd:v3.5.5` | 不暴露 | Milvus 元数据 |
| MinIO | `minio/minio:RELEASE.2023-03-20T20-16-18Z` | 不暴露 | Milvus 对象存储 |

Elasticsearch 镜像在构建时安装与 Elasticsearch 版本严格匹配的 IK 插件。不要改用不带 IK 的官方 Elasticsearch 镜像，否则索引中使用的 `ik_max_word` 和 `ik_smart` 分析器无法创建。

更多 Docker 说明见 [docker/README.md](docker/README.md)。

停止数据库服务：

```powershell
docker compose down
```

不要在不打算清空索引时执行 `docker compose down --volumes`，该命令会删除 Elasticsearch 和 Milvus 的持久化数据卷。

## 2. 配置模型与 DeepSeek

后端只读取 `BackEnd/api/.env`。此文件已被 `.gitignore` 忽略，不能提交 API Key、文档或本机绝对路径。

示例配置如下，其中 API Key 必须替换为自己的密钥：

```dotenv
# 本地模型快照
AEROSPACE_EMBEDDING_MODEL=D:\ADMP\models\BAAI--bge-m3
AEROSPACE_RERANK_MODEL=D:\ADMP\models\BAAI--bge-reranker-v2-m3
AEROSPACE_RERANK_DEVICE=cuda

# DeepSeek OpenAI-compatible Chat Completions
DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_THINKING_ENABLED=true
LLM_REASONING_EFFORT=medium

# 让检索策略代理使用同一个模型与合理的思考超时
RETRIEVAL_AGENT_MODEL=deepseek-v4-flash
RETRIEVAL_AGENT_TIMEOUT=45
```

项目已使用的本地模型目录为：

```text
D:\ADMP\models\BAAI--bge-m3
D:\ADMP\models\BAAI--bge-reranker-v2-m3
```

它们分别对应 ModelScope 上的 `BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3` 快照。重排序模型以本地离线模式加载；请先完成下载再启动后端。

DeepSeek 的 `deepseek-v4-flash` 使用思考模式时，用户配置的 `medium` 会被官方接口映射为实际 `high` 推理档位。详见 [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)。若不需要思考模式，可设为：

```dotenv
LLM_THINKING_ENABLED=false
```

当未配置 `DEEPSEEK_API_KEY` 时，代码保留 `QWEN_API_KEY` 的兼容回退；此时还需按所用服务提供对应的 `LLM_BASE_URL` 与 `LLM_MODEL`。

## 3. 启动后端

必须以模块方式从 `BackEnd` 目录启动：

```powershell
Set-Location D:\ADMP\BackEnd
& C:\Users\MSIK\miniconda3\envs\admp\python.exe -m api.aerospace_api
```

后端地址：

- API: `http://127.0.0.1:8787`
- Swagger UI: `http://127.0.0.1:8787/docs`
- 健康检查: `http://127.0.0.1:8787/api/health`

健康检查应返回 `status: "ok"`，且 `details.elasticsearch`、`details.milvus` 均为 `true`：

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/health
```

## 4. 启动前端

```powershell
Set-Location D:\ADMP\FrontEnd
npm run dev -- --host 127.0.0.1
```

首次启动会由前端脚本执行依赖检查和安装。开发地址默认为：

```text
http://127.0.0.1:5173
```

`FrontEnd/vite.config.ts` 已将 `/api` 代理到 `http://127.0.0.1:8787`。打开 [智能问答页](http://127.0.0.1:5173/aviation-assistant) 后即可查看会话、回答和引用证据。

构建并使用自带 API 反代的静态服务：

```powershell
Set-Location D:\ADMP\FrontEnd
npm run build
$env:API_PROXY_TARGET = "http://127.0.0.1:8787"
npm run serve:dist
```

`serve:dist` 默认监听 4173 端口，并将 `/api/*` 转发到 `API_PROXY_TARGET`。

## 5. 导入与建立索引

将 PDF 或 DOCX 放入 `BackEnd/aerospace_docs/`，可在前端上传并建立索引，也可调用后端接口重建全部索引：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8787/api/rebuild-index
```

索引过程会写入：

- Elasticsearch 索引：`aerospace_kw_db`
- Milvus collection：`aerospace_m3e_db`
- 结构记录缓存：`BackEnd/utils/structural_records/`
- 文档健康报告：`BackEnd/search/document_health_reports.json`

完成后，可在前端提问。例如：

```text
航空发动机燃油排泄和排气排出物应当符合哪些规定？
```

该问题会命中《航空发动机适航规定》中的第 33.1 条，并由 DeepSeek 根据命中的证据生成回答。

## 常用 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 后端、Elasticsearch 和 Milvus 状态 |
| POST | `/api/qa` | 证据驱动的问答 |
| POST | `/api/search` | 仅检索，不调用 LLM |
| POST | `/api/upload` | 上传文档 |
| POST | `/api/index` | 索引指定文档 |
| POST | `/api/rebuild-index` | 重建本地文档索引 |
| DELETE | `/api/index/{doc_name}` | 删除指定文档索引 |
| GET | `/api/evidence/page-image` | 获取 PDF 证据页图 |
| GET | `/api/chat/conversations` | 获取会话列表 |
| GET | `/api/documents` | 获取文档列表 |
| GET | `/api/knowledge-bases` | 获取知识库列表 |

完整接口定义可在后端启动后访问 Swagger UI。

## 检索策略说明

问答链路按如下顺序工作：

1. 分析问题意图、条款编号、目标文档和检索约束。
2. 从向量、关键词、标题、结构、条款、表格和目标文档通道召回候选。
3. 融合召回结果，使用 BGE Reranker 重排序。
4. 对结构类问题优先直接使用章、节、条、款、项等结构记录。
5. 筛选高质量证据并补充必要的 PDF 页上下文。
6. 将问题和证据发送到配置的 LLM，返回带来源、页码与高亮信息的答案。

检索策略代理与最终答案生成使用同一套 LLM 配置。关闭或不可用时，检索会回退到已有的规则策略，但最终问答会返回基于本地证据的不可用提示。

## 本地数据与 Git 边界

下列内容属于本机运行数据，不能提交到 Git：

- `BackEnd/api/.env` 中的 API Key 与本机模型路径。
- `BackEnd/aerospace_docs/` 中的原始文档。
- `models/` 中的模型快照。
- Elasticsearch、Milvus Docker volumes。
- `BackEnd/utils/structural_records/`、文档健康报告和索引结果。
- `BackEnd/api/chat_history.sqlite3` 及其 WAL/SHM 伴随文件。
- 后端、前端运行日志与 `node_modules/`、`dist/`。

在另一台机器复现时，需要单独准备模型快照、原始文档和 API Key，并在新的 Elasticsearch/Milvus 环境中重新建立索引。

## 故障排查

| 现象 | 检查方式 | 处理建议 |
| --- | --- | --- |
| `/api/health` 中 ES 或 Milvus 为 `false` | `docker compose ps` | 确认 Docker Desktop 已启动，再执行 `docker compose up -d --build` |
| ES 建索引时提示 `ik_max_word` 或 `ik_smart` 不存在 | `docker compose logs elasticsearch` | 使用本仓库的 Compose 与 Dockerfile 重建镜像，不要使用裸 ES 镜像 |
| 后端初始化失败 | 查看后端终端输出 | 确认模型目录存在、端口 9200/19530 可访问、`.env` 路径为 `BackEnd/api/.env` |
| 问答返回 LLM 不可用 | 检查 `.env` | 检查 API Key、`LLM_BASE_URL`、`LLM_MODEL` 与网络连接；不要将 Key 写到前端 |
| 前端显示后端不可用 | 打开 `/api/health` | 确认后端运行于 8787，开发前端运行于 5173 |
| 重新修改切块或结构化逻辑后结果未变化 | 检查文档状态和索引 | 删除或重建受影响文档的索引，使新结构字段写入 ES、Milvus 与结构记录缓存 |

## 安全提示

- API Key 仅应保存在 `BackEnd/api/.env` 或安全的环境变量管理工具中。
- 不要将 API Key、原始受限文档、模型私有缓存或 Docker 数据卷提交至仓库。
- LLM 问答会将用户问题与所选证据片段发送给配置的模型服务。处理受限或敏感资料前，请确认组织的数据使用与外发要求。
