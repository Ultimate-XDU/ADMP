# 本轮 Ubuntu 运行方式

工程 `/home/yuexing/src/ADMP-graphrag-20261111`，执行 `./scripts/admp-round2`。这是现有 Ubuntu 环境的启动入口，不是重装程序。已监听的服务会复用，不会启动 Docker，也不会访问旧工程的运行数据。

- 前端 http://localhost:5174，Vite 代理环境 `VITE_API_PROXY=http://127.0.0.1:8788`。
- 完整产品 API http://127.0.0.1:8788，模块 `api.aerospace_api:app`，解释器 `work/venv/bin/python`。
- 本地 Qwen3.5:9b 服务 http://127.0.0.1:11435，`OLLAMA_HOST=127.0.0.1:11435`。
- Elasticsearch http://127.0.0.1:9201，当前工程 `work/runtime/es-config` 和 `work/runtime/elasticsearch-data`，复用现有安装二进制。
- Milvus Lite `BackEnd/data/milvus.db`，通过应用默认路径连接；不把文件路径设置为进程级 `MILVUS_URI`，避免 pymilvus 在 import 阶段把路径当 HTTP URI。
- 图谱本轮目录 `work/round2/graph-data`，初始从 `work/eval20/graph-data` 复制；旧评测目录保留不变。当前来源清单 `GRAPH_CORPUS_DIR=work/corpus-100`，由Catalog加载入口规范化既有来源身份，保留产品chunk IDs。
- `GRAPH_AUTO_BUILD=0`，正常索引登记完整结构和原文，不隐式对全部新文档启动 LLM 语义抽取。显式选择文档构建语义图时 `GRAPH_EXTRACT_SCOPE=all`，覆盖所选文档的全量已解析原文。
- CPU BGE-M3、BGE reranker 使用 `/mnt/d/ADMP/models/BAAI--bge-m3` 与 `/mnt/d/ADMP/models/BAAI--bge-reranker-v2-m3`；CPU线程6；GPU用于现有本地生成模型。
- `GRAPH_MAX_TOOL_CALLS=6` 为三种查询模式统一上限；single实际只做1次检索，不代表与多跳相同消耗。检索候选8、重排池16、上下文5600字符、每段900字符。
- `GRAPH_LOG_RAW_LLM=0` 关闭原始提示、问题、答案与思考文本落盘，避免预留评测题进入工程日志；普通模型/服务运行日志仍保留。

精确环境在 `scripts/admp-round2-runtime.py`。API、Ollama、前端日志/PID写 `work/round2/runtime/`；ES PID在 `work/runtime/elasticsearch.pid`。以 `ss -ltnp` 和 `/proc/<pid>/cmdline` 确认真实进程后再停止或重启，不根据旧 PID 文件直接终止未知任务。

健康验证：`curl http://127.0.0.1:8788/api/health`。原有 `scripts/admp` 的默认端口仍是8787/5173，本轮入口显式使用8788/5174。

本轮启动中 ES IK 插件曾因隔离配置缺少词典失败，已把同安装的 `config/analysis-ik` 复制到本工程 `work/runtime/es-config/analysis-ik`，然后重启当前9201实例使词典初始化完整。未改模型、依赖版本或用户 Docker 配置。
