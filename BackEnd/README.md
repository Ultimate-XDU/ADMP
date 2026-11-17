# ADMP 后端

FastAPI 产品入口为 `api.aerospace_api:app`，其中接入文档管理、混合检索、问答和图谱路由。`api.graph_api:app` 是独立图谱接口入口，不代替完整产品 API。

## 当前 Ubuntu 部署

从项目根目录运行 `./scripts/admp-round2`。API 默认使用本轮隔离端口 8788，模型为本机 Ollama 的 Qwen3.5:9b；关键词索引使用 Elasticsearch，向量库使用本工作目录的 Milvus Lite。脚本复用已有安装，不启动 Docker。

路径、端口与模型配置见 [运行说明](../docs/round2-runtime.md)。原有 `./scripts/admp` 仍保留，默认端口与本轮入口不同。

## 模块

| 目录 | 职责 |
| --- | --- |
| `api/` | 产品 HTTP 接口、文档与会话管理、图谱接口 |
| `search/` | 多格式解析、结构记录、索引、混合召回和重排序 |
| `knowledge/graph_rag/` | 文档来源身份、结构层次、语义图谱、图扩展、证据窗口与 Agent |
| `qa/` | 证据组织、模型回答、引用与图文增强 |
| `utils/` | 结构记录、状态存储和检索契约 |
| `scripts/` | 导入、索引与检索输出工具 |
| `tests/` | 真实实现的回归检查 |

`single` 使用单轮混合检索，`graph` 使用确定性图流程，`agent` 使用模型选择检索动作。工具轨迹中的步数不等同于成功完成的多跳推理。来源全文、命中窗口与实际生成上下文分别记录，引用需要同时核对来源和具体陈述。

结构图与语义图的覆盖范围不同。新入库文档可登记完整结构与片段；实体和关系抽取需要显式构建或开启自动构建，不能据此宣称全部文档已有完整语义图谱。详见 [图谱说明](../docs/graph-rag.md)。

## 测试

```bash
PYTHONPATH=BackEnd work/venv/bin/python -m pytest -q BackEnd/tests
```

依赖在根目录 `requirements.txt`。模型、索引、原始文档、会话数据库及评测标签不包含在仓库中。重新建立索引会影响本地数据，已有部署无需因 Git 远端变化而重建索引。
