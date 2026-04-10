# aerospace_qa.py —— 航空航天智能问答模块（RAG + 通义千问）
import os
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from aerospace_search_main import HybridSearch

# ==========================================
# 1. 配置（从 .env 文件加载，密钥不写在代码里）
# ==========================================
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"

# ==========================================
# 2. Prompt 工程
# ==========================================
SYSTEM_PROMPT = """你是一位资深航空航天适航审定专家，熟悉 CCAR-25、CCAR-33、CCAR-35 等中国民航适航规章及相关咨询通告。

你的职责是基于检索到的文档片段，为用户提供准确、专业的航空技术解答。

## 回答规范

1. **严格基于证据**：只使用下方【参考文档】中的内容作答，不得编造或推测未出现的条款内容。
2. **不要在回答中标注引用来源**：回答中不要包含任何类似 [来源: ...] 的引用标记，系统会自动在回答下方展示参考来源。
3. **结构化输出**：
   - 先给出直接结论
   - 再展开具体条款依据
   - 如涉及多个条款，逐条列出
4. **专业术语**：使用航空领域标准术语，必要时附注英文原文。
5. **诚实声明**：如果参考文档中没有足够信息回答问题，明确说明"当前检索到的文档中未找到相关内容"，不要臆造答案。
6. **故障排查格式**（当问题涉及故障/问题排查时）：
   - 故障现象
   - 可能原因（依据条款）
   - 处置建议（依据条款）
   - 相关适航要求引用"""

def build_context(search_results: List[Dict]) -> str:
    """将检索结果组装为带编号和来源标注的参考文档"""
    if not search_results:
        return "【参考文档】\n无相关检索结果。"

    parts = ["【参考文档】\n"]
    for i, r in enumerate(search_results, 1):
        source = r.get('doc_name', '未知文档')
        heading = r.get('heading', '')
        parent = r.get('parent_heading', '')
        score = r.get('score', 0)

        # 来源标签
        source_label = source
        if parent:
            source_label += f" > {parent}"
        if heading:
            source_label += f" > {heading}"

        parts.append(f"--- 片段 {i} [来源: {source_label}] (相关度: {score:.4f}) ---")
        parts.append(r.get('content', ''))
        parts.append("")

    return "\n".join(parts)

def build_user_prompt(query: str, context: str) -> str:
    """组装用户 prompt"""
    return f"""{context}

---

**用户问题**：{query}

请严格依据上述参考文档作答。如果文档内容不足以回答，请如实说明。不要在回答中标注引用来源。"""

# ==========================================
# 3. LLM 调用
# ==========================================
def call_qwen(system_prompt: str, user_prompt: str) -> str:
    """调用通义千问 API"""
    if not QWEN_API_KEY:
        return "错误：未设置 QWEN_API_KEY 环境变量。请运行: set QWEN_API_KEY=你的密钥"

    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # 低温度，减少幻觉
        max_tokens=2048,
    )
    return response.choices[0].message.content

# ==========================================
# 4. 问答主流程
# ==========================================
def ask(query: str, engine: HybridSearch = None) -> str:
    """完整的 RAG 问答流程：检索 → 构建 prompt → 调用 LLM → 返回回答"""
    # 1. 检索相关文档片段
    print("正在检索相关文档...")
    if engine is None:
        engine = HybridSearch()
    results = engine.search_for_qa(query)

    if not results:
        return "未检索到相关文档内容，无法回答该问题。"

    # 2. 构建 prompt
    context = build_context(results)
    user_prompt = build_user_prompt(query, context)

    # 3. 调用大模型
    print("正在生成回答...")
    answer = call_qwen(SYSTEM_PROMPT, user_prompt)

    # 4. 拼接最终输出（回答 + 引用来源清单）
    sources = []
    for i, r in enumerate(results, 1):
        src = r.get('doc_name', '未知')
        heading = r.get('heading', '')
        label = f"{src}"
        if heading:
            label += f" > {heading}"
        sources.append(f"  [{i}] {label} (相关度: {r.get('score', 0):.4f})")

    output = f"""{'='*60}
📋 问题：{query}
{'='*60}

{answer}

{'─'*60}
📚 参考来源：
{chr(10).join(sources)}
{'='*60}"""

    return output


def ask_structured(query: str, engine: HybridSearch = None) -> Dict:
    """供 API 调用，返回结构化 dict: {answer, sources: [{content, doc_name, heading, parent_heading, score, chunk_id}]}"""
    if engine is None:
        engine = HybridSearch()
    results = engine.search_for_qa(query)

    if not results:
        return {"answer": "未检索到相关文档内容，无法回答该问题。", "sources": []}

    context = build_context(results)
    user_prompt = build_user_prompt(query, context)
    answer = call_qwen(SYSTEM_PROMPT, user_prompt)

    return {"answer": answer, "sources": results}

# ==========================================
# 5. 终端交互
# ==========================================
def main():
    global QWEN_API_KEY
    print("=" * 60)
    print("🛩️  航空航天智能问答系统（RAG + 通义千问）")
    print("=" * 60)

    if not QWEN_API_KEY:
        print("\n⚠️  未检测到 QWEN_API_KEY 环境变量")
        print("请先设置: set QWEN_API_KEY=你的API密钥")
        key = input("\n或直接输入 API Key: ").strip()
        if key:
            os.environ["QWEN_API_KEY"] = key
            QWEN_API_KEY = key
        else:
            print("未提供 API Key，退出。")
            return

    while True:
        print("\n请选择操作:")
        print("  1. 智能问答（检索 + 大模型生成）")
        print("  2. 仅检索（不调用大模型）")
        print("  3. 退出")
        choice = input("\n输入选项 (1/2/3): ").strip()

        if choice == '1':
            query = input("\n❓ 请输入问题: ").strip()
            if not query:
                print("问题不能为空。")
                continue
            result = ask(query)
            print(f"\n{result}")

        elif choice == '2':
            query = input("\n🔍 请输入检索问题: ").strip()
            if not query:
                print("问题不能为空。")
                continue
            engine = HybridSearch()
            result = engine.search(query)
            print(f"\n{result}")

        elif choice == '3':
            print("再见！")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()
