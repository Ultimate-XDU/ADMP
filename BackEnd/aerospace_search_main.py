import os
import re
import hashlib
import warnings
import concurrent.futures
from typing import List, Dict, Any

import pdfplumber
from docx import Document
from elasticsearch import Elasticsearch, helpers
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ==========================================
# 1. 集中配置中心
# ==========================================
class Config:
    # 路径配置
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DOC_DIR = os.path.join(BASE_DIR, "aerospace_docs")
    
    # Elasticsearch 配置
    ES_HOST = "http://localhost:9200"
    ES_INDEX = "aerospace_kw_db"
    
    # Milvus 配置
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    MILVUS_COLLECTION = "aerospace_m3e_db"
    
    # 切片与向量模型参数
    CHUNK_SIZE = 1024
    MIN_CHUNK_SIZE = 300
    EMBEDDING_MODEL = 'moka-ai/m3e-base'
    VECTOR_DIM = 768  # m3e-base 模型的固定维度


# ==========================================
# 2. 文档解析与处理模块
# ==========================================
class DocumentProcessor:
    @staticmethod
    def extract_text(file_path: str) -> str:
        """根据文件类型提取文本"""
        text_blocks = []
        try:
            if file_path.lower().endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text(x_tolerance=1, y_tolerance=1)
                        if t and len(t.strip()) > 50:
                            text_blocks.append(t)
            elif file_path.lower().endswith(".docx"):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    if para.text.strip():
                        text_blocks.append(para.text)
        except Exception as e:
            print(f"⚠️ 文件解析失败 {file_path}: {e}")
        return "\n\n".join(text_blocks)

    @staticmethod
    def clean_text(text: str) -> str:
        """清洗格式与无用信息"""
        filters = [
            r"Keywords:.*?$", r"Academic Editors:.*?$", r"Received:.*?$", r"Published:.*?$",
            r"\d+\s*\.$", r"^[A-Za-z\s]+:", r"Page\s*\d+", r"\[\d+\]", r"^\s*\d+\.?\s*$",
        ]
        for pattern in filters:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    @classmethod
    def chunk_document(cls, text: str) -> List[str]:
        """将长文档切分为符合大小限制的文本块"""
        text = cls.clean_text(text)
        if not text:
            return []
        if len(text) <= Config.CHUNK_SIZE:
            return [text]
        
        sentences = re.split(r'(?<=[。！？；：．.?!])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        raw_chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) <= Config.CHUNK_SIZE:
                current_chunk += " " + sent
            else:
                if len(current_chunk) >= Config.MIN_CHUNK_SIZE:
                    raw_chunks.append(current_chunk.strip())
                    current_chunk = sent
        if current_chunk:
            raw_chunks.append(current_chunk.strip())
            
        return raw_chunks


# ==========================================
# 3. 数据库管理模块
# ==========================================
class DatabaseManager:
    def __init__(self):
        self.es_client = Elasticsearch(Config.ES_HOST)
        # 连接 Milvus
        connections.connect("default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        
    def reset_databases(self) -> Collection:
        """初始化并重置两端数据库"""
        # 1. 重置 Milvus
        if utility.has_collection(Config.MILVUS_COLLECTION):
            utility.drop_collection(Config.MILVUS_COLLECTION)
            
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=Config.VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096)
        ]
        schema = CollectionSchema(fields, description="航空航天领域知识库 (m3e)")
        milvus_coll = Collection(Config.MILVUS_COLLECTION, schema)
        # 建立 IVF_FLAT 索引以加速检索
        milvus_coll.create_index(
            field_name="vector", 
            index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        )

        # 2. 重置 ES 并配置 IK 分词器
        if self.es_client.indices.exists(index=Config.ES_INDEX):
            self.es_client.indices.delete(index=Config.ES_INDEX)
            
        es_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    }
                }
            }
        }
        self.es_client.indices.create(index=Config.ES_INDEX, body=es_mapping)
        return milvus_coll

    def build_index(self):
        """执行全量数据入库"""
        print("\n[INFO] 正在扫描并解析文档库...")
        os.makedirs(Config.DOC_DIR, exist_ok=True)
        milvus_coll = self.reset_databases()
        
        all_chunks = []
        for file_name in os.listdir(Config.DOC_DIR):
            if file_name.lower().endswith(('.pdf', '.docx')):
                raw_text = DocumentProcessor.extract_text(os.path.join(Config.DOC_DIR, file_name))
                all_chunks.extend(DocumentProcessor.chunk_document(raw_text))

        if not all_chunks:
            print("[WARN] 未在目录下找到有效文档内容。")
            return

        print(f"[INFO] 成功切分 {len(all_chunks)} 个文本块，正在生成语义向量...")
        doc_ids = [hashlib.md5(chunk.encode('utf-8')).hexdigest() for chunk in all_chunks]
        embeddings = self.embedder.encode(all_chunks, normalize_embeddings=True).tolist()

        print("[INFO] 正在写入 Milvus 向量数据库...")
        # Milvus 批量插入格式为列式列表：[id列表, 向量列表, 内容列表]
        milvus_coll.insert([doc_ids, embeddings, all_chunks])
        milvus_coll.flush()
        
        print("[INFO] 正在写入 Elasticsearch 全文检索引擎...")
        es_actions = [
            {"_index": Config.ES_INDEX, "_id": d_id, "content": chunk}
            for d_id, chunk in zip(doc_ids, all_chunks)
        ]
        helpers.bulk(self.es_client, es_actions)
        self.es_client.indices.refresh(index=Config.ES_INDEX)
        print("✅ 双擎数据库构建完毕！")


# ==========================================
# 4. 混合检索引擎模块
# ==========================================
class SearchEngine:
    def __init__(self):
        self.es_client = Elasticsearch(Config.ES_HOST)
        connections.connect("default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        self.milvus_coll = Collection(Config.MILVUS_COLLECTION)
        self.milvus_coll.load() # 检索前必须将数据载入内存
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)

    def _vector_search(self, query_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
        """执行 Milvus 向量相似度检索"""
        res = self.milvus_coll.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["content"]
        )
        hits = []
        if res and len(res) > 0:
            for hit in res[0]:
                hits.append({
                    "chunk_id": hit.id,
                    "content": hit.entity.get("content")
                })
        return hits

    def _keyword_search(self, query_str: str, top_k: int) -> List[Dict[str, Any]]:
        """执行 ES 关键词倒排检索"""
        res = self.es_client.search(index=Config.ES_INDEX, query={"match": {"content": query_str}}, size=top_k)
        return [{"chunk_id": hit["_id"], "content": hit["_source"]["content"]} for hit in res["hits"]["hits"]]

    def _rrf_fusion(self, vector_hits: List[Dict], keyword_hits: List[Dict], k: int = 60) -> List[Dict]:
        """倒数排名融合算法"""
        rrf_scores = {}
        content_map = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit["chunk_id"]
            content_map[cid] = hit["content"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        for rank, hit in enumerate(keyword_hits):
            cid = hit["chunk_id"]
            content_map[cid] = hit["content"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_id": cid, "rrf_score": score, "content": content_map[cid]} for cid, score in sorted_docs]

    def execute_query(self, query: str, top_k: int = 3) -> None:
        """主搜索逻辑与格式化输出"""
        try:
            query_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_vec = executor.submit(self._vector_search, query_vec, 10)
                future_kw = executor.submit(self._keyword_search, query, 10)
                
                vector_hits = future_vec.result()
                keyword_hits = future_kw.result()

            fused_results = self._rrf_fusion(vector_hits, keyword_hits)[:top_k]
            
            if not fused_results:
                print("\n[未找到匹配内容]")
                return

            keywords = query.strip().split()
            print(f"\n{'='*60}\n🔍 检索关键字: {query}\n{'='*60}")
            
            for i, hit in enumerate(fused_results):
                content = hit['content']
                for kw in keywords:
                    # 在终端中使用 ANSI 转义码实现黄色高亮
                    content = re.sub(re.escape(kw), f"\033[93m【{kw}】\033[0m", content, flags=re.IGNORECASE)
                
                print(f"🏆 [综合排名 {i+1}] RRF 融合得分: {hit['rrf_score']:.4f}")
                print(f"📄 内容摘录:\n{content}\n{'-'*60}")
                
        except Exception as e:
            print(f"❌ 检索异常: {e}")


# ==========================================
# 5. 终端交互式入口
# ==========================================
def main():
    print("""
    ==================================================
       🚀 航空航天智能文档检索引擎 (终端版)
       组件状态: ES (关键词) + Milvus (语义)
    ==================================================
    """)
    
    choice = input("👉 是否需要重新扫描文档并构建数据库？(y/n) 默认 n: ").strip().lower()
    if choice == 'y':
        try:
            db_manager = DatabaseManager()
            db_manager.build_index()
        except Exception as e:
            print(f"\n❌ 构建失败，请检查 Docker (ES 或 Milvus) 是否启动: {e}")
            return
    
    try:
        engine = SearchEngine()
        print("\n✅ 检索引擎已就绪！可以开始提问。")
    except Exception as e:
        print(f"\n❌ 引擎初始化失败，请确认数据库已构建且正在运行: {e}")
        return

    while True:
        try:
            query = input("\n💬 请输入搜索问题 (输入 'exit' 退出) > ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                print("👋 退出系统，再见！")
                break
                
            engine.execute_query(query)
            
        except KeyboardInterrupt:
            print("\n👋 退出系统，再见！")
            break

if __name__ == "__main__":
    main()