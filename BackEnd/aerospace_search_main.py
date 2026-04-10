# aerospace_search_main.py —— 最终完整版（混合检索+重排序+智能上下文扩展）
import os
import re
import hashlib
import warnings
import concurrent.futures
from typing import List, Dict, Any, Tuple

import numpy as np
import fitz  # PyMuPDF — PDF 文本和表格提取
from docx import Document
from elasticsearch import Elasticsearch, helpers
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================
DOC_DIR = r"D:\Innovat\BackEnd\aerospace_docs"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "aerospace_m3e_db"
ES_HOST = "http://localhost:9200"
ES_INDEX = "aerospace_kw_db"

MAX_CHUNK_SIZE = 1024
EMBEDDING_MODEL = 'moka-ai/m3e-base'
VECTOR_DIM = 768

TOP_K = 5
QA_TOP_K = 3
VEC_RECALL = 150
KW_RECALL = 150
RRF_K = 60
RERANK_MODEL = 'BAAI/bge-reranker-base'

# ==========================================
# 2. 文档解析与清洗
# ==========================================
def extract_text_and_tables(file_path: str) -> Tuple[str, List[Dict]]:
    """提取文档的正文和表格（PDF 使用 PyMuPDF，DOCX 使用 python-docx）"""
    text_blocks = []
    tables = []
    try:
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc, 1):
                # 提取文本
                text = page.get_text("text")
                if text and len(text.strip()) > 50:
                    text_blocks.append(text)
                
                # 提取表格（PyMuPDF 4.x+ 内置表格检测）
                try:
                    page_tables = page.find_tables()
                    for tbl_idx, tbl in enumerate(page_tables.tables):
                        data = tbl.extract()
                        if data and len(data) >= 2:
                            header = data[0]
                            rows = data[1:]
                            md = ['| ' + ' | '.join(str(c) if c else '' for c in header) + ' |']
                            md.append('|' + '|'.join([' --- '] * len(header)) + '|')
                            for row in rows:
                                md.append('| ' + ' | '.join(str(c) if c else '' for c in row) + ' |')
                            markdown = '\n'.join(md)
                            # 尝试从表格上方文本中提取题注
                            title = _find_table_caption(page, tbl.bbox, page_num)
                            tables.append({'title': title, 'markdown': markdown, 'page': page_num})
                except Exception:
                    pass  # 旧版 PyMuPDF 可能不支持 find_tables，跳过表格
            doc.close()
        
        elif file_path.lower().endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text_blocks.append(para.text)
            for table in doc.tables:
                md = []
                for i, row in enumerate(table.rows):
                    row_text = [cell.text.strip() for cell in row.cells]
                    if i == 0:
                        md.append('| ' + ' | '.join(row_text) + ' |')
                        md.append('|' + '|'.join([' --- '] * len(row_text)) + '|')
                    else:
                        md.append('| ' + ' | '.join(row_text) + ' |')
                if md:
                    tables.append({'title': f"表格_{len(tables)+1}", 'markdown': '\n'.join(md), 'page': 0})
    except Exception as e:
        print(f"解析错误 {file_path}: {e}")
    
    full_text = "\n\n".join(text_blocks)
    # 清除中文字符之间被 PDF 排版插入的多余空格
    full_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', full_text)
    return full_text, tables

def _find_table_caption(page, bbox, page_num) -> str:
    """从表格上方区域提取题注文本（如 '表 3-1 发动机参数'）"""
    try:
        # bbox = (x0, y0, x1, y1)，取表格上方 80pt 区域
        clip_rect = fitz.Rect(bbox[0], max(0, bbox[1] - 80), bbox[2], bbox[1])
        text_above = page.get_text("text", clip=clip_rect).strip()
        if text_above:
            match = re.search(r'表\s*[\d]+[-–][\d]+[^\n]*', text_above)
            if match:
                return match.group().strip()
        # 尝试表格下方
        clip_rect = fitz.Rect(bbox[0], bbox[3], bbox[2], bbox[3] + 50)
        text_below = page.get_text("text", clip=clip_rect).strip()
        if text_below:
            match = re.search(r'表\s*[\d]+[-–][\d]+[^\n]*', text_below)
            if match:
                return match.group().strip()
    except Exception:
        pass
    return f"表_{page_num}_{int(bbox[0])}_{int(bbox[1])}"

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    patterns = [
        r'Keywords:.*$', r'Academic Editors:.*$', r'Received:.*$', r'Published:.*$',
        r'Page\s*\d+', r'\[\d+\]', r'^\s*\d+\.?\s*$', r'^[A-Za-z\s]+:$'
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.MULTILINE | re.IGNORECASE)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 保留含有编号格式的短行（如 (a)、(1)、（一）等）
        if stripped and not re.search(r'[\u4e00-\u9fff]', stripped) and len(stripped) < 10:
            if re.match(r'^[\(\（][a-z\d\u4e00-\u9fff]+[\)\）]', stripped):
                cleaned.append(line)
            else:
                continue
        else:
            cleaned.append(line)
    text = '\n'.join(cleaned)
    return text.strip()

# ==========================================
# 3. 规则识别与混合分块
# ==========================================
def detect_heading(line: str) -> Tuple[bool, int, str]:
    s = line.strip()
    if not s:
        return False, 0, ''
    # Level 1: 章
    if re.match(r'^第[零一二三四五六七八九十百千万\d]+章\s*', s) or re.match(r'^[A-Z]\s*章\s*', s):
        return True, 1, s
    # Level 1: 附录
    if re.match(r'^附录\s*[A-Za-z\d]', s):
        return True, 1, s
    # Level 2: 节
    if re.match(r'^第[零一二三四五六七八九十百千万\d]+节\s*', s):
        return True, 2, s
    # Level 2: 条（如 第33.39条）— 必须匹配完整条号格式，避免误判普通小数
    if re.match(r'^第[\d\.]+条\s*', s) or re.match(r'^第[零一二三四五六七八九十百千万]+条\s*', s):
        return True, 2, s
    # Level 2: 多级编号标题（如 1.2.3 xxx），要求编号后跟中文或至少2个字符的文字
    if re.match(r'^\d+\.\d+(\.\d+)*\s+[\u4e00-\u9fffA-Za-z].{1,}', s):
        return True, 2, s
    # Level 2: 图/表标题
    if re.match(r'^图\s*\d+[-–]\d+', s) or re.match(r'^表\s*\d+[-–]\d+', s):
        return True, 2, s
    # Level 3: 字母子条款 (a)、(b)...
    if re.match(r'^\([a-z]\)\s+', s):
        return True, 3, s
    # Level 4: 数字子条款 (1)、(2)...
    if re.match(r'^\(\d+\)\s+', s):
        return True, 4, s
    # Level 5: 罗马数字/嵌套子条款 (i)、(ii)...
    if re.match(r'^\((?:i{1,3}|iv|v|vi{0,3})\)\s+', s):
        return True, 5, s
    return False, 0, ''

def rule_based_split(text: str) -> List[Dict]:
    lines = text.split('\n')
    blocks = []
    cur_lines = []
    for line in lines:
        is_head, level, head_text = detect_heading(line)
        if is_head:
            if cur_lines:
                blocks.append({
                    'content': '\n'.join(cur_lines).strip(),
                    'is_heading': False,
                    'heading_level': 0,
                    'heading_text': ''
                })
                cur_lines = []
            blocks.append({
                'content': line.strip(),
                'is_heading': True,
                'heading_level': level,
                'heading_text': line.strip()
            })
        else:
            cur_lines.append(line)
    if cur_lines:
        blocks.append({
            'content': '\n'.join(cur_lines).strip(),
            'is_heading': False,
            'heading_level': 0,
            'heading_text': ''
        })
    return blocks

def semantic_split(block_text: str, embedder, max_len=MAX_CHUNK_SIZE) -> List[str]:
    if len(block_text) <= max_len:
        return [block_text]
    sentences = re.split(r'(?<=[。！？；：．.?!])\s*', block_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) <= 1:
        return [block_text]
    from sentence_transformers import util
    embs = embedder.encode(sentences, convert_to_tensor=True)
    embs_np = embs.cpu().numpy() if hasattr(embs, 'cpu') else np.array(embs)
    adj_sims = [util.cos_sim(embs[i], embs[i+1]).item() for i in range(len(sentences)-1)]
    def dynamic_thresh(window, base=0.7):
        if len(window) < 3:
            return base
        mean = np.mean(window)
        std = np.std(window)
        return np.clip(mean - 0.3*std, 0.55, 0.85)
    chunks = []
    cur = []
    cur_len = 0
    for i, sent in enumerate(sentences):
        sent_len = len(sent)
        if not cur:
            cur.append(sent)
            cur_len = sent_len
        else:
            idx_start = i - len(cur)
            idx_end = i
            center=torch.mean(embs[idx_start:idx_end], dim=0).cpu().numpy()
            center_tensor = torch.from_numpy(center).to(embs.device)
            sim = util.cos_sim(embs[i], center_tensor).item()
            window = adj_sims[max(0,i-5):min(len(adj_sims), i+5)]
            thresh = dynamic_thresh(window)
            if sim < thresh or cur_len + sent_len + 1 > max_len:
                chunks.append(' '.join(cur))
                cur = [sent]
                cur_len = sent_len
            else:
                cur.append(sent)
                cur_len += sent_len + 1
    if cur:
        chunks.append(' '.join(cur))
    return chunks

def hybrid_chunking(full_text: str, embedder) -> List[Dict]:
    initial = rule_based_split(full_text)
    # 传播标题文本：非标题块继承最近的父标题文本
    current_heading = ''
    for blk in initial:
        if blk['is_heading']:
            current_heading = blk['heading_text']
        else:
            if not blk.get('heading_text'):
                blk['heading_text'] = current_heading
    final = []
    for blk in initial:
        if blk['is_heading'] or len(blk['content']) <= MAX_CHUNK_SIZE:
            final.append(blk)
        else:
            subs = semantic_split(blk['content'], embedder, MAX_CHUNK_SIZE)
            for sub in subs:
                final.append({
                    'content': sub,
                    'is_heading': False,
                    'heading_level': 0,
                    'heading_text': blk.get('heading_text', '')
                })
    # 过滤极短的非标题块（无检索价值的碎片）
    final = [b for b in final if b['is_heading'] or len(b['content'].strip()) >= 20]
    return final

# ==========================================
# 4. 拉链表构建
# ==========================================
def build_chain(blocks: List[Dict]) -> List[Dict]:
    """按文档分组建链，避免跨文档连接"""
    from collections import defaultdict
    doc_groups = defaultdict(list)
    for blk in blocks:
        doc_groups[blk.get('doc_name', '')].append(blk)
    
    result = []
    for doc_name, doc_blocks in doc_groups.items():
        for i, blk in enumerate(doc_blocks):
            blk['chunk_id'] = hashlib.md5(f"{doc_name}_{i}_{blk['content'][:100]}".encode()).hexdigest()[:16]
        for i, blk in enumerate(doc_blocks):
            blk['prev_id'] = doc_blocks[i-1]['chunk_id'] if i > 0 else None
            blk['next_id'] = doc_blocks[i+1]['chunk_id'] if i < len(doc_blocks)-1 else None
            blk['parent_id'] = None
        stack = []
        for blk in doc_blocks:
            if blk.get('is_heading', False):
                level = blk.get('heading_level', 0)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                if stack:
                    blk['parent_id'] = stack[-1][1]
                stack.append((level, blk['chunk_id']))
            else:
                if stack:
                    blk['parent_id'] = stack[-1][1]
        result.extend(doc_blocks)
    return result

def save_metadata_to_es(es_client: Elasticsearch, blocks: List[Dict]):
    """将块的元数据写入 ES（与关键词检索共用同一个索引）"""
    actions = []
    for i, blk in enumerate(blocks):
        actions.append({
            "_index": ES_INDEX,
            "_id": blk['chunk_id'],
            "_source": {
                "chunk_id": blk['chunk_id'],
                "content": blk['content'],
                "prev_id": blk.get('prev_id', ''),
                "next_id": blk.get('next_id', ''),
                "parent_id": blk.get('parent_id', ''),
                "level": blk['heading_level'],
                "heading_text": blk.get('heading_text', ''),
                "is_heading": 1 if blk['is_heading'] else 0,
                "doc_order": i,
                "doc_name": blk.get('doc_name', '')
            }
        })
    helpers.bulk(es_client, actions)
    es_client.indices.refresh(index=ES_INDEX)

def _es_get_by_id(es_client: Elasticsearch, chunk_id: str) -> Dict:
    """从 ES 中按 chunk_id 获取一个文档"""
    try:
        res = es_client.get(index=ES_INDEX, id=chunk_id)
        return res['_source']
    except Exception:
        return {}

def get_chunk_context(es_client: Elasticsearch, chunk_id: str, context_size=1) -> Dict:
    current = _es_get_by_id(es_client, chunk_id)
    if not current:
        return {}
    prev_blocks = []
    p_id = current.get('prev_id', '')
    while p_id and len(prev_blocks) < context_size:
        row = _es_get_by_id(es_client, p_id)
        if row:
            prev_blocks.insert(0, row)
            p_id = row.get('prev_id', '')
        else:
            break
    next_blocks = []
    n_id = current.get('next_id', '')
    while n_id and len(next_blocks) < context_size:
        row = _es_get_by_id(es_client, n_id)
        if row:
            next_blocks.append(row)
            n_id = row.get('next_id', '')
        else:
            break
    parent = None
    if current.get('parent_id', ''):
        parent = _es_get_by_id(es_client, current['parent_id'])
    return {
        'current': current,
        'prev': prev_blocks,
        'next': next_blocks,
        'parent': parent
    }

def format_structural_context(ctx: Dict) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.get('parent'):
        parts.append(f"【所属章节】{ctx['parent'].get('content', '')}")
    for p in ctx.get('prev', []):
        parts.append(f"【上文】{p.get('content', '')}")
    parts.append(f"【当前】{ctx['current'].get('content', '')}")
    for n in ctx.get('next', []):
        parts.append(f"【下文】{n.get('content', '')}")
    return "\n\n".join(parts)

# ==========================================
# 5. 模型单例
# ==========================================
_embedder = None
_reranker = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("加载嵌入模型...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            _embedder = _embedder.to('cuda')
            print("嵌入模型已加载到 GPU")
        else:
            print("嵌入模型使用 CPU")
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            print("加载重排序模型...")
            _reranker = CrossEncoder(RERANK_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"重排序模型加载失败: {e}，将跳过重排序")
            _reranker = None
    return _reranker

# ==========================================
# 6. 双擎数据库管理
# ==========================================
class DualDBManager:
    def __init__(self):
        if connections.has_connection("default"):
            connections.disconnect("default")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.es_client = Elasticsearch(ES_HOST)
        self.embedder = get_embedder()

    def _es_mapping(self):
        return {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "prev_id": {"type": "keyword"},
                    "next_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "level": {"type": "integer"},
                    "heading_text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                    "is_heading": {"type": "integer"},
                    "doc_order": {"type": "integer"},
                    "doc_name": {"type": "keyword"}
                }
            }
        }

    def ensure_databases(self) -> Collection:
        """确保 Milvus collection 和 ES index 存在（不存在则创建，已存在则跳过）"""
        if not utility.has_collection(MILVUS_COLLECTION):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096)
            ]
            schema = CollectionSchema(fields, "航空航天语义库")
            coll = Collection(MILVUS_COLLECTION, schema)
            coll.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        else:
            coll = Collection(MILVUS_COLLECTION)
        coll.load()
        if not self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        return coll

    def reset_databases(self) -> Collection:
        if utility.has_collection(MILVUS_COLLECTION):
            utility.drop_collection(MILVUS_COLLECTION)
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096)
        ]
        schema = CollectionSchema(fields, "航空航天语义库")
        coll = Collection(MILVUS_COLLECTION, schema)
        coll.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
        if self.es_client.indices.exists(index=ES_INDEX):
            self.es_client.indices.delete(index=ES_INDEX)
        self.es_client.indices.create(index=ES_INDEX, body=self._es_mapping())
        return coll

    def _remove_doc_from_es(self, doc_name: str):
        """从 ES 中删除指定文档的所有块"""
        self.es_client.delete_by_query(
            index=ES_INDEX,
            body={"query": {"term": {"doc_name": doc_name}}},
            refresh=True
        )

    def _remove_doc_from_milvus(self, coll: Collection, doc_name: str):
        """从 Milvus 中删除指定文档的所有块（通过 ES 查出 chunk_id 列表）"""
        try:
            res = self.es_client.search(
                index=ES_INDEX,
                body={"query": {"term": {"doc_name": doc_name}}, "_source": ["chunk_id"], "size": 10000}
            )
            ids = [hit['_source']['chunk_id'] for hit in res['hits']['hits']]
            if ids:
                expr = f'chunk_id in {ids}'
                coll.delete(expr)
        except Exception:
            pass

    def add_document(self, file_path: str) -> str:
        """增量索引单个文档，若文档已存在则先删除旧数据再重新索引"""
        file_name = os.path.basename(file_path)
        if not os.path.exists(file_path):
            return f"文件不存在: {file_path}"

        try:
            milvus_coll = self.ensure_databases()
        except Exception as e:
            return f"数据库连接失败: {e}"

        # 先清除该文档的旧数据（支持重复提交同一文档）
        self._remove_doc_from_milvus(milvus_coll, file_name)
        self._remove_doc_from_es(file_name)

        print(f"增量索引: {file_name}")
        text, tables = extract_text_and_tables(file_path)
        doc_blocks: List[Dict] = []
        for tbl in tables:
            content = f"{tbl['title']}\n\n{tbl['markdown']}"
            doc_blocks.append({'content': content, 'is_heading': True, 'heading_level': 2,
                               'heading_text': tbl['title'], 'doc_name': file_name})
        if text:
            cleaned = clean_text(text)
            if cleaned and cleaned.strip():
                blocks = hybrid_chunking(cleaned, self.embedder)
                for blk in blocks:
                    blk['doc_name'] = file_name
                doc_blocks.extend(blocks)

        if not doc_blocks:
            return f"文档 {file_name} 未提取到有效内容"

        # 单文档链构建
        doc_blocks = build_chain(doc_blocks)

        contents = [b['content'] for b in doc_blocks]
        chunk_ids = [b['chunk_id'] for b in doc_blocks]
        print(f"生成向量（{len(doc_blocks)} 块）...")
        embeddings = self.embedder.encode(contents, normalize_embeddings=True, show_progress_bar=True)
        print("写入 Milvus...")
        milvus_coll.insert([chunk_ids, embeddings.tolist(), contents])
        milvus_coll.flush()
        milvus_coll.load()
        print("写入 ES...")
        save_metadata_to_es(self.es_client, doc_blocks)
        return f"✅ 增量索引成功！文档 {file_name} 共 {len(doc_blocks)} 个语义块"

    def remove_document(self, doc_name: str) -> str:
        """从索引中删除指定文档"""
        try:
            milvus_coll = self.ensure_databases()
        except Exception as e:
            return f"数据库连接失败: {e}"
        self._remove_doc_from_milvus(milvus_coll, doc_name)
        self._remove_doc_from_es(doc_name)
        return f"✅ 已从索引中删除文档: {doc_name}"

    def build_index(self) -> str:
        if not os.path.exists(DOC_DIR):
            return f"目录不存在: {DOC_DIR}"
        files = [f for f in os.listdir(DOC_DIR) if f.lower().endswith(('.pdf', '.docx'))]
        if not files:
            return f"目录下无文档文件: {DOC_DIR}"
        print(f"发现 {len(files)} 个文档")
        try:
            milvus_coll = self.reset_databases()
        except Exception as e:
            return f"数据库连接失败: {e}"
        all_blocks = []
        for f in files:
            path = os.path.join(DOC_DIR, f)
            print(f"解析: {f}")
            text, tables = extract_text_and_tables(path)
            for tbl in tables:
                content = f"{tbl['title']}\n\n{tbl['markdown']}"
                all_blocks.append({'content': content, 'is_heading': True, 'heading_level': 2,
                                   'heading_text': tbl['title'], 'doc_name': f})
            if text:
                cleaned = clean_text(text)
                if cleaned and cleaned.strip():
                    blocks = hybrid_chunking(cleaned, self.embedder)
                    for blk in blocks:
                        blk['doc_name'] = f
                    all_blocks.extend(blocks)
        if not all_blocks:
            return "未提取到任何有效内容"
        print(f"总块数: {len(all_blocks)}")
        all_blocks = build_chain(all_blocks)
        contents = [b['content'] for b in all_blocks]
        chunk_ids = [b['chunk_id'] for b in all_blocks]
        print("生成向量...")
        embeddings = self.embedder.encode(contents, normalize_embeddings=True, show_progress_bar=True)
        print("写入 Milvus...")
        milvus_coll.insert([chunk_ids, embeddings.tolist(), contents])
        milvus_coll.flush()
        milvus_coll.load()
        print("写入 ES（含元数据）...")
        save_metadata_to_es(self.es_client, all_blocks)
        return f"✅ 构建成功！共 {len(all_blocks)} 个语义块"

# ==========================================
# 7. 混合检索 + 智能上下文
# ==========================================
class HybridSearch:
    def __init__(self):
        if connections.has_connection("default"):
            connections.disconnect("default")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.es_client = Elasticsearch(ES_HOST)
        self.milvus_coll = Collection(MILVUS_COLLECTION)
        self.milvus_coll.load()
        self.embedder = get_embedder()
        self.reranker = get_reranker()
    
    def _vector_search(self, query_vec, top_k=VEC_RECALL):
        res = self.milvus_coll.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["chunk_id", "content"]
        )
        hits = []
        if res and res[0]:
            for hit in res[0]:
                hits.append({"chunk_id": hit.id, "content": hit.entity.get("content"), "score": hit.score})
        return hits
    
    def _keyword_search(self, query, top_k=KW_RECALL):
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query, "boost": 1}}},
                        {"match_phrase": {"content": {"query": query, "boost": 3}}}
                    ]
                }
            },
            "size": top_k
        }
        try:
            res = self.es_client.search(index=ES_INDEX, body=body)
            return [{"chunk_id": hit["_source"]["chunk_id"], "content": hit["_source"]["content"]} 
                    for hit in res["hits"]["hits"]]
        except:
            return []
    
    def _rrf_fusion(self, vec_hits, kw_hits, k=RRF_K):
        scores = {}
        content_map = {}
        for rank, h in enumerate(vec_hits):
            cid = h["chunk_id"]
            content_map[cid] = h["content"]
            scores[cid] = scores.get(cid, 0.0) + 1.0/(k+rank+1)
        for rank, h in enumerate(kw_hits):
            cid = h["chunk_id"]
            content_map[cid] = h["content"]
            scores[cid] = scores.get(cid, 0.0) + 1.0/(k+rank+1)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_id": cid, "rrf_score": score, "content": content_map[cid]} for cid, score in sorted_items]
    
    def _semantic_context(self, chunk_id: str, query_vec, top_k=3):
        res = self.milvus_coll.query(expr=f"chunk_id == '{chunk_id}'", output_fields=["vector"])
        if not res:
            return []
        chunk_vec = res[0]['vector']
        sim_res = self.milvus_coll.search(
            data=[chunk_vec],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=top_k+1,
            output_fields=["chunk_id", "content"]
        )
        similar = []
        for hit in sim_res[0]:
            if hit.id != chunk_id:
                similar.append({"chunk_id": hit.id, "content": hit.entity.get("content"), "score": hit.score})
        return similar

    def _resolve_reference(self, ref_str: str) -> str:
        """
        根据引用字符串（如 '(b)'、'第33.39条'）从 ES 中查询对应的条款内容。
        返回文本片段，如果未找到则返回空字符串。
        """
        try:
            # 处理像 '(b)'、'(c)' 这样的子条款引用
            match_sub = re.match(r'^\(([a-z\d]+)\)$', ref_str)
            if match_sub:
                sub = match_sub.group(1)
                body = {"query": {"match_phrase": {"content": f"({sub})"}}, "size": 1}
                res = self.es_client.search(index=ES_INDEX, body=body)
                if res['hits']['hits']:
                    return res['hits']['hits'][0]['_source']['content']
            
            # 处理像 '第33.39条' 这样的条款引用
            match_article = re.match(r'^第[\d\.]+条', ref_str)
            if match_article:
                article = match_article.group(0)
                body = {
                    "query": {
                        "bool": {
                            "should": [
                                {"match_phrase": {"heading_text": article}},
                                {"match_phrase": {"content": article}}
                            ]
                        }
                    },
                    "size": 1
                }
                res = self.es_client.search(index=ES_INDEX, body=body)
                if res['hits']['hits']:
                    return res['hits']['hits'][0]['_source']['content']
        except Exception:
            pass
        return ""

    def _get_children_content(self, chunk_id: str, heading_level: int, max_children: int = 20) -> List[str]:
        """沿 next 链向下遍历，收集当前标题下所有子块，
           直到遇到同级或更高级别的标题为止。"""
        current = _es_get_by_id(self.es_client, chunk_id)
        if not current:
            return []
        
        children = []
        n_id = current.get('next_id', '')
        while n_id and len(children) < max_children:
            row = _es_get_by_id(self.es_client, n_id)
            if not row:
                break
            if row.get('is_heading') and row.get('level', 0) <= heading_level:
                break
            children.append(row.get('content', ''))
            n_id = row.get('next_id', '')
        
        return children

    def search(self, query: str, structural_depth=1, semantic_extra=2) -> str:
        try:
            q_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                vec_fut = ex.submit(self._vector_search, q_vec, VEC_RECALL)
                kw_fut = ex.submit(self._keyword_search, query, KW_RECALL)
                vec_hits = vec_fut.result()
                kw_hits = kw_fut.result()
            
            fused = self._rrf_fusion(vec_hits, kw_hits)
            if not fused:
                return "未找到匹配内容。"
            
            # 重排序
            if self.reranker:
                pairs = [(query, hit['content']) for hit in fused[:50]]
                scores = self.reranker.predict(pairs, batch_size=16)
                for i, hit in enumerate(fused[:50]):
                    hit['rerank_score'] = float(scores[i])
                fused.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            output = ""
            for i, hit in enumerate(fused[:TOP_K]):
                # 1. 结构上下文
                struct_ctx = get_chunk_context(self.es_client, hit['chunk_id'], context_size=structural_depth)
                content_with_ctx = format_structural_context(struct_ctx)
                
                # 1.5 标题命中时，展开其下所有子内容
                chunk_meta = struct_ctx.get('current')
                if chunk_meta and chunk_meta.get('is_heading'):
                    children = self._get_children_content(
                        hit['chunk_id'], chunk_meta.get('level', 0)
                    )
                    if children:
                        content_with_ctx += "\n\n【条款完整内容】\n" + "\n".join(children)
                
                # 1.6 文档来源
                doc_source = ""
                if chunk_meta and chunk_meta.get('doc_name'):
                    doc_source = f"\n📄 来源文档: {chunk_meta['doc_name']}"
                
                # 2. 引用解释（自动补充被引用的条款）
                ref_section = ""
                # 匹配形如 (a)、(b)、(c) 或 （a） 等模式
                ref_pattern = r'[（(][a-z\d]+[）)]'
                refs = re.findall(ref_pattern, content_with_ctx)
                if refs:
                    explanations = []
                    for ref in set(refs):
                        ref_clean = ref.strip('（）()')
                        ref_full = f"({ref_clean})"
                        ref_content = self._resolve_reference(ref_full)
                        if ref_content:
                            preview = ref_content[:200].replace('\n', ' ')
                            explanations.append(f"• {ref_full}：{preview}...")
                    if explanations:
                        ref_section = "\n\n【引用解释】\n" + "\n".join(explanations)
                
                # 3. 语义相关块
                semantic_section = ""
                if semantic_extra > 0:
                    semantic_blocks = self._semantic_context(hit['chunk_id'], q_vec, top_k=semantic_extra)
                    if semantic_blocks:
                        semantic_section = "\n\n【语义相关】\n"
                        for sb in semantic_blocks:
                            semantic_section += f"- {sb['content'][:200]}...\n"
                
                # 最终组合
                final_content = content_with_ctx + ref_section + semantic_section
                
                # 高亮查询词
                for kw in query.split():
                    final_content = re.sub(re.escape(kw), f"【{kw}】", final_content, flags=re.IGNORECASE)
                
                score = hit.get('rerank_score', hit['rrf_score'])
                output += f"🔍 结果 {i+1} | 相关度: {score:.4f}{doc_source}\n{final_content}\n{'='*80}\n\n"
            
            return output
        except Exception as e:
            return f"检索异常: {str(e)}"

    def search_for_qa(self, query: str, structural_depth=1, semantic_extra=0) -> List[Dict]:
        """供 QA 模块调用，返回精简的结构化检索结果（TOP_K=QA_TOP_K，无语义/引用扩展）"""
        try:
            q_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                vec_fut = ex.submit(self._vector_search, q_vec, VEC_RECALL)
                kw_fut = ex.submit(self._keyword_search, query, KW_RECALL)
                vec_hits = vec_fut.result()
                kw_hits = kw_fut.result()

            fused = self._rrf_fusion(vec_hits, kw_hits)
            if not fused:
                return []

            if self.reranker:
                pairs = [(query, hit['content']) for hit in fused[:50]]
                scores = self.reranker.predict(pairs, batch_size=16)
                for i, hit in enumerate(fused[:50]):
                    hit['rerank_score'] = float(scores[i])
                fused.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

            results = []
            for hit in fused[:QA_TOP_K]:
                struct_ctx = get_chunk_context(self.es_client, hit['chunk_id'], context_size=structural_depth)
                chunk_meta = struct_ctx.get('current', {})

                content = chunk_meta.get('content', '')
                # 标题命中时展开子内容
                if chunk_meta.get('is_heading'):
                    children = self._get_children_content(
                        hit['chunk_id'], chunk_meta.get('level', 0)
                    )
                    if children:
                        content = content + "\n" + "\n".join(children)

                # 截断过长内容，避免 prompt 膨胀
                if len(content) > 800:
                    content = content[:800] + "..."

                parent = struct_ctx.get('parent')
                results.append({
                    'content': content,
                    'doc_name': chunk_meta.get('doc_name', ''),
                    'heading': chunk_meta.get('heading_text', ''),
                    'parent_heading': parent.get('content', '') if parent else '',
                    'score': hit.get('rerank_score', hit.get('rrf_score', 0)),
                    'chunk_id': hit['chunk_id'],
                })
            return results
        except Exception:
            return []

# ==========================================
# 8. 终端交互界面
# ==========================================
def main():
    print("=" * 60)
    print("🚀 航空航天智能文档检索系统（终端模式）")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("  1. 构建数据库（首次使用或文档更新后执行）")
        print("  2. 检索查询")
        print("  3. 退出")
        choice = input("\n输入选项 (1/2/3): ").strip()
        
        if choice == '1':
            print("\n正在构建数据库，请稍候...")
            mgr = DualDBManager()
            result = mgr.build_index()
            print(f"\n{result}")
        
        elif choice == '2':
            query = input("\n🔍 请输入检索问题: ").strip()
            if not query:
                print("查询不能为空。")
                continue
            depth = input("结构上下文深度 (0-3, 默认1): ").strip()
            depth = int(depth) if depth.isdigit() and 0 <= int(depth) <= 3 else 1
            extra = input("语义扩展块数量 (0-5, 默认2): ").strip()
            extra = int(extra) if extra.isdigit() and 0 <= int(extra) <= 5 else 2
            
            print("\n正在检索...")
            engine = HybridSearch()
            result = engine.search(query, depth, extra)
            print(f"\n{result}")
        
        elif choice == '3':
            print("再见！")
            break
        
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()