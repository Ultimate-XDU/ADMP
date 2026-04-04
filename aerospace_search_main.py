# aerospace_search_main.py 【最终纯净版】修复论文冗余文本
import os
import re
import gradio as gr
import numpy as np
import pdfplumber
from docx import Document
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# ===================== 配置 =====================
DOC_DIR = "./aerospace_docs"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "aerospace_cnn_db"

MAX_VOCAB = 15000
MAX_SEQ_LEN = 512
EMBED_DIM = 256
CHUNK_SIZE = 1024
L2_RATE = 1e-4
DROPOUT_RATE = 0.2
VECTOR_DIM = 256

SEMANTIC_SIM_THRESHOLD = 0.5
OVERLAP_TOKENS = 50
MIN_CHUNK_SIZE = 300

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")

# ===================== 文档解析 =====================
def extract_text(file_path):
    text = []
    try:
        if file_path.lower().endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if t and len(t.strip()) > 50:
                        text.append(t)
        elif file_path.lower().endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text.append(para.text)
    except Exception as e:
        print(f"解析错误 {file_path}: {e}")
    return "\n\n".join(text)

# ===================== 【修复】清洗论文冗余文本 =====================
def clean_text(text: str) -> str:
    filters = [
        r"Keywords:.*?$",
        r"Academic Editors:.*?$",
        r"Received:.*?$",
        r"Published:.*?$",
        r"\d+\s*\.$",
        r"^[A-Za-z\s]+:",
        r"Page\s*\d+",
        r"\[\d+\]",
        r"^\s*\d+\.?\s*$",
    ]
    for pattern in filters:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# ===================== 你的语义切片（仅新增清洗调用） =====================
def split_sentences(text: str) -> list:
    pattern = r'(?<=[。！？；：．.?!])\s+'
    sents = re.split(pattern, text)
    return [s.strip() for s in sents if s.strip()]

def chunk_text(text: str) -> list:
    # 修复点：过滤垃圾文本
    text = clean_text(text)
    
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    
    raw_chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) <= CHUNK_SIZE:
            current_chunk += " " + sent
        else:
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                raw_chunks.append(current_chunk.strip())
                current_chunk = sent
    if current_chunk:
        raw_chunks.append(current_chunk.strip())
    
    return raw_chunks

# ===================== TextCNN =====================
def build_cnn():
    inputs = Input(shape=(MAX_SEQ_LEN,))
    emb = Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_SEQ_LEN, embeddings_regularizer=l2(L2_RATE))(inputs)
    c1 = Conv1D(128, 3, activation='relu', kernel_regularizer=l2(L2_RATE))(emb)
    p1 = GlobalMaxPooling1D()(c1)
    c2 = Conv1D(64, 5, activation='relu', kernel_regularizer=l2(L2_RATE))(emb)
    p2 = GlobalMaxPooling1D()(c2)
    concat = Concatenate(axis=-1)([p1, p2])
    drop = Dropout(DROPOUT_RATE)(concat)
    dense = Dense(VECTOR_DIM, activation='relu', kernel_regularizer=l2(L2_RATE))(drop)
    out = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense)
    return Model(inputs, out)

def text2vec(text_list):
    if not text_list:
        return np.array([])
    tokenizer.fit_on_texts(text_list)
    seq = tokenizer.texts_to_sequences(text_list)
    pad_seq = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    model = build_cnn()
    vecs = model.predict(pad_seq, verbose=0)
    return vecs.astype(np.float32)

# ===================== Milvus =====================
def init_milvus():
    if connections.has_connection("default"):
        connections.disconnect("default")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096)
    ]
    schema = CollectionSchema(fields, "航空航天文档库")
    coll = Collection(COLLECTION_NAME, schema)
    coll.create_index("vector", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
    return coll

def build_db():
    try:
        os.makedirs(DOC_DIR, exist_ok=True)
        coll = init_milvus()
        all_chunks = []

        for f in os.listdir(DOC_DIR):
            if f.lower().endswith(('.pdf', '.docx')):
                raw = extract_text(os.path.join(DOC_DIR, f))
                all_chunks.extend(chunk_text(raw))

        if not all_chunks:
            return "❌ 错误：未读取到有效内容"

        vecs = text2vec(all_chunks)
        entities = [{"vector": vec.tolist(), "content": chunk} for vec, chunk in zip(vecs, all_chunks)]
        coll.insert(entities)
        coll.flush()
        coll.load()
        return f"✅ 构建成功！共导入 {len(all_chunks)} 段文本"
    except Exception as e:
        return f"❌ 构建错误：{str(e)}"

# ===================== 检索 =====================
def search(query):
    try:
        if not utility.has_collection(COLLECTION_NAME):
            return "请先构建数据库"
        coll = Collection(COLLECTION_NAME)
        coll.load()
        vec = text2vec([query])[0].tolist()
        res = coll.search(
            data=[vec],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=3,
            output_fields=["content"]
        )
        output = ""
        keywords = query.strip().split()
        for i, hit in enumerate(res[0]):
            content = hit.entity.get('content')
            for kw in keywords:
                content = re.sub(re.escape(kw), f"【{kw}】", content, flags=re.IGNORECASE)
            output += f"🔍 结果 {i+1} | 相似度：{round(hit.score*100,2)}%\n{content}\n{'='*80}\n\n"
        return output if output else "未找到匹配内容"
    except Exception as e:
        return f"检索异常：{str(e)}"

# ===================== 界面 =====================
with gr.Blocks(title="航空航天检索") as demo:
    gr.Markdown("# 航空航天文档检索系统")
    build_btn = gr.Button("构建数据库", variant="primary")
    status = gr.Textbox(label="状态")
    q = gr.Textbox(label="检索关键词")
    search_btn = gr.Button("开始检索")
    res_box = gr.Textbox(label="结果", lines=25)

    build_btn.click(build_db, outputs=status)
    search_btn.click(search, inputs=q, outputs=res_box)

if __name__ == "__main__":
    demo.launch(server_port=7860)