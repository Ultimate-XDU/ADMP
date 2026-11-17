import json
import os
import socket
import shutil
import subprocess
import time
from pathlib import Path

root = Path('/home/yuexing/src/ADMP-graphrag-20261111')
run = root / 'work/round2/runtime'
run.mkdir(parents=True, exist_ok=True)
graph_data = root / 'work/round2/graph-data'
if not graph_data.exists():
    shutil.copytree(root / 'work/eval20/graph-data', graph_data)

def listening(port):
    with socket.socket() as client:
        client.settimeout(0.5)
        return client.connect_ex(('127.0.0.1', port)) == 0

started = {}
if not listening(11435):
    environment = dict(os.environ, OLLAMA_HOST='127.0.0.1:11435')
    with (run / 'ollama.log').open('ab') as log:
        process = subprocess.Popen(['/home/yuexing/.local/bin/ollama', 'serve'], cwd=root, env=environment, stdout=log, stderr=log, start_new_session=True)
    (run / 'ollama.pid').write_text(str(process.pid))
    started['ollama'] = process.pid

if not listening(9201):
    environment = dict(os.environ, ES_JAVA_HOME='/home/yuexing/.local/share/admp/jre', ES_PATH_CONF=str(root / 'work/runtime/es-config'), ES_JAVA_OPTS='-Xms512m -Xmx1g')
    with (run / 'elasticsearch-start.log').open('ab') as log:
        process = subprocess.Popen(['/home/yuexing/.local/share/admp/elasticsearch/bin/elasticsearch', '-p', str(root / 'work/runtime/elasticsearch.pid')], cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    started['elasticsearch'] = process.pid

for attempt in range(45):
    if listening(9201) and listening(11435):
        break
    time.sleep(1)
else:
    raise SystemExit('Existing Elasticsearch/Ollama did not become ready; inspect work/round2/runtime logs.')

if not listening(8788):
    environment = dict(os.environ,
        PYTHONPATH=str(root / 'BackEnd'),
        GRAPH_DATA_DIR=str(graph_data),
        GRAPH_CORPUS_DIR=str(root / 'work/corpus-100'),
        GRAPH_OLLAMA_HOST='http://127.0.0.1:11435',
        GRAPH_LLM_MODEL='qwen3.5:9b', GRAPH_AUTO_BUILD='0', GRAPH_EXTRACT_SCOPE='all',
        GRAPH_LOG_RAW_LLM='0', GRAPH_MAX_TOOL_CALLS='6',
        AEROSPACE_ES_HOST='http://127.0.0.1:9201',
        MILVUS_MODE='embedded',
        AEROSPACE_EMBEDDING_MODEL='/mnt/d/ADMP/models/BAAI--bge-m3',
        AEROSPACE_RERANK_MODEL='/mnt/d/ADMP/models/BAAI--bge-reranker-v2-m3',
        AEROSPACE_EMBEDDING_DEVICE='cpu', AEROSPACE_RERANK_DEVICE='cpu',
        QWEN_API_KEY='ollama', LLM_BASE_URL='http://127.0.0.1:11435/v1',
        LLM_MODEL='qwen3.5:9b', LLM_THINKING_ENABLED='false',
        AEROSPACE_CORS_ORIGINS='http://localhost:5174,http://127.0.0.1:5174',
        OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', TOKENIZERS_PARALLELISM='false')
    environment.pop('MILVUS_URI', None)
    with (run / 'api.log').open('ab') as log:
        process = subprocess.Popen([str(root / 'work/venv/bin/python'), '-m', 'uvicorn', 'api.aerospace_api:app', '--app-dir', str(root / 'BackEnd'), '--host', '127.0.0.1', '--port', '8788'], cwd=root, env=environment, stdout=log, stderr=log, start_new_session=True)
    (run / 'api.pid').write_text(str(process.pid))
    started['api'] = process.pid

if not listening(5174):
    environment = dict(os.environ, VITE_API_PROXY='http://127.0.0.1:8788')
    with (run / 'frontend.log').open('ab') as log:
        process = subprocess.Popen(['npm', 'run', 'dev', '--', '--host', '0.0.0.0', '--port', '5174'], cwd=root / 'FrontEnd', env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    (run / 'frontend.pid').write_text(str(process.pid))
    started['frontend'] = process.pid

print(json.dumps({'started': started, 'root': str(root), 'frontend': 'http://localhost:5174', 'api': 'http://127.0.0.1:8788', 'model': 'http://127.0.0.1:11435', 'elasticsearch': 'http://127.0.0.1:9201'}))
