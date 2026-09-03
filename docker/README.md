# ADMP local retrieval services

This Compose stack exposes only local ports required by the backend:

- Elasticsearch with the IK Chinese analyzer: `http://127.0.0.1:9200`
- Milvus gRPC: `127.0.0.1:19530`
- Milvus Web UI: `http://127.0.0.1:9091/webui/`

The version choices are intentional:

- Elasticsearch `8.4.1` is paired with the matching `analysis-ik` plugin version. The backend creates fields using `ik_max_word` and `ik_smart`, so a plain Elasticsearch image will fail to create the index.
- Milvus `2.4.23` follows Milvus's official v2.4 standalone Compose topology and supports the `pymilvus.Collection` API used by this project. Its required etcd and MinIO services are included.
- Elasticsearch runs with a fixed 1 GB heap and Java 17 supplied through `ES_JAVA_HOME`. This avoids the cgroup v2 startup defect in Elasticsearch 8.4.1's bundled JDK while preserving the exact Elasticsearch/IK plugin version pair.

## Start

```powershell
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:9200
```

The first run downloads images and builds the small Elasticsearch image that contains the IK plugin. Data is stored in Docker named volumes and survives `docker compose down`.

## Verify from the backend

With the backend running, open `http://127.0.0.1:8787/api/health`. A healthy response includes `"elasticsearch": true` and `"milvus": true`. The endpoint creates the initially empty Milvus collection and Elasticsearch index as needed.

## Stop and reset

```powershell
docker compose down
```

Do not run `docker compose down --volumes` unless you intend to permanently erase the Elasticsearch and Milvus indexes.
