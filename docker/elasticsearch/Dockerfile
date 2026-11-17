ARG ELASTIC_VERSION=8.4.1
FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTIC_VERSION}

# Elasticsearch 8.4.1's bundled JDK cannot run on cgroup v2 hosts. Supply
# Java 17 through ES_JAVA_HOME while retaining the ES/IK version pair.
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV ES_JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
USER 1000

# BackEnd/search/index_manager.py configures ik_max_word and ik_smart analyzers.
# The IK plugin must use the exact same Elasticsearch version.
ARG ELASTIC_VERSION
RUN elasticsearch-plugin install --batch https://get.infini.cloud/elasticsearch/analysis-ik/${ELASTIC_VERSION}
