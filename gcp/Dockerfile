ARG BASE_IMAGE_NAME
FROM ${BASE_IMAGE_NAME}

# Install GKE/GCP dependencies
RUN uv pip install google-cloud-pubsub google-cloud-secret-manager google-cloud-storage

# GKE mounts the network volume at /network-volume directly via a PersistentVolume,
# so no symlink needed — just ensure the path exists as a mount point.
RUN mkdir -p /network-volume

# Copy GCP worker scripts (Pub/Sub handler, used by Batch and GKE)
COPY handler.py /handler.py
COPY model_sync.py /model_sync.py

RUN chmod +x /start.sh

CMD ["/start.sh"]
