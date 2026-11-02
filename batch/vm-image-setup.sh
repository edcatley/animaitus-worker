#!/bin/bash
# Turns a fresh Batch Debian 12 VM into a GPU worker VM image.
#
# Runs as the startup script of a temporary builder VM (no GPU needed) created by
# the build-vm-image step in gcp/cloudbuild.yaml. Installs the NVIDIA driver and
# container toolkit, pre-pulls the worker container so Batch VMs start it
# instantly, then reports DONE/FAILED via guest attributes and powers off.
#
# Metadata in:  worker-image  — full image URI to pre-pull (e.g. ...:gke-<sha>)
# Guest attrs out: build/status, build/log (last 40 lines)

set -euo pipefail

LOG=/var/log/image-build.log
exec > >(tee -a "$LOG") 2>&1

METADATA=http://metadata.google.internal/computeMetadata/v1/instance

report() {
  curl -s -X PUT -H 'Metadata-Flavor: Google' --data "$1" "$METADATA/guest-attributes/build/status" || true
  tail -n 40 "$LOG" | curl -s -X PUT -H 'Metadata-Flavor: Google' --data-binary @- "$METADATA/guest-attributes/build/log" || true
}

WORKER_IMAGE="$(curl -s -H 'Metadata-Flavor: Google' "$METADATA/attributes/worker-image")"
DRIVER_BRANCH="580"  # Blackwell (RTX PRO 6000) needs 580+

trap 'echo "IMAGE-BUILD-FAILED (line $LINENO)"; report "FAILED line $LINENO"; sleep 60; poweroff' ERR

echo "== [1/5] Freezing the kernel so the driver module matches at runtime"
systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
apt-mark hold "linux-image-$(uname -r)" || true

echo "== [2/5] Installing NVIDIA driver ${DRIVER_BRANCH} (open kernel modules, required for Blackwell)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y "linux-headers-$(uname -r)" wget
wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y "nvidia-driver-pinning-${DRIVER_BRANCH}" || echo "(no pinning package, installing latest branch)"
apt-get install -y nvidia-open
echo "Built driver module version: $(modinfo -F version nvidia)"

echo "== [3/5] Installing Docker (if missing) and the NVIDIA container toolkit"
command -v docker >/dev/null || apt-get install -y docker.io
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "== [4/5] Pre-pulling ${WORKER_IMAGE}"
REGISTRY_HOST="${WORKER_IMAGE%%/*}"
TOKEN="$(curl -s -H 'Metadata-Flavor: Google' "$METADATA/service-accounts/default/token" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
echo "$TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"
docker pull "$WORKER_IMAGE"
docker logout "https://${REGISTRY_HOST}"

echo "== [5/5] Cleaning up"
apt-get clean
rm -f cuda-keyring_1.1-1_all.deb

echo "IMAGE-BUILD-DONE driver=$(modinfo -F version nvidia)"
report "DONE driver=$(modinfo -F version nvidia)"
sleep 60  # give the build step time to read the result before shutdown
poweroff
