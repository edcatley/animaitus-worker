import json
import os
import signal
import socket
import sys
import time
import threading
import traceback
import requests
from google.cloud import pubsub_v1, secretmanager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ["PUBSUB_PROJECT_ID"]
SUBSCRIPTION = os.environ["PUBSUB_SUBSCRIPTION"]
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())

def _get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Add root to path for local imports
sys.path.insert(0, "/")
from src.base_handler import handler
from model_sync import extract_required_models, ensure_models_on_disk

# Initialize Pub/Sub
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION)

# Global tracker for the job currently in progress
_current_wrapped_message = None
_current_job_id = None
_current_webhook_url = None

# ---------------------------------------------------------------------------
# Message Wrapper
# ---------------------------------------------------------------------------
class PubSubMessageWrapper:
    """
    Makes a Synchronous Pull message behave like a Streaming message.
    This allows us to use .ack() and .nack() inside our processing logic.
    """
    def __init__(self, received_msg):
        self.data = received_msg.message.data
        self.ack_id = received_msg.ack_id

    def ack(self):
        subscriber.acknowledge(
            request={"subscription": subscription_path, "ack_ids": [self.ack_id]}
        )

    def nack(self):
        # Setting deadline to 0 makes the message immediately available to others
        subscriber.modify_ack_deadline(
            request={
                "subscription": subscription_path, 
                "ack_ids": [self.ack_id], 
                "ack_deadline_seconds": 0
            }
        )

# ---------------------------------------------------------------------------
# Lease Extender
# ---------------------------------------------------------------------------
class LeaseExtender(threading.Thread):
    """Background thread to keep the Pub/Sub lease alive during long renders."""
    def __init__(self, wrapper):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.stop_event = threading.Event()

    def run(self):
        print(f"[gke-handler] Lease heartbeat started.")
        while not self.stop_event.is_set():
            self.stop_event.wait(45)  # extend every 45s with a 15s safety buffer
            if self.stop_event.is_set():
                break
            try:
                subscriber.modify_ack_deadline(
                    request={
                        "subscription": subscription_path,
                        "ack_ids": [self.wrapper.ack_id],
                        "ack_deadline_seconds": 60,
                    }
                )
            except Exception as e:
                print(f"[gke-handler] Heartbeat failed: {e}")

    def stop(self):
        self.stop_event.set()


# ---------------------------------------------------------------------------
# Webhook Helper
# ---------------------------------------------------------------------------
def _post_status(webhook_url: str, job_id: str, status: str, extra: dict = None):
    payload = {"jobId": job_id, "status": status, "workerId": WORKER_ID}
    if extra:
        payload.update(extra)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_secret('WORKER_WEBHOOK_SECRET')}"
    }

    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[gke-handler] WARNING: Webhook failed for {job_id}: {e}")

# ---------------------------------------------------------------------------
# SIGTERM (Spot Reclaim) Handling
# ---------------------------------------------------------------------------
def _handle_sigterm(signum, frame):
    """If Google takes the Blackwell back, return the job to the queue."""
    global _current_wrapped_message, _current_job_id, _current_webhook_url
    print("[gke-handler] SIGTERM received. Reclaiming node...")
    if _current_wrapped_message:
        print(f"[gke-handler] Nacking current job to ensure redelivery.")
        _current_wrapped_message.nack()
        if _current_job_id and _current_webhook_url:
            _post_status(_current_webhook_url, _current_job_id, "IN_QUEUE")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)

# ---------------------------------------------------------------------------
# The Processing Logic
# ---------------------------------------------------------------------------
def _process_job(wrapped_message):
    global _current_job_id, _current_webhook_url
    job_id = None
    webhook_url = None
    
    try:
        payload = json.loads(wrapped_message.data.decode("utf-8"))
        job_id = payload.get("jobId")
        webhook_url = payload.get("webhookUrl")
        job_input = payload.get("input")

        if not all([job_id, webhook_url, job_input]):
            print(f"[gke-handler] ERROR: Malformed message. Discarding.")
            wrapped_message.ack()
            return

        _current_job_id = job_id
        _current_webhook_url = webhook_url

        print(f"[gke-handler] Starting job {job_id}")
        _post_status(webhook_url, job_id, "IN_PROGRESS")

        heartbeat = LeaseExtender(wrapped_message)
        heartbeat.start()

        try:
            workflow = job_input.get("workflow", {})
            required_models = extract_required_models(workflow)
            sync_errors = ensure_models_on_disk(required_models)
            if sync_errors:
                error_msg = f"Failed to fetch required model(s): {sync_errors}"
                print(f"[gke-handler] ERROR: {error_msg}")
                _post_status(webhook_url, job_id, "FAILED", {"error": error_msg})
                wrapped_message.ack()
                return

            # Call the ComfyUI handler
            result = handler({"id": job_id, "input": job_input})

            if "error" in result:
                _post_status(webhook_url, job_id, "FAILED", {
                    "error": result["error"],
                    "details": result.get("details"),
                })
            else:
                _post_status(webhook_url, job_id, "COMPLETED", {"output": result})

            wrapped_message.ack()
            print(f"[gke-handler] Job {job_id} finished.")
        finally:
            heartbeat.stop()
            heartbeat.join()

    except Exception as e:
        print(f"[gke-handler] Unexpected error: {e}")
        wrapped_message.nack()
    finally:
        _current_job_id = None
        _current_webhook_url = None

# ---------------------------------------------------------------------------
# Main Loop (The Pull Engine)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[gke-handler] Blackwell Worker {WORKER_ID} online.")
    
    while True:
        try:
            # Synchronous pull 1 message
            response = subscriber.pull(
                request={"subscription": subscription_path, "max_messages": 1},
                timeout=20
            )

            # --- EXIT CONDITION ---
            if not response.received_messages:
                print("[gke-handler] Queue empty. Shutting down Blackwell.")
                break 

            for msg in response.received_messages:
                wrapped = PubSubMessageWrapper(msg)
                _current_wrapped_message = wrapped
                _process_job(wrapped)
                _current_wrapped_message = None

        except Exception as e:
            # Handle timeouts or network blips
            if "DeadlineExceeded" not in str(e):
                print(f"[gke-handler] Loop error: {e}")
            time.sleep(2)

    print("[gke-handler] Process exited naturally. Billing stopped.")
    sys.exit(0)