import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import websocket
import uuid
import socket
import traceback
import shutil

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 1000
# Websocket reconnection behaviour (can be overridden through environment variables)
# NOTE: more attempts and diagnostics improve debuggability whenever ComfyUI crashes mid-job.
#   • WEBSOCKET_RECONNECT_ATTEMPTS sets how many times we will try to reconnect.
#   • WEBSOCKET_RECONNECT_DELAY_S sets the sleep in seconds between attempts.
#
# If the respective env-vars are not supplied we fall back to sensible defaults ("5" and "3").
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Extra verbose websocket trace logs (set WEBSOCKET_TRACE=true to enable)
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    # This prints low-level frame information to stdout which is invaluable for diagnosing
    # protocol errors but can be noisy in production – therefore gated behind an env-var.
    websocket.enableTrace(True)

# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"

# ---------------------------------------------------------------------------
# Helper: quick reachability probe of ComfyUI HTTP endpoint (port 8188)
# ---------------------------------------------------------------------------


def _comfy_server_status():
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {
            "reachable": resp.status_code == 200,
            "status_code": resp.status_code,
        }
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """
    Attempts to reconnect to the WebSocket server after a disconnect.

    Args:
        ws_url (str): The WebSocket URL (including client_id).
        max_attempts (int): Maximum number of reconnection attempts.
        delay_s (int): Delay in seconds between attempts.
        initial_error (Exception): The error that triggered the reconnect attempt.

    Returns:
        websocket.WebSocket: The newly connected WebSocket object.

    Raises:
        websocket.WebSocketConnectionClosedException: If reconnection fails after all attempts.
    """
    print(
        f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect..."
    )
    last_reconnect_error = initial_error
    for attempt in range(max_attempts):
        # Log current server status before each reconnect attempt so that we can
        # see whether ComfyUI is still alive (HTTP port 8188 responding) even if
        # the websocket dropped. This is extremely useful to differentiate
        # between a network glitch and an outright ComfyUI crash/OOM-kill.
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            # If ComfyUI itself is down there is no point in retrying the websocket –
            # bail out immediately so the caller gets a clear "ComfyUI crashed" error.
            print(
                f"worker-comfyui - ComfyUI HTTP unreachable – aborting websocket reconnect: {srv_status.get('error', 'status '+str(srv_status.get('status_code')))}"
            )
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )

        # Otherwise we proceed with reconnect attempts while server is up
        print(
            f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}... (ComfyUI HTTP reachable, status {srv_status.get('status_code')})"
        )
        try:
            # Need to create a new socket object for reconnect
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)  # Use existing ws_url
            print(f"worker-comfyui - Websocket reconnected successfully.")
            return new_ws  # Return the new connected socket
        except (
            websocket.WebSocketException,
            ConnectionRefusedError,
            socket.timeout,
            OSError,
        ) as reconn_err:
            last_reconnect_error = reconn_err
            print(
                f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}"
            )
            if attempt < max_attempts - 1:
                print(
                    f"worker-comfyui - Waiting {delay_s} seconds before next attempt..."
                )
                time.sleep(delay_s)
            else:
                print(f"worker-comfyui - Max reconnection attempts reached.")

    # If loop completes without returning, raise an exception
    print("worker-comfyui - Failed to reconnect websocket after connection closed.")
    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate 'workflow' in input
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    # Validate 'images' in input, if provided (base64 encoded)
    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return (
                None,
                "'images' must be a list of objects with 'name' and 'image' keys",
            )

    # Validate 'videos' in input, if provided (base64 encoded)
    videos = job_input.get("videos")
    if videos is not None:
        if not isinstance(videos, list) or not all(
            "name" in video and "video" in video for video in videos
        ):
            return (
                None,
                "'videos' must be a list of objects with 'name' and 'video' keys",
            )

    # Validate 'download_urls' in input, if provided (signed download URLs for images/videos)
    download_urls = job_input.get("download_urls")
    if download_urls is not None:
        if not isinstance(download_urls, list) or not all(
            "name" in item and "url" in item for item in download_urls
        ):
            return (
                None,
                "'download_urls' must be a list of objects with 'name' and 'url' keys",
            )

    # Optional: API key for Comfy.org API Nodes, passed per-request
    comfy_org_api_key = job_input.get("comfy_org_api_key")

    # Optional: Array of upload URL objects (consistent with image_urls/video_urls structure)
    upload_urls = job_input.get("upload_urls")
    if upload_urls is not None:
        if not isinstance(upload_urls, list) or not all(
            "name" in item and "url" in item for item in upload_urls
        ):
            return (
                None,
                "'upload_urls' must be a list of objects with 'name' and 'url' keys",
            )

    # Optional: Output filename
    output_filename = job_input.get("output_filename")

    # Return validated data and no error
    return {
        "workflow": workflow,
        "images": images,
        "videos": videos,
        "download_urls": download_urls,
        "comfy_org_api_key": comfy_org_api_key,
        "upload_urls": upload_urls,
        "output_filename": output_filename,
    }, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    print(f"worker-comfyui - Checking API server at {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"worker-comfyui - API is reachable")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException as e:
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.

    Returns:
        dict: A dictionary indicating success or error.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    for image in images:
        try:
            name = image["name"]
            image_data_uri = image["image"]  # Get the full string (might have prefix)

            # --- Strip Data URI prefix if present ---
            if "," in image_data_uri:
                # Find the comma and take everything after it
                base64_data = image_data_uri.split(",", 1)[1]
            else:
                # Assume it's already pure base64
                base64_data = image_data_uri
            # --- End strip ---

            blob = base64.b64decode(base64_data)  # Decode the cleaned data

            # Prepare the form data
            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            # POST request to upload the image
            response = requests.post(
                f"http://{COMFY_HOST}/upload/image", files=files, timeout=30
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            )
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print(f"worker-comfyui - image(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"worker-comfyui - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def upload_videos(videos):
    """
    Upload a list of base64 encoded videos to the ComfyUI server using the /upload/video endpoint.

    Args:
        videos (list): A list of dictionaries, each containing the 'name' of the video and the 'video' as a base64 encoded string.

    Returns:
        dict: A dictionary indicating success or error.
    """
    if not videos:
        return {"status": "success", "message": "No videos to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"worker-comfyui - Uploading {len(videos)} video(s)...")

    for video in videos:
        try:
            name = video["name"]
            video_data_uri = video["video"]  # Get the full string (might have prefix)

            # --- Strip Data URI prefix if present ---
            if "," in video_data_uri:
                base64_data = video_data_uri.split(",", 1)[1]
            else:
                base64_data = video_data_uri
            # --- End strip ---

            blob = base64.b64decode(base64_data)  # Decode the cleaned data

            # Prepare the form data
            files = {
                "video": (name, BytesIO(blob), "video/mp4"),
                "overwrite": (None, "true"),
            }

            # POST request to upload the video
            response = requests.post(
                f"http://{COMFY_HOST}/upload/video", files=files, timeout=60
            )
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except base64.binascii.Error as e:
            error_msg = f"Error decoding base64 for {video.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {video.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error uploading {video.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error uploading {video.get('name', 'unknown')}: {e}"
            )
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print(f"worker-comfyui - video(s) upload finished with errors")
        return {
            "status": "error",
            "message": "Some videos failed to upload",
            "details": upload_errors,
        }

    print(f"worker-comfyui - video(s) upload complete")
    return {
        "status": "success",
        "message": "All videos uploaded successfully",
        "details": responses,
    }


def download_and_upload_files(download_urls):
    """
    Download files from signed URLs and upload them to ComfyUI.
    File type (image/video) is auto-detected from filename extension.
    Supports subdirectories via path separators in the name field.

    Args:
        download_urls (list): A list of dictionaries, each containing 'name' and 'url' keys.
                             The 'name' can include subdirectories (e.g., "head/image.png")

    Returns:
        dict: A dictionary indicating success or error.
    """
    if not download_urls:
        return {"status": "success", "message": "No files to download", "details": []}

    responses = []
    download_errors = []

    print(f"worker-comfyui - Downloading and uploading {len(download_urls)} file(s) from URLs (parallel)...")

    def _download_and_upload_one(file_item):
        name = file_item["name"]
        url = file_item["url"]

        print(f"worker-comfyui - Downloading {name} from {url[:100]}...")

        # Download the file from the signed URL
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        file_bytes = response.content

        print(f"worker-comfyui - Downloaded {name} ({len(file_bytes)} bytes), uploading to ComfyUI...")

        # Parse subfolder and filename from name
        # e.g., "head/image.png" -> subfolder="head", filename="image.png"
        if "/" in name:
            parts = name.rsplit("/", 1)
            subfolder = parts[0]
            filename = parts[1]
        else:
            subfolder = ""
            filename = name

        # Auto-detect file type from extension
        name_lower = filename.lower()
        if name_lower.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
            endpoint = f"http://{COMFY_HOST}/upload/video"
            content_type = "video/mp4"
            form_field = "video"
        else:
            endpoint = f"http://{COMFY_HOST}/upload/image"
            content_type = "image/png"
            form_field = "image"

        files = {
            form_field: (filename, BytesIO(file_bytes), content_type),
            "overwrite": (None, "true"),
        }

        if subfolder:
            files["subfolder"] = (None, subfolder)
            print(f"worker-comfyui - Uploading to subfolder: {subfolder}")

        upload_response = requests.post(endpoint, files=files, timeout=60)
        upload_response.raise_for_status()

        print(f"worker-comfyui - Successfully uploaded {name} to ComfyUI")
        return f"Successfully downloaded and uploaded {name}"

    with ThreadPoolExecutor(max_workers=len(download_urls)) as executor:
        futures = {executor.submit(_download_and_upload_one, item): item for item in download_urls}
        for future in as_completed(futures):
            item = futures[future]
            try:
                responses.append(future.result())
            except requests.Timeout:
                error_msg = f"Timeout downloading or uploading {item.get('name', 'unknown')}"
                print(f"worker-comfyui - {error_msg}")
                download_errors.append(error_msg)
            except requests.RequestException as e:
                error_msg = f"Error downloading or uploading {item.get('name', 'unknown')}: {e}"
                print(f"worker-comfyui - {error_msg}")
                download_errors.append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error processing {item.get('name', 'unknown')}: {e}"
                print(f"worker-comfyui - {error_msg}")
                download_errors.append(error_msg)

    if download_errors:
        print(f"worker-comfyui - File download/upload finished with errors")
        return {
            "status": "error",
            "message": "Some files failed to download/upload",
            "details": download_errors,
        }

    print(f"worker-comfyui - File download/upload complete")
    return {
        "status": "success",
        "message": "All files downloaded and uploaded successfully",
        "details": responses,
    }


def get_available_models():
    """
    Get list of available models from ComfyUI

    Returns:
        dict: Dictionary containing available models by type
    """
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

        # Extract available checkpoints from CheckpointLoaderSimple
        available_models = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = (
                        ckpt_options[0] if isinstance(ckpt_options[0], list) else []
                    )

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed
        client_id (str): The client ID for the websocket connection
        comfy_org_api_key (str, optional): Comfy.org API key for API Nodes

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow

    Raises:
        ValueError: If the workflow validation fails with detailed error information
    """
    # Include client_id in the prompt payload
    payload = {"prompt": workflow, "client_id": client_id}

    # Optionally inject Comfy.org API key for API Nodes.
    # Precedence: per-request key (argument) overrides environment variable.
    # Note: We use our consistent naming (comfy_org_api_key) but transform to
    # ComfyUI's expected format (api_key_comfy_org) when sending.
    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}
    data = json.dumps(payload).encode("utf-8")

    # Use requests for consistency and timeout
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30
    )

    # Handle validation errors with detailed information
    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = response.json()
            print(f"worker-comfyui - Parsed error data: {error_data}")

            # Try to extract meaningful error information
            error_message = "Workflow validation failed"
            error_details = []

            # ComfyUI seems to return different error formats, let's handle them all
            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                    if error_info.get("type") == "prompt_outputs_failed_validation":
                        error_message = "Workflow validation failed"
                else:
                    error_message = str(error_info)

            # Check for node validation errors in the response
            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(
                                f"Node {node_id} ({error_type}): {error_msg}"
                            )
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            # Check if the error data itself contains validation info
            if error_data.get("type") == "prompt_outputs_failed_validation":
                error_message = error_data.get("message", "Workflow validation failed")
                # For this type of error, we need to parse the validation details from logs
                # Since ComfyUI doesn't seem to include detailed validation errors in the response
                # Let's provide a more helpful generic message
                available_models = get_available_models()
                if available_models.get("checkpoints"):
                    error_message += f"\n\nThis usually means a required model or parameter is not available."
                    error_message += f"\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                else:
                    error_message += "\n\nThis usually means a required model or parameter is not available."
                    error_message += "\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(error_message)

            # If we have specific validation errors, format them nicely
            if error_details:
                detailed_message = f"{error_message}:\n" + "\n".join(
                    f"• {detail}" for detail in error_details
                )

                # Try to provide helpful suggestions for common errors
                if any(
                    "not in list" in detail and "ckpt_name" in detail
                    for detail in error_details
                ):
                    available_models = get_available_models()
                    if available_models.get("checkpoints"):
                        detailed_message += f"\n\nAvailable checkpoint models: {', '.join(available_models['checkpoints'])}"
                    else:
                        detailed_message += "\n\nNo checkpoint models appear to be available. Please check your model installation."

                raise ValueError(detailed_message)
            else:
                # Fallback to the raw response if we can't parse specific errors
                raise ValueError(f"{error_message}. Raw response: {response.text}")

        except (json.JSONDecodeError, KeyError) as e:
            # If we can't parse the error response, fall back to the raw text
            raise ValueError(
                f"ComfyUI validation failed (could not parse error response): {response.text}"
            )

    # For other HTTP errors, raise them normally
    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    # Use requests for consistency and timeout
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def cleanup_comfyui_directories():
    """
    Clean up ComfyUI input/output/temp directories to prevent file collisions
    between jobs when workers are reused.
    
    This removes all files from:
    - /ComfyUI/input/ (uploaded input files) - EXCEPT /ComfyUI/input/demo/
    - /ComfyUI/output/ (generated output files)  
    - /ComfyUI/temp/ (temporary files)
    
    Models in /ComfyUI/models/ are NOT touched to keep them loaded in RAM.
    Demo data in /ComfyUI/input/demo/ is preserved for testing.
    """
    print("worker-comfyui - Starting cleanup of ComfyUI directories...")
    
    directories_to_clean = [
        "/comfyui/input",
        "/comfyui/output",
        "/comfyui/temp"
    ]
    
    # Directories to preserve (don't delete)
    preserve_dirs = ["/comfyui/input/demo"]
    
    cleaned_count = 0
    
    for directory in directories_to_clean:
        print(f"worker-comfyui - Checking directory: {directory}")
        
        if not os.path.exists(directory):
            print(f"worker-comfyui - Directory does not exist: {directory}")
            continue
        
        try:
            items = os.listdir(directory)
            print(f"worker-comfyui - Found {len(items)} item(s) in {directory}: {items}")
            
            # Remove all files in the directory
            for item in items:
                item_path = os.path.join(directory, item)
                
                # Skip preserved directories
                if item_path in preserve_dirs:
                    print(f"worker-comfyui - Skipping preserved directory: {item_path}")
                    continue
                
                try:
                    if os.path.isfile(item_path):
                        print(f"worker-comfyui - Deleting file: {item_path}")
                        os.remove(item_path)
                        cleaned_count += 1
                    elif os.path.isdir(item_path):
                        print(f"worker-comfyui - Deleting directory: {item_path}")
                        shutil.rmtree(item_path)
                        cleaned_count += 1
                except Exception as e:
                    print(f"worker-comfyui - ERROR: Could not remove {item_path}: {e}")
                    print(f"worker-comfyui - Exception type: {type(e).__name__}")
                    traceback.print_exc()
        except Exception as e:
            print(f"worker-comfyui - ERROR: Could not access {directory}: {e}")
            print(f"worker-comfyui - Exception type: {type(e).__name__}")
            traceback.print_exc()
    
    print(f"worker-comfyui - Cleanup complete. Cleaned up {cleaned_count} item(s) from previous job")


def get_image_data(filename, subfolder, image_type):
    """
    Fetch image bytes from the ComfyUI /view endpoint.

    Args:
        filename (str): The filename of the image.
        subfolder (str): The subfolder where the image is stored.
        image_type (str): The type of the image (e.g., 'output').

    Returns:
        bytes: The raw image data, or None if an error occurs.
    """
    print(
        f"worker-comfyui - Fetching image data: type={image_type}, subfolder={subfolder}, filename={filename}"
    )
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    try:
        # Use requests for consistency and timeout
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=60)
        response.raise_for_status()
        print(f"worker-comfyui - Successfully fetched image data for {filename}")
        return response.content
    except requests.Timeout:
        print(f"worker-comfyui - Timeout fetching image data for {filename}")
        return None
    except requests.RequestException as e:
        print(f"worker-comfyui - Error fetching image data for {filename}: {e}")
        return None
    except Exception as e:
        print(
            f"worker-comfyui - Unexpected error fetching image data for {filename}: {e}"
        )
        return None

def handler(job):
    """
    Handles a job using ComfyUI via websockets for status and image retrieval.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]
    job_id = job["id"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    workflow = validated_data["workflow"]
    print(f"worker-comfyui - DEBUG: Workflow JSON: {json.dumps(workflow)}")
    input_images = validated_data.get("images")
    input_videos = validated_data.get("videos")
    download_urls = validated_data.get("download_urls")

    # Clean up ComfyUI directories from previous job to prevent file collisions
    cleanup_comfyui_directories()

    # Make sure that the ComfyUI HTTP API is available before proceeding
    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {
            "error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."
        }

    # Upload base64 encoded images if they exist
    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            return {
                "error": "Failed to upload one or more input images",
                "details": upload_result["details"],
            }

    # Upload base64 encoded videos if they exist
    if input_videos:
        upload_result = upload_videos(input_videos)
        if upload_result["status"] == "error":
            return {
                "error": "Failed to upload one or more input videos",
                "details": upload_result["details"],
            }

    # Download and upload files from URLs if they exist (images/videos auto-detected)
    if download_urls:
        download_result = download_and_upload_files(download_urls)
        if download_result["status"] == "error":
            return {
                "error": "Failed to download/upload one or more files from URLs",
                "details": download_result["details"],
            }

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    errors = []

    try:
        # Establish WebSocket connection
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-comfyui - Websocket connected")

        # Queue the workflow
        try:
            # Pass per-request API key if provided in input
            queued_workflow = queue_workflow(
                workflow,
                client_id,
                comfy_org_api_key=validated_data.get("comfy_org_api_key"),
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(
                    f"Missing 'prompt_id' in queue response: {queued_workflow}"
                )
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except requests.RequestException as e:
            print(f"worker-comfyui - Error queuing workflow: {e}")
            raise ValueError(f"Error queuing workflow: {e}")
        except Exception as e:
            print(f"worker-comfyui - Unexpected error queuing workflow: {e}")
            # For ValueError exceptions from queue_workflow, pass through the original message
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Unexpected error queuing workflow: {e}")

        # Wait for execution completion via WebSocket
        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...")
        execution_done = False
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        status_data = message.get("data", {}).get("status", {})
                        print(
                            f"worker-comfyui - Status update: {status_data.get('exec_info', {}).get('queue_remaining', 'N/A')} items remaining in queue"
                        )
                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("node") is None
                            and data.get("prompt_id") == prompt_id
                        ):
                            print(
                                f"worker-comfyui - Execution finished for prompt {prompt_id}"
                            )
                            execution_done = True
                            break
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, Message: {data.get('exception_message')}"
                            print(
                                f"worker-comfyui - Execution error received: {error_details}"
                            )
                            errors.append(f"Workflow execution error: {error_details}")
                            break
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                # Timeout is normal during long-running operations, don't spam logs
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    # Attempt to reconnect
                    ws = _attempt_websocket_reconnect(
                        ws_url,
                        WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S,
                        closed_err,
                    )

                    print(
                        "worker-comfyui - Resuming message listening after successful reconnect."
                    )
                    continue
                except (
                    websocket.WebSocketConnectionClosedException
                ) as reconn_failed_err:
                    # If _attempt_websocket_reconnect fails, it raises this exception
                    # Let this exception propagate to the outer handler's except block
                    raise reconn_failed_err

            except json.JSONDecodeError:
                print(f"worker-comfyui - Received invalid JSON message via websocket.")

        if not execution_done and not errors:
            raise ValueError(
                "Workflow monitoring loop exited without confirmation of completion or error."
            )

        # Fetch history even if there were execution errors, some outputs might exist
        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history after execution."
            print(f"worker-comfyui - {error_msg}")
            if not errors:
                return {"error": error_msg}
            else:
                errors.append(error_msg)
                return {
                    "error": "Job processing failed, prompt ID not found in history.",
                    "details": errors,
                }

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        if not outputs:
            warning_msg = f"No outputs found in history for prompt {prompt_id}."
            print(f"worker-comfyui - {warning_msg}")
            if not errors:
                errors.append(warning_msg)

        # --------------------------------------------------------------------------
        # ------------------- MODIFIED OUTPUT PROCESSING BLOCK -------------------
        # --------------------------------------------------------------------------
        print(f"worker-comfyui - Processing {len(outputs)} output nodes...")

        # Build upload URL lookup map from array structure
        upload_url_map = {}
        if validated_data.get("upload_urls"):
            upload_url_map = {
                item["name"]: item["url"] 
                for item in validated_data["upload_urls"]
            }
            print(f"worker-comfyui - Built upload URL map with {len(upload_url_map)} entries")

        expected_uploads = set(upload_url_map.keys())
        completed_uploads = set()

        # Collect all output files first
        output_files = []
        for node_id, node_output in outputs.items():
            for key in ["videos", "gifs", "images"]:
                if key in node_output:
                    print(f"worker-comfyui - Node {node_id} contains {len(node_output[key])} file(s) in output key '{key}'")
                    for file_info in node_output[key]:
                        filename = file_info.get("filename")
                        if filename and file_info.get("type") != "temp":
                            output_files.append((filename, file_info.get("subfolder", ""), file_info.get("type")))

        def _fetch_and_upload(file_tuple):
            original_filename, subfolder, file_type = file_tuple
            file_bytes = get_image_data(original_filename, subfolder, file_type)
            if not file_bytes:
                return {"error": f"Failed to fetch file data for {original_filename} from /view endpoint."}

            upload_url = upload_url_map.get(original_filename)
            if upload_url:
                content_type = "application/octet-stream"
                if original_filename.endswith(('.mp4', '.mov', '.avi')):
                    content_type = "video/mp4"
                elif original_filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    content_type = "image/png"
                elif original_filename.endswith('.gif'):
                    content_type = "image/gif"

                print(f"worker-comfyui - Uploading {original_filename} to signed URL...")
                resp = requests.put(upload_url, data=file_bytes, headers={"Content-Type": content_type}, timeout=60)
                resp.raise_for_status()
                print(f"worker-comfyui - Successfully uploaded {original_filename} to signed URL")
                return {"filename": original_filename, "type": "uploaded", "url": upload_url}
            else:
                base64_file = base64.b64encode(file_bytes).decode("utf-8")
                return {"filename": original_filename, "type": "base64", "data": base64_file}

        if output_files:
            with ThreadPoolExecutor(max_workers=len(output_files)) as executor:
                futures = {executor.submit(_fetch_and_upload, f): f for f in output_files}
                for future in as_completed(futures):
                    original_filename = futures[future][0]
                    try:
                        result = future.result()
                        if "error" in result:
                            errors.append(result["error"])
                        else:
                            output_data.append(result)
                            if result["type"] == "uploaded":
                                completed_uploads.add(original_filename)
                    except Exception as e:
                        error_msg = f"Error processing {original_filename}: {e}"
                        print(f"worker-comfyui - {error_msg}")
                        errors.append(error_msg)

        # --------------------------------------------------------------------------
        # ----------------- END OF MODIFIED OUTPUT PROCESSING BLOCK ----------------
        # --------------------------------------------------------------------------

        # Check that every expected signed-URL upload was actually completed.
        # This is a hard failure regardless of other outputs — return immediately
        # so RunPod marks the job as failed rather than completed.
        if expected_uploads:
            missing_uploads = expected_uploads - completed_uploads
            if missing_uploads:
                missing_list = ", ".join(sorted(missing_uploads))
                error_msg = f"Silent upload failure: the following files had signed URLs but were never uploaded: {missing_list}"
                print(f"worker-comfyui - {error_msg}")
                return {"error": error_msg}
    except websocket.WebSocketException as e:
        print(f"worker-comfyui - WebSocket Error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}
    except requests.RequestException as e:
        print(f"worker-comfyui - HTTP Request Error: {e}")
        print(traceback.format_exc())
        return {"error": f"HTTP communication error with ComfyUI: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - Value Error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    except Exception as e:
        print(f"worker-comfyui - Unexpected Handler Error: {e}")
        print(traceback.format_exc())
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if ws and ws.connected:
            print(f"worker-comfyui - Closing websocket connection.")
            ws.close()

    final_result = {}

    if output_data:
        final_result["images"] = output_data

    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - Job completed with errors/warnings: {errors}")

    if not output_data and errors:
        print(f"worker-comfyui - Job failed with no output images.")
        return {
            "error": "Job processing failed",
            "details": errors,
        }
    elif not output_data and not errors:
        print(
            f"worker-comfyui - Job completed successfully, but the workflow produced no images."
        )
        final_result["status"] = "success_no_images"
        final_result["images"] = []

    print(f"worker-comfyui - Job completed. Returning {len(output_data)} image(s).")
    return final_result


# This is a generic handler module meant to be imported by platform-specific wrappers
# See runpod/handler.py for an example
