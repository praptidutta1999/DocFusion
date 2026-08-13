"""
DocFusion AI - Text-to-Image client

Local-side helper used by app.py.
The actual image diffusion inference runs in the separate
Colab FastAPI backend.
"""

import os
import tempfile
import time

import requests


# =========================================================
# IMAGE COLAB API
# =========================================================

IMAGE_COLAB_API = os.getenv(
    "DOCFUSION_IMAGE_COLAB_API",
    "https://recolor-outshine-chest.ngrok-free.dev"
).rstrip("/")


# =========================================================
# IMAGE MODEL REGISTRY
# =========================================================

IMAGE_MODEL_NAMES = {
    "sd15": "Stable Diffusion v1.5",
    "sdxl": "Stable Diffusion XL 1.0",
}


# =========================================================
# IMAGE GENERATION
# =========================================================

def generate_image(text, model="sd15"):
    """
    Send document text to the image-generation Colab backend
    and save the returned PNG locally.

    Returns:
        tuple[str, float]:
            local image filepath,
            backend processing time
    """

    # -----------------------------------------------------
    # Validate text
    # -----------------------------------------------------

    if not text or not text.strip():
        raise ValueError(
            "No text was provided for image generation."
        )

    # -----------------------------------------------------
    # Validate API URL
    # -----------------------------------------------------

    if (
        not IMAGE_COLAB_API
        or "PASTE_IMAGE_NGROK_URL_HERE" in IMAGE_COLAB_API
    ):
        raise RuntimeError(
            "Image Colab API URL is not configured. "
            "Set DOCFUSION_IMAGE_COLAB_API to your image backend "
            "ngrok URL."
        )

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    model = (
        model
        if model in IMAGE_MODEL_NAMES
        else "sd15"
    )

    model_display = IMAGE_MODEL_NAMES[model]

    # -----------------------------------------------------
    # Request information
    # -----------------------------------------------------

    endpoint = f"{IMAGE_COLAB_API}/image"

    print()
    print("=" * 60)
    print("DOCFUSION AI - IMAGE GENERATION")
    print("=" * 60)
    print(f"Backend : {IMAGE_COLAB_API}")
    print(f"Endpoint: {endpoint}")
    print(f"Model   : {model_display}")
    print(f"Text    : {len(text)} characters")
    print("=" * 60)

    start = time.perf_counter()

    # -----------------------------------------------------
    # Send request to Colab
    # -----------------------------------------------------

    try:

        response = requests.post(
            endpoint,
            json={
                "text": text,
                "model": model,
            },
            timeout=900,
        )

    except requests.exceptions.Timeout as e:

        elapsed = time.perf_counter() - start

        raise RuntimeError(
            f"Image backend request timed out after "
            f"{elapsed:.2f} seconds. "
            f"The Colab model may still be loading or generating."
        ) from e

    except requests.exceptions.ConnectionError as e:

        raise RuntimeError(
            "Could not connect to the Image Colab backend. "
            f"Check that the ngrok tunnel is running.\n"
            f"Backend URL: {IMAGE_COLAB_API}"
        ) from e

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"Image backend request failed: {str(e)}"
        ) from e

    # -----------------------------------------------------
    # Backend response status
    # -----------------------------------------------------

    if not response.ok:

        status_code = response.status_code

        try:
            error_data = response.json()

            message = error_data.get("error")

            if not message:
                message = error_data

        except ValueError:

            message = response.text.strip()

        if not message:
            message = (
                "The backend returned an empty error response."
            )

        elapsed = time.perf_counter() - start

        print()
        print("IMAGE BACKEND ERROR")
        print(f"HTTP status : {status_code}")
        print(f"Message     : {message}")
        print(f"Time        : {elapsed:.2f}s")
        print()

        raise RuntimeError(
            f"Image backend error "
            f"(HTTP {status_code}): {message}"
        )

    # -----------------------------------------------------
    # Validate content type
    # -----------------------------------------------------

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    if "image/" not in content_type:

        try:

            data = response.json()

            message = data.get(
                "error",
                "The image backend did not return an image."
            )

        except ValueError:

            message = response.text.strip()

            if not message:
                message = (
                    "The image backend returned an empty response."
                )

        raise RuntimeError(
            f"Image backend returned an unexpected response: "
            f"{message}"
        )

    # -----------------------------------------------------
    # Backend processing time
    # -----------------------------------------------------

    backend_time_header = response.headers.get(
        "X-Processing-Time"
    )

    try:

        backend_time = float(
            backend_time_header
        )

    except (TypeError, ValueError):

        backend_time = (
            time.perf_counter() - start
        )

    # -----------------------------------------------------
    # Save generated image locally
    # -----------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        prefix="docfusion_image_",
        delete=False,
    ) as image_file:

        image_file.write(
            response.content
        )

        image_path = image_file.name

    # -----------------------------------------------------
    # Final logging
    # -----------------------------------------------------

    total_time = (
        time.perf_counter() - start
    )

    print()
    print("=" * 60)
    print("IMAGE GENERATION SUCCESSFUL")
    print("=" * 60)
    print(f"Model         : {model_display}")
    print(f"Backend time  : {backend_time:.2f}s")
    print(f"Total time    : {total_time:.2f}s")
    print(f"Image file    : {image_path}")
    print(f"Image size    : {len(response.content) / 1024:.2f} KB")
    print("=" * 60)
    print()

    return image_path, backend_time