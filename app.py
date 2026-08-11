import gradio as gr
import os
import requests
import tempfile
import time

from Parser import DocumentParser

# =========================================================
# COLAB BACKEND
# =========================================================

COLAB_API = os.getenv(
    "DOCFUSION_COLAB_API",
    "https://recolor-outshine-chest.ngrok-free.dev"
).rstrip("/")


# =========================================================
# LOAD CSS
# =========================================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()


# =========================================================
# MODEL DISPLAY NAMES
# =========================================================

MODEL_NAMES = {
    "bart": "BART-large-CNN",
    "pegasus": "PEGASUS-CNN/DailyMail",
    "distilbart": "DistilBART-CNN",
}



# =========================================================
# FILE PATH HELPER
# =========================================================
# FIX: Required by show_uploaded_file() and parse_document_input().
def get_file_path(file):
    """
    Normalize Gradio UploadButton output to a filesystem path.
    Supports filepath strings and objects exposing .name.
    """
    if file is None:
        return None

    if isinstance(file, str):
        return file

    file_path = getattr(file, "name", None)
    return file_path if file_path else None


# =========================================================
# INPUT / GENERATION HELPERS
# =========================================================

def parse_document_input(file, pasted_text):
    """
    Parse the current user input once and return only the extracted text.
    Keeping the parsed text in Gradio state makes Generate/Regenerate
    independent of the temporary UploadButton file lifecycle.
    """
    if pasted_text and pasted_text.strip():
        parser = DocumentParser(text=pasted_text)
    elif file is not None:
        file_path = get_file_path(file)

        if not file_path:
            raise ValueError("The uploaded file could not be read.")

        parser = DocumentParser(file_path=file_path)
    else:
        raise ValueError(
            "Please upload a PDF, DOCX, TXT file, or paste some text first."
        )

    result = parser.parse()
    document_text = result.get("text", "").strip()

    if not document_text:
        raise ValueError(
            "No readable text could be extracted from the provided content."
        )

    return document_text


def generate_summary_from_text(
    document_text,
    selected_model,
    generation_count=0
):
    """
    Generate a summary from already-parsed text.

    This is deliberately independent of the UploadButton. The document is
    parsed once when the Summary workspace opens; Generate and Regenerate
    reuse the cached text.
    """
    start_time = time.perf_counter()

    selected_model = selected_model or "bart"
    generation_number = int(generation_count or 0) + 1

    if selected_model not in MODEL_NAMES:
        selected_model = "bart"

    model_display = MODEL_NAMES[selected_model]

    def elapsed_seconds():
        return time.perf_counter() - start_time

    def elapsed_html():
        elapsed = elapsed_seconds()

        return f"""
        <div class="workspace-timing">
            <span class="timing-icon">⏱</span>
            <span>Processing completed in
                <strong>{elapsed:.2f}s</strong>
            </span>
        </div>
        """

    def model_info_html(backend_time=None):
        backend_text = (
            f"{backend_time:.2f}s"
            if isinstance(backend_time, (int, float))
            else "—"
        )

        return f"""
        <div class="model-info-card">

            <div class="model-info-item">
                <span class="model-info-label">Model</span>
                <span class="model-info-value">
                    {model_display}
                </span>
            </div>

            <div class="model-info-item">
                <span class="model-info-label">Inference</span>
                <span class="model-info-value">
                    Colab Backend
                </span>
            </div>

            <div class="model-info-item">
                <span class="model-info-label">Task</span>
                <span class="model-info-value">
                    Abstractive Summarization
                </span>
            </div>

            <div class="model-info-item">
                <span class="model-info-label">Backend Time</span>
                <span class="model-info-value">
                    {backend_text}
                </span>
            </div>

        </div>
        """

    def kpi_html(backend_time=None):
        total_time = elapsed_seconds()

        backend_text = (
            f"{backend_time:.2f}s"
            if isinstance(backend_time, (int, float))
            else "—"
        )

        return f"""
        <div class="summary-kpi-row">

            <div class="summary-kpi-card">
                <span class="summary-kpi-label">PROCESSING TIME</span>
                <span class="summary-kpi-value">
                    {total_time:.2f}s
                </span>
                <span class="summary-kpi-sub">
                    End-to-end
                </span>
            </div>

            <div class="summary-kpi-card">
                <span class="summary-kpi-label">MODEL</span>
                <span class="summary-kpi-value model-kpi">
                    {model_display}
                </span>
                <span class="summary-kpi-sub">
                    Selected model
                </span>
            </div>

            <div class="summary-kpi-card">
                <span class="summary-kpi-label">GENERATION</span>
                <span class="summary-kpi-value">
                    #{generation_number}
                </span>
                <span class="summary-kpi-sub">
                    Current run
                </span>
            </div>

            <div class="summary-kpi-card">
                <span class="summary-kpi-label">BACKEND TIME</span>
                <span class="summary-kpi-value">
                    {backend_text}
                </span>
                <span class="summary-kpi-sub">
                    Colab inference
                </span>
            </div>

        </div>
        """

    if not document_text or not document_text.strip():
        return (
            "### Error\n\nNo document text is available. Please go back and "
            "upload a document or paste text.",
            None,
            "No document text available.",
            elapsed_html(),
            model_info_html(),
            kpi_html(),
            generation_number,
        )

    try:
        # -------------------------------------------------
        # Send cached extracted text + selected model
        # -------------------------------------------------

        response = requests.post(
            f"{COLAB_API}/summarize",
            json={
                "text": document_text,
                "model": selected_model,
            },
            timeout=600
        )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                "The Colab backend returned a non-JSON response."
            )

        if not data.get("success"):
            raise RuntimeError(
                data.get(
                    "error",
                    "Summarization failed on the Colab backend."
                )
            )

        summary = data.get("summary", "").strip()

        if not summary:
            raise RuntimeError(
                "The summarization service returned an empty summary."
            )

        backend_time = data.get("processing_time")

        # -------------------------------------------------
        # Create downloadable text file
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="docfusion_summary_",
            delete=False,
            encoding="utf-8"
        ) as summary_file:
            summary_file.write(summary)
            summary_file_path = summary_file.name

        return (
            summary,
            summary_file_path,
            "",
            elapsed_html(),
            model_info_html(backend_time),
            kpi_html(backend_time),
            generation_number,
        )

    except requests.exceptions.RequestException as e:
        return (
            f"""### Connection Error

Could not connect to the Colab backend.

`{str(e)}`

Check that FastAPI and the ngrok tunnel are still running.
""",
            None,
            "Colab connection failed.",
            elapsed_html(),
            model_info_html(),
            kpi_html(),
            generation_number,
        )

    except Exception as e:
        return (
            f"""### Error

{str(e)}
""",
            None,
            "Something went wrong.",
            elapsed_html(),
            model_info_html(),
            kpi_html(),
            generation_number,
        )


def prepare_summary_workspace(file, pasted_text):
    """
    Open the workspace and cache the parsed document text.
    The actual first generation is performed in the chained .then() call.
    """
    try:
        document_text = parse_document_input(file, pasted_text)

        return (
            gr.update(visible=False),  # home_page
            gr.update(visible=True),   # workspace_page
            "⏳ Generating summary...",
            None,
            "",
            "⏳ Connecting to the Colab summarization backend...",
            gr.update(value="bart"),
            0,
            document_text,
        )

    except Exception as e:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            None,
            "",
            f"### Input Error\n\n{str(e)}",
            gr.update(value="bart"),
            0,
            "",
        )


def go_back_home():
    return (
        gr.update(visible=True),
        gr.update(visible=False)
    )


def show_uploaded_file(file):
    file_path = get_file_path(file)

    if not file_path or not os.path.exists(file_path):
        return ""

    size = os.path.getsize(file_path)

    if size < 1024:
        size_text = f"{size} Bytes"
    elif size < 1024 * 1024:
        size_text = f"{size / 1024:.2f} KB"
    else:
        size_text = f"{size / (1024 * 1024):.2f} MB"

    return f"""
    <div class="attachment-card">
        <div class="attachment-icon">📄</div>
        <div class="attachment-info">
            <div class="attachment-title">
                File Uploaded Successfully
            </div>
            <div class="attachment-details">
                <span>{os.path.basename(file_path)}</span>
                <span>•</span>
                <span>{size_text}</span>
            </div>
        </div>
    </div>
    """


def show_pasted_text(text):
    if not text or not text.strip():
        return ""

    words = len(text.split())
    characters = len(text)

    return f"""
    <div class="text-ready-status">
        <span>✓</span>
        <span>Text ready</span>
        <span>•</span>
        <span>{words} words</span>
        <span>•</span>
        <span>{characters} characters</span>
    </div>
    """


def model_selection_changed(model):
    model = model or "bart"

    model_display = MODEL_NAMES.get(
        model,
        model
    )

    return f"""
    <div class="selected-model-status">
        <span class="selected-model-dot">●</span>
        <span>Selected model:</span>
        <strong>{model_display}</strong>
    </div>
    """


def show_image_click():
    return """
    <div class="tool-timing tool-timing-info">
        <span class="timing-icon">⏱</span>
        <span>Image generation selected</span>
    </div>
    """


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="DocFusion AI"
) as demo:

    # Tracks how many summary generations have been performed
    # in the current workspace session.
    generation_state = gr.State(0)
    # Cached extracted document text. Generate/Regenerate reuse this state.
    document_text_state = gr.State("")

    # =====================================================
    # HOME PAGE
    # =====================================================

    with gr.Column(
        visible=True,
        elem_classes="home-page"
    ) as home_page:

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        gr.HTML("""
        <div class="header">
            <h1>📄 DocFusion AI</h1>
            <p>Upload a document or paste your text and let AI do the rest.</p>
        </div>
        """)

        # -------------------------------------------------
        # Input Card
        # -------------------------------------------------

        with gr.Column(
            elem_classes="main-card"
        ):

            gr.HTML("""
            <div class="input-heading">
                <h2>Add your content</h2>
                <p>Paste your text or attach a PDF, DOCX, or TXT file.</p>
            </div>
            """)

            with gr.Row(
                elem_classes="compact-input-wrapper"
            ):

                upload_file = gr.UploadButton(
                    "+",
                    file_types=[
                        ".pdf",
                        ".docx",
                        ".txt"
                    ],
                    file_count="single",
                    elem_classes="plus-button"
                )

                pasted_text = gr.Textbox(
                    placeholder="Ask anything",
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    container=False,
                    elem_classes="compact-textbox"
                )

            upload_status = gr.HTML(
                "",
                elem_classes="upload-status"
            )

            paste_status = gr.HTML(
                "",
                elem_classes="paste-status"
            )

            upload_file.upload(
                fn=show_uploaded_file,
                inputs=upload_file,
                outputs=upload_status
            )

            pasted_text.change(
                fn=show_pasted_text,
                inputs=pasted_text,
                outputs=paste_status
            )

        # -------------------------------------------------
        # Generation Section
        # -------------------------------------------------

        gr.HTML("""
        <div class="generate-header">
            <h2>What would you like to generate?</h2>
            <p>AI suggestions based on your content</p>
        </div>
        """)

        with gr.Row(
            equal_height=True,
            elem_classes="generation-cards-row"
        ):

            # =============================================
            # SUMMARY
            # =============================================

            with gr.Column(
                elem_classes="tool-card"
            ):
                summary_button = gr.Button(
                    value="📄  Summary",
                    elem_classes="tool-card-button"
                )

                summary_time = gr.HTML(
                    "",
                    elem_classes="tool-timing-slot"
                )

            # =============================================
            # IMAGE
            # =============================================

            with gr.Column(
                elem_classes="tool-card"
            ):
                generate_image_button = gr.Button(
                    value="🖼️  Generate Image",
                    elem_classes="tool-card-button"
                )

                image_time = gr.HTML(
                    "",
                    elem_classes="tool-timing-slot"
                )

            # =============================================
            # SPEECH
            # =============================================

            with gr.Column(
                elem_classes="tool-card"
            ):
                gr.Button(
                    value="🔊  Text to Speech",
                    elem_classes="tool-card-button"
                )

                gr.HTML(
                    "",
                    elem_classes="tool-timing-slot"
                )

        # -------------------------------------------------
        # Small AI Tools
        # -------------------------------------------------

        gr.HTML("""
        <div class="small-tools">

            <div class="small-tool">⭐ Key Points</div>
            <div class="small-tool">🧠 Mind Map</div>
            <div class="small-tool">🌐 Translate</div>
            <div class="small-tool">💬 Q&A</div>
            <div class="small-tool">🕒 Timeline</div>

        </div>
        """)

        home_status = gr.HTML(
            "",
            elem_classes="home-status"
        )

    # =====================================================
    # AI WORKSPACE
    # =====================================================

    with gr.Column(
        visible=False,
        elem_classes="workspace-page"
    ) as workspace_page:

        # -------------------------------------------------
        # Workspace Top Bar
        # -------------------------------------------------

        with gr.Row(
            elem_classes="workspace-topbar"
        ):

            back_button = gr.Button(
                "← Back",
                elem_classes="back-button"
            )

            gr.HTML("""
            <div class="workspace-title">
                <h1>AI Workspace</h1>
                <p>DocFusion AI</p>
            </div>
            """)

        # -------------------------------------------------
        # Workspace Tool Header
        # -------------------------------------------------

        gr.HTML("""
        <div class="workspace-tool-header">

            <div class="workspace-tool-icon">
                📄
            </div>

            <div>
                <h2>Document Summary</h2>
                <p>
                    AI-generated summary powered by
                    DocFusion AI
                </p>
            </div>

        </div>
        """)

        # -------------------------------------------------
        # Model Selector
        # -------------------------------------------------

        with gr.Column(
            elem_classes="model-selector-card"
        ):

            with gr.Row(
                elem_classes="model-selector-row"
            ):

                with gr.Column(
                    elem_classes="model-selector-copy"
                ):
                    gr.HTML("""
                    <div class="model-selector-title">
                        Choose Summarization Model
                    </div>
                    <div class="model-selector-description">
                        Generate with one model at a time and compare
                        the result and processing time.
                    </div>
                    """)

                model_dropdown = gr.Dropdown(
                    choices=[
                        ("BART-large-CNN", "bart"),
                        ("PEGASUS-CNN/DailyMail", "pegasus"),
                        ("DistilBART-CNN", "distilbart"),
                    ],
                    value="bart",
                    label="Model",
                    show_label=True,
                    allow_custom_value=False,
                    elem_classes="model-dropdown"
                )

            selected_model_status = gr.HTML(
                """
                <div class="selected-model-status">
                    <span class="selected-model-dot">●</span>
                    <span>Selected model:</span>
                    <strong>BART-large-CNN</strong>
                </div>
                """,
                elem_classes="selected-model-status-wrapper"
            )

        # -------------------------------------------------
        # Generate Controls
        # -------------------------------------------------

        with gr.Row(
            elem_classes="generation-control-row"
        ):

            generate_summary_button = gr.Button(
                "✨ Generate Summary",
                variant="primary",
                interactive=True,
                elem_classes="generate-summary-button"
            )

            regenerate_button = gr.Button(
                "🔄 Regenerate Same Model",
                interactive=True,
                elem_classes="regenerate-button"
            )

        # -------------------------------------------------
        # Summary Result
        # -------------------------------------------------

        with gr.Column(
            elem_classes="summary-result-card"
        ):

            gr.HTML("""
            <div class="summary-result-header">

                <div>
                    <h3>Generated Summary</h3>
                    <p>
                        A concise representation of your document
                    </p>
                </div>

                <div class="ai-badge">
                    AI Generated
                </div>

            </div>
            """)

            summary_output = gr.Markdown(
                value="Your summary will appear here.",
                elem_classes="summary-output"
            )

        # -------------------------------------------------
        # Timing / KPI
        # -------------------------------------------------

        summary_kpis = gr.HTML(
            "",
            elem_classes="summary-kpis"
        )

        # -------------------------------------------------
        # Workspace Actions
        # -------------------------------------------------

        with gr.Row(
            elem_classes="workspace-actions"
        ):

            download_summary = gr.DownloadButton(
                "⬇️ Download Summary",
                value=None,
                elem_classes="workspace-action-button"
            )

        # -------------------------------------------------
        # Model Information
        # -------------------------------------------------

        model_info = gr.HTML(
            "",
            elem_classes="dynamic-model-info"
        )

        workspace_status = gr.HTML(
            "",
            elem_classes="workspace-status"
        )


    # =====================================================
    # EVENTS
    # =====================================================

    # -----------------------------------------------------
    # Open workspace + parse/cache input + first generation
    # -----------------------------------------------------

    summary_button.click(
        fn=prepare_summary_workspace,
        inputs=[
            upload_file,
            pasted_text,
        ],
        outputs=[
            home_page,
            workspace_page,
            summary_output,
            download_summary,
            summary_time,
            workspace_status,
            model_dropdown,
            generation_state,
            document_text_state,
        ],
        show_progress="hidden"
    ).then(
        fn=generate_summary_from_text,
        inputs=[
            document_text_state,
            model_dropdown,
            generation_state,
        ],
        outputs=[
            summary_output,
            download_summary,
            workspace_status,
            summary_time,
            model_info,
            summary_kpis,
            generation_state,
        ],
        show_progress="hidden"
    )

    # -----------------------------------------------------
    # Generate using CURRENTLY selected model
    # -----------------------------------------------------

    generate_summary_button.click(
        fn=generate_summary_from_text,
        inputs=[
            document_text_state,
            model_dropdown,
            generation_state,
        ],
        outputs=[
            summary_output,
            download_summary,
            workspace_status,
            summary_time,
            model_info,
            summary_kpis,
            generation_state,
        ],
        show_progress="hidden"
    )

    # -----------------------------------------------------
    # Regenerate using the SAME currently selected model
    # -----------------------------------------------------

    regenerate_button.click(
        fn=generate_summary_from_text,
        inputs=[
            document_text_state,
            model_dropdown,
            generation_state,
        ],
        outputs=[
            summary_output,
            download_summary,
            workspace_status,
            summary_time,
            model_info,
            summary_kpis,
            generation_state,
        ],
        show_progress="hidden"
    )

    # -----------------------------------------------------
    # Model selector status
    # -----------------------------------------------------

    model_dropdown.change(
        fn=model_selection_changed,
        inputs=model_dropdown,
        outputs=selected_model_status,
        show_progress="hidden"
    )

    # -----------------------------------------------------
    # Image card
    # -----------------------------------------------------

    generate_image_button.click(
        fn=show_image_click,
        inputs=[],
        outputs=[image_time],
        show_progress="hidden"
    )

    # -----------------------------------------------------
    # Back
    # -----------------------------------------------------

    back_button.click(
        fn=go_back_home,
        inputs=[],
        outputs=[
            home_page,
            workspace_page
        ],
        show_progress="hidden"
    )


# =========================================================
# LAUNCH
# =========================================================

# Queue generation requests so two model-switch/generation requests
# cannot modify the shared backend state at the same time.
demo.queue(
    max_size=20,
    default_concurrency_limit=1
)

demo.launch(
    css=css
)