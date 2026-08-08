import gradio as gr
import os
import requests
import tempfile

from Parser import DocumentParser

# =========================================================
# COLAB BACKEND
# =========================================================

COLAB_API = "https://recolor-outshine-chest.ngrok-free.dev"


# =========================================================
# LOAD CSS
# =========================================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()


# =========================================================
# BACKEND FUNCTIONS
# =========================================================

def generate_summary(file, pasted_text):
    """
    Parse the uploaded document / pasted text locally,
    send only extracted text to the Colab BART API,
    and return the generated summary.
    """

    try:
        # -------------------------------------------------
        # Determine input
        # -------------------------------------------------

        if pasted_text and pasted_text.strip():

            parser = DocumentParser(
                text=pasted_text
            )

        elif file is not None:

            parser = DocumentParser(
                file_path=file.name
            )

        else:

            raise ValueError(
                "Please upload a PDF, DOCX, TXT file, "
                "or paste some text first."
            )

        # -------------------------------------------------
        # Parse locally
        # -------------------------------------------------

        result = parser.parse()

        document_text = result.get(
            "text",
            ""
        ).strip()

        if not document_text:

            raise ValueError(
                "No readable text could be extracted "
                "from the provided content."
            )

        # -------------------------------------------------
        # Send extracted text to Colab
        # -------------------------------------------------

        response = requests.post(
            f"{COLAB_API}/summarize",
            json={
                "text": document_text
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("success"):

            raise RuntimeError(
                data.get(
                    "error",
                    "Summarization failed."
                )
            )

        summary = data.get(
            "summary",
            ""
        ).strip()

        if not summary:

            raise RuntimeError(
                "The summarization service returned "
                "an empty summary."
            )

        # -------------------------------------------------
        # Create downloadable text file
        # -------------------------------------------------

        summary_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="docfusion_summary_",
            delete=False,
            encoding="utf-8"
        )

        summary_file.write(summary)
        summary_file.close()

        # -------------------------------------------------
        # Open workspace
        # -------------------------------------------------

        return (
            summary,
            summary_file.name,
            gr.update(visible=False),
            gr.update(visible=True),
            ""
        )

    except requests.exceptions.RequestException as e:

        return (
            f"### Connection Error\n\n"
            f"Could not connect to the Colab backend.\n\n"
            f"`{str(e)}`\n\n"
            f"Make sure your Colab notebook and ngrok tunnel "
            f"are still running.",
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "Colab connection failed."
        )

    except Exception as e:

        return (
            f"### Error\n\n{str(e)}",
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "Something went wrong."
        )


def go_back_home():
    return (
        gr.update(visible=True),
        gr.update(visible=False)
    )


def show_uploaded_file(file):
    if file is None:
        return ""

    size = os.path.getsize(file.name)

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
                <span>{os.path.basename(file.name)}</span>
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


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    title="DocFusion AI"
) as demo:

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
            equal_height=True
        ):

            # =============================================
            # SUMMARY
            # =============================================

            summary_button = gr.Button(
                value="📄  Summary",
                elem_classes="tool-card tool-card-button"
            )

            # =============================================
            # IMAGE
            # =============================================

            gr.Button(
                value="🖼️  Generate Image",
                elem_classes="tool-card tool-card-button"
            )

            # =============================================
            # SPEECH
            # =============================================

            gr.Button(
                value="🔊  Text to Speech",
                elem_classes="tool-card tool-card-button"
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

        gr.HTML("""
        <div class="model-info-card">

            <div class="model-info-item">
                <span class="model-info-label">Model</span>
                <span class="model-info-value">
                    BART-large-CNN
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

        </div>
        """)

    # =====================================================
    # EVENTS
    # =====================================================

    summary_button.click(
        fn=generate_summary,
        inputs=[
            upload_file,
            pasted_text
        ],
        outputs=[
            summary_output,
            download_summary,
            home_page,
            workspace_page,
            home_status
        ]
    )

    back_button.click(
        fn=go_back_home,
        inputs=[],
        outputs=[
            home_page,
            workspace_page
        ]
    )


# =========================================================
# LAUNCH
# =========================================================

demo.launch(
    css=css
)