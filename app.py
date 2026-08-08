import gradio as gr
import os


# ==========================================
# Load CSS
# ==========================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()


# ==========================================
# Functions
# ==========================================

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


# ==========================================
# UI
# ==========================================

with gr.Blocks(title="DocFusion AI") as demo:

    # ======================================
    # Header
    # ======================================

    gr.HTML("""
    <div class="header">
        <h1>📄 DocFusion AI</h1>
        <p>Upload a document or paste your text and let AI do the rest.</p>
    </div>
    """)


    # ======================================
    # Input Card
    # ======================================

    with gr.Column(elem_classes="main-card"):

        gr.HTML("""
        <div class="input-heading">
            <h2>Add your content</h2>
            <p>Paste your text or attach a PDF, DOCX, or TXT file.</p>
        </div>
        """)

        # ==============================================
        # Compact Unified Input
        # ==============================================

        with gr.Row(elem_classes="compact-input-wrapper"):

            # Use Gradio's native UploadButton.
            # This is more reliable than manually calling a hidden
            # <input type="file"> with JavaScript.
            upload_file = gr.UploadButton(
                "+",
                file_types=[".pdf", ".docx", ".txt"],
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

        # ==============================================
        # Content Status
        # ==============================================

        upload_status = gr.HTML(
            "",
            elem_classes="upload-status"
        )

        paste_status = gr.HTML(
            "",
            elem_classes="paste-status"
        )

        # ==============================================
        # Input Events
        # ==============================================

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


    # ==========================================
    # What would you like to generate?
    # ==========================================

    gr.HTML("""
    <div class="generate-header">
        <h2>What would you like to generate?</h2>
        <p>AI suggestions based on your content</p>
    </div>
    """)

    with gr.Row(equal_height=True):

        # ======================================
        # Summary
        # ======================================

        with gr.Column(
            scale=1,
            min_width=180,
            elem_classes="tool-card"
        ):

            gr.HTML("""
            <div class="tool-content">
                <div class="tool-icon purple">📄</div>

                <div class="tool-info">
                    <div class="tool-title">Summary</div>

                    <div class="tool-description">
                        Get a clear and concise summary of this content.
                    </div>
                </div>
            </div>
            """)

        # ======================================
        # Generate Image
        # ======================================

        with gr.Column(
            scale=1,
            min_width=180,
            elem_classes="tool-card"
        ):

            gr.HTML("""
            <div class="tool-content">
                <div class="tool-icon green">🖼️</div>

                <div class="tool-info">
                    <div class="tool-title">Generate Image</div>

                    <div class="tool-description">
                        Create relevant images, diagrams or infographics.
                    </div>
                </div>
            </div>
            """)

        # ======================================
        # Text to Speech
        # ======================================

        with gr.Column(
            scale=1,
            min_width=180,
            elem_classes="tool-card"
        ):

            gr.HTML("""
            <div class="tool-content">
                <div class="tool-icon orange">🔊</div>

                <div class="tool-info">
                    <div class="tool-title">Text to Speech</div>

                    <div class="tool-description">
                        Listen to this content with a natural AI voice.
                    </div>
                </div>
            </div>
            """)


    # ==========================================
    # Small AI Tools
    # ==========================================

    gr.HTML("""
    <div class="small-tools">

        <div class="small-tool">⭐ Key Points</div>
        <div class="small-tool">🧠 Mind Map</div>
        <div class="small-tool">🌐 Translate</div>
        <div class="small-tool">💬 Q&A</div>
        <div class="small-tool">🕒 Timeline</div>

    </div>
    """)


# ==========================================
# Launch
# ==========================================

demo.launch(css=css)