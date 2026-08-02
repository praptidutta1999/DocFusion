import gradio as gr
from Parser import PDFParser
import os


# ==========================================
# Load CSS
# ==========================================

with open("style.css", "r", encoding="utf-8") as f:
    css = f.read()


# ==========================================
# Functions
# ==========================================

import os

def show_uploaded_file(file):
    if file is None:
        return "No file uploaded."

    size = os.path.getsize(file.name)

    if size < 1024:
        size_text = f"{size} Bytes"
    elif size < 1024 * 1024:
        size_text = f"{size/1024:.2f} KB"
    else:
        size_text = f"{size/(1024*1024):.2f} MB"

    return f"""✅ File Uploaded Successfully

📄 File Name : {os.path.basename(file.name)}
📦 Size      : {size_text}
"""








def analyze(file):

    if file is None:
        return "Please upload a document."

    parser = PDFParser(file.name)

    result = parser.parse()

    return f"""
    File Name : {result['file_info']['file_name']}
    Pages     : {result['statistics']['pages']}
    Words     : {result['statistics']['words']}
    Language  : {result['language']}
    PDF Type  : {result['pdf_type']}
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
        <p>Upload your document and let AI do the rest.</p>
    </div>
    """)

    # ======================================
    # Upload Card
    # ======================================

    with gr.Column(elem_classes="main-card"):

        with gr.Group(elem_classes="upload-wrapper"):

            gr.HTML("""
            <div class="upload-box">

                <div class="upload-icon">↑</div>

                <div class="upload-title">
                    Drag & Drop your file here
                </div>

                <div class="upload-subtitle">
                    or click anywhere to browse
                </div>

                <div class="file-badges">
                    <span>PDF</span>
                    <span>DOCX</span>
                    <span>TXT</span>
                    <span>PNG</span>
                    <span>JPG</span>
                </div>

            </div>
            """)

            file = gr.File(
                label="",
                elem_id="real_upload",
                file_count="single",
                file_types=[".pdf", ".docx", ".txt", ".png", ".jpg"],
            )
            
            upload_status = gr.Textbox(
                label="Uploaded File",
                 interactive=False
                )
            file.change(
                    fn=show_uploaded_file,
                    inputs=file,
                    outputs=upload_status
                    )


        analyze_btn = gr.Button("Analyze Document", variant="primary")

        status = gr.Textbox(
            label="Status",
            interactive=False
        )

        analyze_btn.click(
            fn=analyze,
            inputs=file,
            outputs=status
        )

    # ======================================
    # What would you like to generate?
    # ======================================

    gr.HTML("""
    <div class="generate-header">
        <h2>What would you like to generate?</h2>
        <p>AI suggestions based on your document</p>
    </div>
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=1, min_width=180, elem_classes="tool-card"):

            gr.HTML("""
            <div class="tool-content">

                <div class="tool-icon purple">📄</div>

                <div class="tool-info">
                    <div class="tool-title">Summary</div>

                    <div class="tool-description">
                        Get a clear and concise summary of this document.
                    </div>
                </div>

            </div>
            """)

        with gr.Column(scale=1, min_width=180, elem_classes="tool-card"):

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

        with gr.Column(scale=1, min_width=180, elem_classes="tool-card"):

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

    # ======================================
    # Small AI Tools
    # ======================================

    gr.HTML("""
    <div class="small-tools">

        <div class="small-tool">
            ⭐ Key Points
        </div>

        <div class="small-tool">
            🧠 Mind Map
        </div>

        <div class="small-tool">
            🌐 Translate
        </div>

        <div class="small-tool">
            💬 Q&A
        </div>

        <div class="small-tool">
            🕒 Timeline
        </div>

    </div>
    """)


# ==========================================
# Launch
# ==========================================

demo.launch(css=css)