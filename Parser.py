import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from langdetect import detect, LangDetectException


class PDFParser:
    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, pdf_path):
        self.pdf_path = str(pdf_path)
        self.file_path = Path(pdf_path)
        self.doc = None

    # ----------------------------
    # Validation
    # ----------------------------
    def validate_file(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found.")

        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError("Only PDF files are supported.")

        return True

    # ----------------------------
    # Open PDF
    # ----------------------------
    def open_pdf(self):
        self.doc = fitz.open(self.pdf_path)
        return self.doc

    # ----------------------------
    # Metadata
    # ----------------------------
    def extract_metadata(self):
        meta = self.doc.metadata or {}
        return {
            "title": meta.get("title"),
            "author": meta.get("author"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "subject": meta.get("subject"),
            "keywords": meta.get("keywords"),
        }

    # ----------------------------
    # Text Cleaning
    # ----------------------------
    def clean_text(self, text):
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    # ----------------------------
    # Text Extraction
    # ----------------------------
    def extract_text(self):
        pages = []
        full_text = []

        for i, page in enumerate(self.doc):
            txt = self.clean_text(page.get_text("text"))
            pages.append(
                {
                    "page": i + 1,
                    "text": txt
                }
            )
            full_text.append(txt)

        return pages, "\n".join(full_text)

    # ----------------------------
    # Statistics
    # ----------------------------
    def get_statistics(self, text):
        words = len(text.split())
        chars = len(text)
        paragraphs = len([p for p in text.split("\n") if p.strip()])
        reading = max(1, round(words / 200))

        return {
            "pages": len(self.doc),
            "words": words,
            "characters": chars,
            "paragraphs": paragraphs,
            "estimated_reading_minutes": reading,
        }

    # ----------------------------
    # Language
    # ----------------------------
    def detect_language(self, text):
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    # ----------------------------
    # Searchable or Scanned
    # ----------------------------
    def detect_pdf_type(self):
        searchable = 0

        for page in self.doc:
            if page.get_text().strip():
                searchable += 1

        if searchable == len(self.doc):
            return "Searchable PDF"

        return "Scanned / OCR Required"

    # ----------------------------
    # Images
    # ----------------------------
    def extract_images(self, output_dir="outputs/images"):
        os.makedirs(output_dir, exist_ok=True)

        image_paths = []

        for page_index in range(len(self.doc)):
            images = self.doc[page_index].get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                pix = fitz.Pixmap(self.doc, xref)

                if pix.n >= 5:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                name = f"page_{page_index+1}_{img_index+1}.png"
                path = os.path.join(output_dir, name)

                pix.save(path)
                image_paths.append(path)

        return image_paths

    # ----------------------------
    # Tables
    # ----------------------------
    def extract_tables(self):
        tables = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()

                if page_tables:
                    tables.append(
                        {
                            "page": page_no + 1,
                            "tables": page_tables
                        }
                    )

        return tables

    # ----------------------------
    # File Info
    # ----------------------------
    def file_info(self):
        size = round(self.file_path.stat().st_size / (1024 * 1024), 2)

        return {
            "file_name": self.file_path.name,
            "file_size_mb": size,
        }

    # ----------------------------
    # Parse
    # ----------------------------
    def parse(self,
              extract_images=False,
              extract_tables=False):

        self.validate_file()
        self.open_pdf()

        pages, full_text = self.extract_text()

        data = {
            "file_info": self.file_info(),
            "metadata": self.extract_metadata(),
            "statistics": self.get_statistics(full_text),
            "language": self.detect_language(full_text),
            "pdf_type": self.detect_pdf_type(),
            "pages": pages,
            "text": full_text,
            "images": [],
            "tables": [],
        }

        if extract_images:
            data["images"] = self.extract_images()

        if extract_tables:
            data["tables"] = self.extract_tables()

        self.doc.close()

        return data


if __name__ == "__main__":
    parser = PDFParser("sample.pdf")

    result = parser.parse(
        extract_images=False,
        extract_tables=False
    )

    print(result["file_info"])
    print(result["statistics"])