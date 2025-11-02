#!/usr/bin/env python3
"""
Convert PDF to Markdown using Qwen3-VL via llama-server (GGUF)

Uses your local GGUF model with llama-server for vision tasks.

Dependencies:
  uv pip install --upgrade pymupdf pillow requests

Usage:
  python3 pdf_to_markdown_llamacpp.py input.pdf output.md
  python3 pdf_to_markdown_llamacpp.py input.pdf output.md --dpi 150 --model Q8_0
"""

import sys
import argparse
from pathlib import Path
import tempfile
import shutil
import subprocess
import time
import signal
import base64
import json

# Check dependencies
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import requests
except ImportError as e:
    print("Missing dependencies. Please install with:")
    print("  uv pip install --upgrade pymupdf pillow requests")
    print(f"\nError: {e}")
    sys.exit(1)


class LlamaServer:
    """Manage llama-server lifecycle"""

    def __init__(self, model_path: Path, mmproj_path: Path, port: int = 8080):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def start(self):
        """Start llama-server"""
        print(f"Starting llama-server...")
        print(f"  Model: {self.model_path.name}")
        print(f"  MMProj: {self.mmproj_path.name}")
        print(f"  Port: {self.port}")

        cmd = [
            "llama-server",
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--port", str(self.port),
            "--ctx-size", "4096",
            "-ngl", "999",  # Offload all layers to GPU
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to be ready
        print("Waiting for server to start...")
        for i in range(60):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print("✓ Server ready!")
                    return True
            except requests.exceptions.RequestException:
                pass

            if self.process.poll() is not None:
                print("✗ Server failed to start!")
                stdout, stderr = self.process.communicate()
                print("STDOUT:", stdout[-500:] if stdout else "")
                print("STDERR:", stderr[-500:] if stderr else "")
                return False

            time.sleep(1)

        print("✗ Server startup timeout!")
        return False

    def stop(self):
        """Stop llama-server"""
        if self.process:
            print("\nStopping llama-server...")
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=5)
            print("✓ Server stopped")

    def generate(self, prompt: str, image_path: Path, max_tokens: int = 2048):
        """Generate text from image using vision model"""

        # Read and encode image as data URL
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Create data URL for image
        image_url = f"data:image/png;base64,{image_data}"

        # Prepare request using OpenAI-compatible chat completions API
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            # Extract content from OpenAI-format response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            print(f"✗ Error generating response: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return ""


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150):
    """Convert PDF pages to images"""
    print(f"\nConverting PDF to images (DPI: {dpi})...")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    image_files = []

    for page_num in range(total_pages):
        page = doc[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        output_file = output_dir / f"page_{page_num + 1:04d}.png"
        pix.save(output_file)
        image_files.append(output_file)
        print(f"  Page {page_num + 1}/{total_pages} converted")

    doc.close()
    return image_files


def extract_text_from_image(server: LlamaServer, image_path: Path, page_num: int):
    """Extract text from image using llama-server"""
    print(f"\nProcessing page {page_num}...")

    prompt = """Perform OCR on this image. Extract ALL visible text exactly as it appears.

Preserve the original formatting, structure, headings, lists, and tables using markdown syntax.
Do NOT add any commentary, explanations, or content that is not visible in the image.
Only output the text you can actually read from the image."""

    return server.generate(prompt, image_path, max_tokens=4096)


def pdf_to_markdown(pdf_path: Path, output_path: Path, model_variant: str = "Q8_0", dpi: int = 150):
    """Convert PDF to markdown using Qwen3-VL GGUF"""

    # Determine model paths
    model_dir = Path.home() / "models" / "Qwen3-VL-32B-Instruct-GGUF"
    model_path = model_dir / f"Qwen3VL-32B-Instruct-{model_variant}.gguf"
    mmproj_path = model_dir / f"mmproj-Qwen3VL-32B-Instruct-{model_variant}.gguf"

    # Verify files exist
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print(f"\nAvailable models in {model_dir}:")
        for f in model_dir.glob("Qwen3VL-32B-Instruct-*.gguf"):
            print(f"  - {f.name}")
        sys.exit(1)

    if not mmproj_path.exists():
        print(f"✗ MMProj not found: {mmproj_path}")
        sys.exit(1)

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Convert PDF to images
        image_files = pdf_to_images(pdf_path, temp_dir, dpi=dpi)

        # Start llama-server
        server = LlamaServer(model_path, mmproj_path)
        if not server.start():
            print("Failed to start server!")
            sys.exit(1)

        try:
            # Process each page
            markdown_content = []
            markdown_content.append(f"# {pdf_path.stem}\n\n")
            markdown_content.append(f"*Converted from PDF using Qwen3-VL*\n\n")
            markdown_content.append("---\n\n")

            for idx, image_file in enumerate(image_files, start=1):
                page_text = extract_text_from_image(server, image_file, idx)

                markdown_content.append(f"## Page {idx}\n\n")
                markdown_content.append(page_text.strip())
                markdown_content.append("\n\n---\n\n")

            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(markdown_content))

            print(f"\n✓ Markdown saved to: {output_path}")

        finally:
            server.stop()

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using Qwen3-VL GGUF model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Q8_0 quantization (default)
  python3 pdf_to_markdown_llamacpp.py document.pdf output.md

  # Use Q4_K_M quantization (faster, less VRAM)
  python3 pdf_to_markdown_llamacpp.py document.pdf output.md --model Q4_K_M

  # Use F16 (full precision, best quality)
  python3 pdf_to_markdown_llamacpp.py document.pdf output.md --model F16

  # Custom DPI
  python3 pdf_to_markdown_llamacpp.py document.pdf output.md --dpi 200
        """
    )

    parser.add_argument("input", type=str, help="Input PDF file")
    parser.add_argument("output", type=str, help="Output markdown file")
    parser.add_argument(
        "--model",
        type=str,
        default="Q8_0",
        choices=["Q4_K_M", "Q8_0", "F16"],
        help="Model quantization variant (default: Q8_0)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF conversion (default: 150)"
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix.lower() == '.pdf':
        print(f"✗ Input must be a PDF file: {input_path}")
        sys.exit(1)

    pdf_to_markdown(input_path, output_path, model_variant=args.model, dpi=args.dpi)


if __name__ == "__main__":
    main()
