#!/usr/bin/env python3
"""
Convert PDF to XHTML using Qwen3-VL via llama-server (GGUF)

Uses your local GGUF model with llama-server for vision tasks.

Dependencies:
  uv pip install --upgrade pymupdf pillow requests

Usage:
  python3 pdf_to_xhtml_llamacpp.py input.pdf output.xhtml
  python3 pdf_to_xhtml_llamacpp.py input.pdf output.xhtml --dpi 150 --model Q8_0
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
import html
import logging
from datetime import datetime

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


def extract_text_from_image(server: LlamaServer, image_path: Path, page_num: int, logger: logging.Logger):
    """Extract text from image using llama-server"""
    print(f"\nProcessing page {page_num}...")
    logger.info(f"Processing page {page_num}...")

    prompt = """Perform OCR on this image. Extract ALL visible text exactly as it appears.

Preserve the original formatting and structure. Identify the following elements:
- Headings (mark with [H1], [H2], [H3] prefix)
- Paragraphs (regular text blocks)
- Lists (bulleted or numbered)
- Block quotes or indented text
- Any other structural elements

Do NOT add any commentary or explanations.
Only output the text you can actually read from the image with structural markers."""

    try:
        result = server.generate(prompt, image_path, max_tokens=4096)
        if result:
            logger.info(f"Page {page_num}: Successfully extracted {len(result)} characters")
        else:
            logger.warning(f"Page {page_num}: No text extracted")
        return result
    except Exception as e:
        logger.error(f"Page {page_num}: Error during extraction - {str(e)}")
        raise


def convert_to_xhtml_paragraphs(text: str) -> str:
    """Convert extracted text to XHTML paragraphs with proper entity encoding"""
    if not text.strip():
        return ""

    lines = text.split('\n')
    xhtml_parts = []
    current_para = []

    for line in lines:
        line = line.strip()

        if not line:
            # Empty line ends current paragraph
            if current_para:
                para_text = ' '.join(current_para)
                # Convert special characters to HTML entities
                para_text = html.escape(para_text, quote=False)
                # Convert common punctuation to proper entities
                para_text = para_text.replace('"', '&ldquo;').replace('"', '&rdquo;')
                para_text = para_text.replace("'", '&lsquo;').replace("'", '&rsquo;')
                para_text = para_text.replace('—', '&mdash;')
                para_text = para_text.replace('–', '&ndash;')

                xhtml_parts.append(f"<p>\n{para_text}\n</p>\n")
                current_para = []
        elif line.startswith('[H1]'):
            # Heading 1
            if current_para:
                para_text = html.escape(' '.join(current_para), quote=False)
                xhtml_parts.append(f"<p>\n{para_text}\n</p>\n")
                current_para = []
            heading_text = html.escape(line[4:].strip(), quote=False)
            xhtml_parts.append(f"<h1>\n{heading_text}\n</h1>\n")
        elif line.startswith('[H2]'):
            # Heading 2
            if current_para:
                para_text = html.escape(' '.join(current_para), quote=False)
                xhtml_parts.append(f"<p>\n{para_text}\n</p>\n")
                current_para = []
            heading_text = html.escape(line[4:].strip(), quote=False)
            xhtml_parts.append(f"<h2>\n{heading_text}\n</h2>\n")
        elif line.startswith('[H3]'):
            # Heading 3
            if current_para:
                para_text = html.escape(' '.join(current_para), quote=False)
                xhtml_parts.append(f"<p>\n{para_text}\n</p>\n")
                current_para = []
            heading_text = html.escape(line[4:].strip(), quote=False)
            xhtml_parts.append(f"<h3>\n{heading_text}\n</h3>\n")
        else:
            # Regular text - accumulate for paragraph
            current_para.append(line)

    # Don't forget the last paragraph
    if current_para:
        para_text = ' '.join(current_para)
        para_text = html.escape(para_text, quote=False)
        para_text = para_text.replace('"', '&ldquo;').replace('"', '&rdquo;')
        para_text = para_text.replace("'", '&lsquo;').replace("'", '&rsquo;')
        para_text = para_text.replace('—', '&mdash;')
        para_text = para_text.replace('–', '&ndash;')
        xhtml_parts.append(f"<p>\n{para_text}\n</p>\n")

    return '\n'.join(xhtml_parts)


def create_xhtml_header(title: str) -> str:
    """Create XHTML document header with CSS styling"""
    return f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8" />
<meta http-equiv="Content-Style-Type" content="text/css" />
<title>{html.escape(title)}</title>
<style type="text/css">

body {{ margin-left: 10%;
       margin-right: 10%;
       text-align: justify }}

h1, h2, h3, h4, h5 {{text-align: center; font-style: normal; font-weight:
normal; line-height: 1.5; margin-top: .5em; margin-bottom: .5em;}}

h1 {{font-size: 300%;
    margin-top: 0.6em;
    margin-bottom: 0.6em;
    letter-spacing: 0.12em;
    word-spacing: 0.2em;
    text-indent: 0em;}}
h2 {{font-size: 150%; margin-top: 2em; margin-bottom: 1em;}}
h3 {{font-size: 130%; margin-top: 1em;}}
h4 {{font-size: 120%;}}
h5 {{font-size: 110%;}}

.no-break {{page-break-before: avoid;}} /* for epubs */

div.chapter {{page-break-before: always; margin-top: 4em;}}

hr {{width: 80%; margin-top: 2em; margin-bottom: 2em;}}

p {{text-indent: 1em;
   margin-top: 0.25em;
   margin-bottom: 0.25em; }}

p.right {{text-align: right;
         margin-right: 10%;
         margin-top: 1em;
         margin-bottom: 1em; }}

div.fig {{ display:block;
          margin:0 auto;
          text-align:center;
          margin-top: 1em;
          margin-bottom: 1em;}}

a:link {{color:blue; text-decoration:none}}
a:visited {{color:blue; text-decoration:none}}
a:hover {{color:red}}

</style>

</head>

<body>
"""


def create_xhtml_footer() -> str:
    """Create XHTML document footer"""
    return """</body>
</html>
"""


def setup_logger(output_path: Path) -> logging.Logger:
    """Setup logger to write to log file alongside output"""
    log_path = output_path.parent / f"{output_path.stem}.log"

    logger = logging.getLogger('pdf_to_xhtml')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Format: timestamp - level - message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def pdf_to_xhtml(pdf_path: Path, output_path: Path, model_variant: str = "Q8_0", dpi: int = 150):
    """Convert PDF to XHTML using Qwen3-VL GGUF"""

    # Setup logging
    logger = setup_logger(output_path)
    logger.info("="*60)
    logger.info("PDF to XHTML Conversion Started")
    logger.info("="*60)
    logger.info(f"Input PDF: {pdf_path}")
    logger.info(f"Output XHTML: {output_path}")
    logger.info(f"Model variant: {model_variant}")
    logger.info(f"DPI: {dpi}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Determine model paths
    model_dir = Path.home() / "models" / "Qwen3-VL-32B-Instruct-GGUF"
    model_path = model_dir / f"Qwen3VL-32B-Instruct-{model_variant}.gguf"
    mmproj_path = model_dir / f"mmproj-Qwen3VL-32B-Instruct-{model_variant}.gguf"

    # Verify files exist
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        print(f"✗ Model not found: {model_path}")
        print(f"\nAvailable models in {model_dir}:")
        for f in model_dir.glob("Qwen3VL-32B-Instruct-*.gguf"):
            print(f"  - {f.name}")
        sys.exit(1)

    if not mmproj_path.exists():
        logger.error(f"MMProj not found: {mmproj_path}")
        print(f"✗ MMProj not found: {mmproj_path}")
        sys.exit(1)

    logger.info(f"Model path: {model_path}")
    logger.info(f"MMProj path: {mmproj_path}")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        # Convert PDF to images
        logger.info("Starting PDF to image conversion...")
        image_files = pdf_to_images(pdf_path, temp_dir, dpi=dpi)
        logger.info(f"Converted {len(image_files)} pages to images")

        # Start llama-server
        logger.info("Starting llama-server...")
        server = LlamaServer(model_path, mmproj_path)
        if not server.start():
            logger.error("Failed to start llama-server")
            print("Failed to start server!")
            sys.exit(1)
        logger.info("llama-server started successfully")

        try:
            # Start building XHTML content
            xhtml_content = []
            xhtml_content.append(create_xhtml_header(pdf_path.stem))

            # Add title
            title = html.escape(pdf_path.stem)
            xhtml_content.append(f"<h1>\n{title}\n</h1>\n\n")
            xhtml_content.append("<p class=\"right\">\n<em>Converted from PDF using Qwen3-VL</em>\n</p>\n\n")
            xhtml_content.append("<hr />\n\n")

            # Process each page
            logger.info(f"Starting OCR processing for {len(image_files)} pages...")
            successful_pages = 0
            failed_pages = 0

            for idx, image_file in enumerate(image_files, start=1):
                try:
                    page_text = extract_text_from_image(server, image_file, idx, logger)

                    # Convert to XHTML paragraphs
                    xhtml_paragraphs = convert_to_xhtml_paragraphs(page_text)

                    if xhtml_paragraphs.strip():
                        xhtml_content.append(xhtml_paragraphs)
                        xhtml_content.append("\n")
                        successful_pages += 1
                    else:
                        logger.warning(f"Page {idx}: Generated empty XHTML content")
                        # Add error marker in output
                        xhtml_content.append(f'<p class="right"><em><strong>Page {idx} failed to process: No text extracted</strong></em></p>\n\n')
                        failed_pages += 1

                except Exception as e:
                    logger.error(f"Page {idx}: Failed to process - {str(e)}")
                    # Add error marker in output with error details
                    error_msg = html.escape(str(e))
                    xhtml_content.append(f'<p class="right"><em><strong>Page {idx} failed to process: {error_msg}</strong></em></p>\n\n')
                    failed_pages += 1
                    # Continue with next page instead of failing completely
                    continue

            logger.info(f"OCR processing completed: {successful_pages} successful, {failed_pages} failed")

            # Add footer
            xhtml_content.append(create_xhtml_footer())

            # Write output
            logger.info(f"Writing XHTML output to: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(''.join(xhtml_content))

            logger.info(f"XHTML file written successfully ({len(''.join(xhtml_content))} bytes)")
            print(f"\n✓ XHTML saved to: {output_path}")

        finally:
            logger.info("Stopping llama-server...")
            server.stop()
            logger.info("llama-server stopped")

    except Exception as e:
        logger.error(f"Fatal error during conversion: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        logger.info("Temporary files cleaned up")
        logger.info("="*60)
        logger.info("PDF to XHTML Conversion Completed")
        logger.info(f"End timestamp: {datetime.now().isoformat()}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to XHTML using Qwen3-VL GGUF model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Q8_0 quantization (default)
  python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml

  # Use Q4_K_M quantization (faster, less VRAM)
  python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --model Q4_K_M

  # Use F16 (full precision, best quality)
  python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --model F16

  # Custom DPI
  python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --dpi 200
        """
    )

    parser.add_argument("input", type=str, help="Input PDF file")
    parser.add_argument("output", type=str, help="Output XHTML file")
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

    pdf_to_xhtml(input_path, output_path, model_variant=args.model, dpi=args.dpi)


if __name__ == "__main__":
    main()
