#!/usr/bin/env python3
"""Test OCR on a single page"""

import sys
from pathlib import Path

# Import from the main script
from pdf_to_markdown_llamacpp import LlamaServer, pdf_to_images

def test_single_page(pdf_path: Path, page_num: int = 1):
    """Test extracting just one page"""
    import tempfile
    import shutil

    # Determine model paths
    model_dir = Path.home() / "models" / "Qwen3-VL-32B-Instruct-GGUF"
    model_path = model_dir / "Qwen3VL-32B-Instruct-Q8_0.gguf"
    mmproj_path = model_dir / "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf"

    if not model_path.exists() or not mmproj_path.exists():
        print(f"Model files not found!")
        return

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Convert PDF to images
        print(f"Converting page {page_num} to image...")
        image_files = pdf_to_images(pdf_path, temp_dir, dpi=150)

        if page_num > len(image_files):
            print(f"Only {len(image_files)} pages available")
            return

        test_image = image_files[page_num - 1]
        print(f"Testing with: {test_image}")

        # Start server
        server = LlamaServer(model_path, mmproj_path)
        if not server.start():
            print("Failed to start server!")
            return

        try:
            # Test extraction
            prompt = """Perform OCR on this image. Extract ALL visible text exactly as it appears.

Preserve the original formatting, structure, headings, lists, and tables using markdown syntax.
Do NOT add any commentary, explanations, or content that is not visible in the image.
Only output the text you can actually read from the image."""

            print("\n" + "="*60)
            print(f"Extracting text from page {page_num}...")
            print("="*60 + "\n")

            result = server.generate(prompt, test_image, max_tokens=4096)

            print("RESULT:")
            print("-" * 60)
            print(result)
            print("-" * 60)

        finally:
            server.stop()

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_single_page.py <pdf_path> [page_num]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    test_single_page(pdf_path, page_num)
