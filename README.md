# PDF OCR Conversion Tools

Convert PDF documents to Markdown or XHTML format using Qwen3-VL vision model via llama.cpp.

## Features

- Uses local Qwen3-VL GGUF models (no internet required after initial setup)
- Supports multiple quantization levels (Q4_K_M, Q8_0, F16)
- Runs on AMD GPU with ROCm acceleration
- Preserves document structure, headings, lists, and tables
- Configurable DPI for image quality
- Two output formats:
  - **Markdown**: Clean, readable format for documentation
  - **XHTML**: Properly formatted HTML with CSS styling for ebooks
- Comprehensive logging for XHTML conversion:
  - Automatic log file creation (`.log` file alongside output)
  - Per-page processing status and statistics
  - Error tracking with full tracebacks
  - Inline error markers in output for failed pages

## Prerequisites

### System Requirements

- Docker container with llama.cpp installed
- AMD GPU with ROCm support (or CUDA for NVIDIA GPUs)
- Python 3.10+

### Model Files

The script expects Qwen3-VL models in:
```
~/models/Qwen3-VL-32B-Instruct-GGUF/
├── Qwen3VL-32B-Instruct-Q4_K_M.gguf
├── Qwen3VL-32B-Instruct-Q8_0.gguf
├── Qwen3VL-32B-Instruct-F16-split-00001-of-00002.gguf
├── Qwen3VL-32B-Instruct-F16-split-00002-of-00002.gguf
├── mmproj-Qwen3VL-32B-Instruct-Q4_K_M.gguf
├── mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf
└── mmproj-Qwen3VL-32B-Instruct-F16.gguf
```

## Installation

Inside your Docker container:

```bash
# Install Python dependencies
uv pip install --upgrade pymupdf pillow requests
```

## Usage

### Basic Usage

```bash
# Convert PDF to Markdown with Q8_0 quantization (default)
python3 pdf_to_markdown_llamacpp.py input.pdf output.md

# Convert PDF to XHTML with Q8_0 quantization (default)
python3 pdf_to_xhtml_llamacpp.py input.pdf output.xhtml
```

### Advanced Options

```bash
# Use Q4_K_M quantization (faster, less VRAM)
python3 pdf_to_markdown_llamacpp.py document.pdf output.md --model Q4_K_M
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --model Q4_K_M

# Use F16 quantization (best quality, more VRAM)
python3 pdf_to_markdown_llamacpp.py document.pdf output.md --model F16
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --model F16

# Increase DPI for better OCR quality
python3 pdf_to_markdown_llamacpp.py document.pdf output.md --dpi 200
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --dpi 200

# Lower DPI for faster processing
python3 pdf_to_markdown_llamacpp.py document.pdf output.md --dpi 100
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --dpi 100

# Extract specific pages only (XHTML conversion)
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --pages "1-5"
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --pages "1,3,5,10"
python3 pdf_to_xhtml_llamacpp.py document.pdf output.xhtml --pages "1-3,7,10-12"
```

### Complete Example

```bash
# Enter your Docker container (example)
docker exec -it <your-llama-container> /bin/bash

# Convert a PDF document to Markdown
python3 pdf_to_markdown_llamacpp.py /path/to/book.pdf /path/to/output.md --model Q8_0 --dpi 150

# Convert a PDF document to XHTML (for ebooks)
python3 pdf_to_xhtml_llamacpp.py /path/to/book.pdf /path/to/output.xhtml --model Q8_0 --dpi 150
```

## Model Quantization Comparison

| Quantization | VRAM Usage | Quality | Speed | Recommended For |
|--------------|------------|---------|-------|-----------------|
| Q4_K_M       | ~10 GB     | Good    | Fast  | Quick conversions, limited VRAM |
| Q8_0         | ~17 GB     | Better  | Medium| Balanced quality/performance |
| F16          | ~33 GB     | Best    | Slower| Maximum quality, ample VRAM |

## How It Works

1. **PDF Conversion**: Converts each PDF page to PNG images at specified DPI using PyMuPDF
2. **Server Startup**: Launches llama-server with Qwen3-VL model and mmproj file
3. **OCR Processing**: Sends each image to the vision model via OpenAI-compatible API (`/v1/chat/completions`)
4. **Text Extraction**: Vision model performs OCR and returns structured text
5. **Format Generation**:
   - **Markdown**: Combines extracted text into a single markdown document
   - **XHTML**: Generates well-formed XHTML with proper CSS styling, paragraph structure, and HTML entity encoding
6. **Logging** (XHTML only): Comprehensive logging to `.log` file with timestamps, per-page status, and error details
7. **Cleanup**: Stops llama-server and removes temporary image files

### Technical Details

The script uses llama.cpp's OpenAI-compatible chat completions API with vision support:
- Images are encoded as base64 data URLs
- Sent via `/v1/chat/completions` endpoint with `image_url` content type
- Custom OCR prompt instructs model to extract visible text only
- Temperature set to 0.1 for consistent, deterministic output

### XHTML Logging

The XHTML converter automatically creates a detailed log file alongside the output:

**Log file location**: Same directory as output file, with `.log` extension (e.g., `output.xhtml` → `output.log`)

**Log contents include**:
- Start/end timestamps
- Input/output file paths and conversion parameters
- Model and MMProj file paths
- PDF to image conversion progress
- llama-server lifecycle events (start/stop)
- Per-page OCR status:
  - Processing start time
  - Number of characters extracted
  - Success or failure status
  - Detailed error messages with tracebacks
- Final statistics: successful vs failed pages
- Output file size

**Error handling**:
- If a page fails to process, an error marker is inserted into the XHTML output at that location
- Example: `Page 5 failed to process: Connection timeout`
- The conversion continues with remaining pages instead of aborting
- All errors are logged with full details in the log file

## Troubleshooting

### Server won't start

- Check if port 8080 is already in use
- Verify model files exist in the expected location
- Check VRAM availability matches quantization level

### Out of memory errors

- Try a smaller quantization (Q4_K_M instead of Q8_0)
- Reduce DPI (e.g., --dpi 100)
- Close other GPU-intensive applications

### Poor OCR quality

- Increase DPI (e.g., --dpi 200 or 300)
- Use better quantization (Q8_0 or F16)
- Ensure source PDF has good image quality

### Model hallucinating / generating random text

If the model generates unrelated content instead of extracting text from the PDF:
- Verify you're using the latest version of the script (uses `/v1/chat/completions` API)
- Check that llama-server supports vision models (version with multimodal support)
- Ensure mmproj file matches the main model quantization

### Model not found error

The script will list available models. Ensure you have:
- Main model file: `Qwen3VL-32B-Instruct-{variant}.gguf`
- MMProj file: `mmproj-Qwen3VL-32B-Instruct-{variant}.gguf`

Both files must use the same quantization variant (Q4_K_M, Q8_0, or F16).

## Files

- `pdf_to_markdown_llamacpp.py` - PDF to Markdown conversion script
- `pdf_to_xhtml_llamacpp.py` - PDF to XHTML conversion script (for ebooks)
- `test_single_page.py` - Test script for debugging single page extraction
- `debug_vision.py` - Low-level vision API testing tool
- `README.md` - This file

## Command-Line Options

### Markdown Conversion

```
usage: pdf_to_markdown_llamacpp.py [-h] [--model {Q4_K_M,Q8_0,F16}] [--dpi DPI] input output

Convert PDF to Markdown using Qwen3-VL GGUF model

positional arguments:
  input                 Input PDF file
  output                Output markdown file

optional arguments:
  -h, --help            show this help message and exit
  --model {Q4_K_M,Q8_0,F16}
                        Model quantization variant (default: Q8_0)
  --dpi DPI             DPI for PDF conversion (default: 150)
```

### XHTML Conversion

```
usage: pdf_to_xhtml_llamacpp.py [-h] [--model {Q4_K_M,Q8_0,F16}] [--dpi DPI] [--pages PAGES] input output

Convert PDF to XHTML using Qwen3-VL GGUF model

positional arguments:
  input                 Input PDF file
  output                Output XHTML file

optional arguments:
  -h, --help            show this help message and exit
  --model {Q4_K_M,Q8_0,F16}
                        Model quantization variant (default: Q8_0)
  --dpi DPI             DPI for PDF conversion (default: 150)
  --pages PAGES         Pages to extract (e.g., "1-5", "1,3,5", "1-3,7,10-12"). Default: all pages
```

## License

MIT License
