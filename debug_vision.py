#!/usr/bin/env python3
"""
Debug script to test llama-server vision capabilities
"""

import sys
import base64
import json
import requests
from pathlib import Path

def test_vision(image_path: Path, port: int = 8080):
    """Test vision model with simple prompt"""

    base_url = f"http://localhost:{port}"

    # Check if server is running
    try:
        health = requests.get(f"{base_url}/health", timeout=2)
        print(f"✓ Server is running: {health.status_code}")
    except Exception as e:
        print(f"✗ Server not running: {e}")
        return

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    print(f"\nImage encoded: {len(image_data)} bytes")

    # Test 1: Simple prompt
    print("\n" + "="*60)
    print("TEST 1: Simple description")
    print("="*60)

    payload1 = {
        "prompt": "Describe what you see in this image.",
        "n_predict": 200,
        "temperature": 0.7,
        "image_data": [{"data": image_data, "id": 1}],
        "cache_prompt": False
    }

    try:
        response = requests.post(
            f"{base_url}/completion",
            json=payload1,
            timeout=120
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\nGenerated text:\n{result.get('content', 'NO CONTENT')}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: OCR-specific prompt
    print("\n" + "="*60)
    print("TEST 2: OCR prompt")
    print("="*60)

    payload2 = {
        "prompt": "What text is visible in this image? Extract all readable text.",
        "n_predict": 500,
        "temperature": 0.1,
        "image_data": [{"data": image_data, "id": 1}],
        "cache_prompt": False
    }

    try:
        response = requests.post(
            f"{base_url}/completion",
            json=payload1,
            timeout=120
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"\nGenerated text:\n{result.get('content', 'NO CONTENT')}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_vision.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    test_vision(image_path)
