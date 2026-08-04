"""
Image processing module for semantic understanding of images in documents.
Uses Gemini Vision API to extract context from flowcharts, graphs, diagrams, and icons.
"""

import os
import mimetypes
from typing import Optional, Dict, Any
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

SUPPORTED_MIME_TYPES = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif", "image/gif"]


def analyze_image_semantic(image_bytes: bytes, file_extension: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze an image using Gemini Vision API to extract semantic context.
    
    Args:
        image_bytes: Raw image bytes
        file_extension: File extension (e.g., 'png', 'jpg')
        context: Optional context about where the image appears (e.g., "Page 5, Figure 2")
    
    Returns:
        Dictionary with:
        - type: Detected image type (flowchart, graph, diagram, icon, photo, etc.)
        - description: Detailed semantic description
        - caption: Concise caption
        - elements: Key elements detected (for graphs/charts: axes, data points, etc.)
    """
    if not API_KEY:
        return {
            "type": "unknown",
            "description": "[Image detected but no API key configured for semantic analysis]",
            "caption": "Image",
            "elements": []
        }
    
    try:
        ai_client = genai.Client(api_key=API_KEY)
        mimetype = mimetypes.guess_type(f"dummy.{file_extension}")[0]
        
        if mimetype not in SUPPORTED_MIME_TYPES:
            mimetype = "image/png"  # Default fallback
        
        # Build prompt based on context
        context_prompt = f"Context: {context}\n\n" if context else ""
        
        prompt = f"""{context_prompt}Analyze this image in detail for a RAG (Retrieval-Augmented Generation) application. 
Provide a comprehensive semantic description that captures:

1. **Image Type**: Identify if this is a flowchart, graph/chart, diagram, icon/symbol, screenshot, photo, or other type.

2. **Detailed Description**: Describe all visible content including:
   - For flowcharts/diagrams: All shapes, arrows, connections, decision points, processes
   - For graphs/charts: Axes labels, data series, trends, key values, legends
   - For icons/symbols: What the symbol represents, its meaning, and any text labels
   - For screenshots: UI elements, buttons, menus, displayed information
   - For photos: Objects, people, scenes, text if any

3. **Key Elements**: List important elements that should be searchable (e.g., "Revenue increased 20%", "User login flow", "Warning icon indicating security risk")

4. **Concise Caption**: A brief one-sentence summary suitable for search indexing.

Format your response as structured text that preserves all semantic information."""

        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mimetype),
                prompt
            ]
        )
        
        description = response.text
        
        # Parse response to extract structured information
        result = _parse_vision_response(description, context)
        return result
        
    except Exception as e:
        return {
            "type": "error",
            "description": f"[Error analyzing image: {str(e)}]",
            "caption": "Image (analysis failed)",
            "elements": []
        }


def _parse_vision_response(response_text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse Gemini Vision API response to extract structured information.
    """
    text = response_text.lower()
    
    # Detect image type
    image_type = "diagram"
    if any(word in text for word in ["flowchart", "flow chart", "process flow"]):
        image_type = "flowchart"
    elif any(word in text for word in ["graph", "chart", "plot", "bar chart", "line chart", "pie chart"]):
        image_type = "graph"
    elif any(word in text for word in ["icon", "symbol", "logo", "emblem"]):
        image_type = "icon"
    elif any(word in text for word in ["screenshot", "ui", "interface", "screen"]):
        image_type = "screenshot"
    elif any(word in text for word in ["photo", "photograph", "picture"]):
        image_type = "photo"
    elif any(word in text for word in ["diagram", "illustration", "drawing"]):
        image_type = "diagram"
    
    # Extract caption (first sentence or a summary line)
    lines = response_text.split("\n")
    caption = lines[0] if lines else "Image"
    if len(caption) > 150:
        caption = caption[:147] + "..."
    
    # Extract key elements (look for bullet points or numbered lists)
    elements = []
    for line in lines:
        line = line.strip()
        if line.startswith(("-", "•", "*", "1.", "2.", "3.")):
            elements.append(line.lstrip("-•*1234567890. ").strip())
    
    return {
        "type": image_type,
        "description": response_text,
        "caption": caption,
        "elements": elements[:5]  # Top 5 elements
    }


def format_image_description(image_info: Dict[str, Any], page_num: Optional[int] = None, 
                             image_num: Optional[int] = None) -> str:
    """
    Format image analysis results into a text string suitable for embedding.
    
    Args:
        image_info: Result from analyze_image_semantic()
        page_num: Optional page number where image appears
        page_num: Optional image number on the page
    
    Returns:
        Formatted text string with semantic image description
    """
    parts = []
    
    # Header with location info
    location = []
    if page_num is not None:
        location.append(f"Page {page_num}")
    if image_num is not None:
        location.append(f"Image {image_num}")
    
    location_str = ", ".join(location) if location else "Document"
    
    # Build formatted description
    parts.append(f"[{image_info['type'].upper()}] Location: {location_str}")
    
    if image_info.get("caption"):
        parts.append(f"Caption: {image_info['caption']}")
    
    if image_info.get("elements"):
        parts.append("Key Elements:")
        for elem in image_info["elements"]:
            parts.append(f"  - {elem}")
    
    parts.append(f"Detailed Description: {image_info['description']}")
    
    return "\n".join(parts)


def extract_images_from_pdf(file_path: str) -> list[tuple[bytes, str, int, int]]:
    """
    Extract all images from a PDF file.
    
    Returns:
        List of tuples: (image_bytes, extension, page_num, image_num)
    """
    try:
        import pymupdf
        
        doc = pymupdf.open(file_path)
        images = []
        
        for page_num, page in enumerate(doc, start=1):
            img_list = page.get_images()
            for img_num, img in enumerate(img_list, start=1):
                xref = img[0]
                image_data = doc.extract_image(xref)
                img_bytes = image_data["image"]
                img_ext = image_data.get("ext", "png")
                images.append((img_bytes, img_ext, page_num, img_num))
        
        doc.close()
        return images
        
    except ImportError:
        print("Warning: PyMuPDF not available, cannot extract images from PDF")
        return []
    except Exception as e:
        print(f"Warning: Error extracting images from PDF: {e}")
        return []


def extract_images_from_unstructured_elements(elements, source_file: str) -> list[tuple[bytes, str, Optional[int], Optional[int]]]:
    """
    Extract images from unstructured library elements.
    
    Args:
        elements: Elements from unstructured.partition.auto.partition()
        source_file: Path to source document
    
    Returns:
        List of tuples: (image_bytes, extension, page_num, image_num)
    """
    images = []
    
    # Check if elements have image metadata
    for idx, element in enumerate(elements):
        # Check for Image category
        if getattr(element, "category", None) == "Image" or "Image" in element.__class__.__name__:
            # Try to get image path or bytes from metadata
            metadata = getattr(element, "metadata", {}) or {}
            image_path = metadata.get("image_path") or metadata.get("image_base64")
            
            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, "rb") as f:
                        img_bytes = f.read()
                    ext = Path(image_path).suffix.lstrip(".")
                    page_num = metadata.get("page_number")
                    images.append((img_bytes, ext, page_num, idx + 1))
                except Exception as e:
                    print(f"Warning: Could not read image {image_path}: {e}")
    
    # Fallback: If no images found via unstructured, try PyMuPDF for PDFs
    if not images and source_file.lower().endswith(".pdf"):
        images = extract_images_from_pdf(source_file)
    
    return images
