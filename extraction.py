import os
import re
from typing import Iterable, List, Optional

from unstructured.partition.auto import partition
from extractors.images import analyze_image_semantic, format_image_description, extract_images_from_pdf, extract_images_from_unstructured_elements

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def parse(file_path: str) -> List[str]:
    """
    Use the unstructured library to parse a document (pdf/doc/ppt/txt/csv/md/images).
    Returns chunked text suitable for embedding.
    
    Uses separate logic for processing standalone image files and images extracted from documents.
    """
    # Handle standalone image files directly
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.gif'}
    file_ext = os.path.splitext(file_path.lower())[1]
    
    if file_ext in image_extensions:
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            
            image_info = analyze_image_semantic(image_bytes, file_ext.lstrip('.'))
            text = format_image_description(image_info)
            return chunk_splitter(text)
        except Exception as e:
            print(f"Warning: Error processing standalone image: {e}")
            return []
    
    # Process documents with unstructured
    elements = partition(filename=file_path)
    text = elements_to_text(elements, file_path)
    return chunk_splitter(text)


def elements_to_text(elements: Iterable, source_file: Optional[str] = None) -> str:
    """
    Join all element text and minimal metadata to preserve context (e.g., tables, headers).
    Also processes images semantically using vision models.
    """
    processed = []
    
    # Track images we've processed to avoid duplicates
    processed_images = set()
    
    # First, extract and process all images from the document
    image_descriptions = {}
    if source_file:
        try:
            # Try extracting images from unstructured elements first
            images = extract_images_from_unstructured_elements(elements, source_file)
            
            # Process each image semantically
            for img_bytes, ext, page_num, img_num in images:
                # Create unique key to avoid processing same image twice
                img_key = (page_num, img_num)
                if img_key in processed_images:
                    continue
                processed_images.add(img_key)
                
                context = f"Page {page_num}, Image {img_num}" if page_num else f"Image {img_num}"
                print(f"Processing {context} semantically...")
                
                image_info = analyze_image_semantic(img_bytes, ext, context)
                image_descriptions[img_key] = format_image_description(
                    image_info, page_num, img_num
                )
        except Exception as e:
            print(f"Warning: Error processing images: {e}")
    
    # Track which image descriptions we've added
    added_image_descriptions = set()
    
    # Process text elements
    image_counter = {}  # Track image numbers per page
    for element in elements:
        element_category = getattr(element, "category", None) or element.__class__.__name__
        
        # Check if this is an Image element
        if element_category == "Image" or "Image" in element_category:
            # Try to match with processed images
            metadata = getattr(element, "metadata", {}) or {}
            page_num = metadata.get("page_number")
            
            if page_num:
                img_num = image_counter.get(page_num, 0) + 1
                image_counter[page_num] = img_num
                img_key = (page_num, img_num)
                
                if img_key in image_descriptions and img_key not in added_image_descriptions:
                    # Add semantic image description
                    processed.append(image_descriptions[img_key])
                    added_image_descriptions.add(img_key)
                    continue
        
        # Regular text elements
        element_text = getattr(element, "text", "").strip()
        if element_text:
            kind = element_category
            processed.append(f"[{kind}] {element_text}")
    
    # Add any remaining images that weren't matched to Image elements
    for img_key, description in image_descriptions.items():
        if img_key not in added_image_descriptions:
            processed.append(description)
            added_image_descriptions.add(img_key)
    
    return "\n".join(processed)


def chunk_splitter(text: str) -> List[str]:
    """
    Simple sentence-based chunker with overlap to keep context.
    """
    chunks: List[str] = []
    current_chunk = ""

    sentences = sentence_splitter(text)
    chunk_start = 0
    i = 0
    print("Splitting text into chunks....")

    while i < len(sentences):
        if len(current_chunk) + len(sentences[i]) > CHUNK_SIZE and current_chunk != "":
            chunks.append(current_chunk)

            current_chunk = ""
            overlap_size = 0

            for j in range(i - 1, chunk_start - 1, -1):
                overlap_size += len(sentences[j])
                if overlap_size >= CHUNK_OVERLAP:
                    chunk_start = j
                    break
            current_chunk = " ".join(sentences[chunk_start : i + 1]) + " "
            i += 1
            if i < len(sentences):
                current_chunk += sentences[i]
                i += 1
        else:
            current_chunk += sentences[i]
            i += 1
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(f"{len(chunks)} chunks generated\n")
    return chunks


def sentence_splitter(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    sentences = re.split(r"[?!.]", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences