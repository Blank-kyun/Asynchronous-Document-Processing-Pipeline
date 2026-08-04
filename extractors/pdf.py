import pymupdf
import re
import os
from images import generate_image_description

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_CHUNKS__TO_RETRIEVE = 5

def extract_content_from_pdf(file_address: str) -> str:
    """
    Extracts text from a PDF document using PyMuPDF.
    Uses 'text' extraction method for general content.
    """
    doc = pymupdf.open(file_address)
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text() + '\n\n'
        
        img_list = page.get_images()
        img_context = ""
        for img_num, img in enumerate(img_list):
            xref = img[0]
            image = doc.extract_image(xref)
            
            img_bytes = image["image"]
            img_extension = image['ext']

            img_context = f"Page {page_num + 1}, Image {img_num + 1}, Content: "
            img_context += generate_image_description(img_bytes, img_extension) +'\n\n'

        text += page_text + img_context

    print(f"Extracted text from pdf: {len(text)} characters detected..")

    doc.close()
    return text.strip()

def chunk_splitter(text) -> list[str]:
    chunks = []
    current_chunk = ""

    sentences = sentence_splitter(text)
    chunk_start = 0
    i = 0
    print("Splitting text into chunks....")

    while(i < len(sentences)):
        if len(current_chunk) + len(sentences[i]) > CHUNK_SIZE and current_chunk != "":
            chunks.append(current_chunk)

            current_chunk = ""
            overlap_size = 0

            for j in range(i-1,chunk_start-1,-1):
                overlap_size += len(sentences[j])
                if overlap_size >= CHUNK_OVERLAP:
                    chunk_start = j
                    break
            current_chunk = " ".join(sentences[chunk_start:i+1]) + " "
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

def sentence_splitter(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text) # Replaces whitespace with single space

    # Splits text at instances of a sentence ending - a ".", "?" or a "!"
    # followed by whitespace and a capitalized letter
    sentences = re.split(r"[?!.]", text)
    sentences = [sentence.strip() for sentence in sentences]

    return sentences

def answer_question(ai_client, question, context) -> str:
    response = ai_client.models.generate_content(
        model = "gemini-2.5-flash",
        config = types.GenerateContentConfig(
            system_instruction = "Your job is to answer questions related to context objectively as part of a RAG.Answer question in concise human-readable plain-text. Augment the context with your own knowldege base:\n" + context),
        contents = question
    )

    return response.text