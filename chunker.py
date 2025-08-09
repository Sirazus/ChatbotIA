import os
import logging
import json
import argparse
from typing import List, Dict, Optional

from pypdf import PdfReader
import docx as python_docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Text Extraction Helper Functions ---
# Note: These are duplicated from llm_handling.py to make this a standalone script.
def extract_text_from_file(file_path: str, file_type: str) -> Optional[str]:
    logger.info(f"Extracting text from {file_type.upper()} file: {os.path.basename(file_path)}")
    text_content = None
    try:
        if file_type == 'pdf':
            reader = PdfReader(file_path)
            text_content = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
        elif file_type == 'docx':
            doc = python_docx.Document(file_path)
            text_content = "\n".join(para.text for para in doc.paragraphs if para.text)
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        else:
            logger.warning(f"Unsupported file type for text extraction: {file_type} for file {os.path.basename(file_path)}")
            return None

        if not text_content or not text_content.strip():
            logger.warning(f"No text content extracted from {os.path.basename(file_path)}")
            return None
        return text_content.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {os.path.basename(file_path)} ({file_type.upper()}): {e}", exc_info=True)
        return None

SUPPORTED_EXTENSIONS = {
    'pdf': lambda path: extract_text_from_file(path, 'pdf'),
    'docx': lambda path: extract_text_from_file(path, 'docx'),
    'txt': lambda path: extract_text_from_file(path, 'txt'),
}

def process_sources_and_create_chunks(
    sources_dir: str,
    output_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    text_output_dir: Optional[str] = None  # MODIFIED: Added optional parameter
) -> None:
    """
    Scans a directory for source files, extracts text, splits it into chunks,
    and saves the chunks to a single JSON file.
    Optionally saves the raw extracted text to a specified directory.
    """
    if not os.path.isdir(sources_dir):
        logger.error(f"Source directory not found: '{sources_dir}'")
        raise FileNotFoundError(f"Source directory not found: '{sources_dir}'")

    logger.info(f"Starting chunking process. Sources: '{sources_dir}', Output: '{output_file}'")
    
    # MODIFIED: Create text output directory if provided
    if text_output_dir:
        os.makedirs(text_output_dir, exist_ok=True)
        logger.info(f"Will save raw extracted text to: '{text_output_dir}'")

    all_chunks_for_json: List[Dict] = []
    processed_files_count = 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for filename in os.listdir(sources_dir):
        file_path = os.path.join(sources_dir, filename)
        if not os.path.isfile(file_path):
            continue

        file_ext = filename.split('.')[-1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping unsupported file: {filename}")
            continue

        logger.info(f"Processing source file: {filename}")
        text_content = SUPPORTED_EXTENSIONS[file_ext](file_path)

        if text_content:
            # MODIFIED: Save the raw text to a file if directory is specified
            if text_output_dir:
                try:
                    text_output_path = os.path.join(text_output_dir, f"{filename}.txt")
                    with open(text_output_path, 'w', encoding='utf-8') as f_text:
                        f_text.write(text_content)
                    logger.info(f"Saved extracted text for '{filename}' to '{text_output_path}'")
                except Exception as e_text_save:
                    logger.error(f"Could not save extracted text for '{filename}': {e_text_save}")

            chunks = text_splitter.split_text(text_content)
            if not chunks:
                logger.warning(f"No chunks generated from {filename}. Skipping.")
                continue

            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    "page_content": chunk_text,
                    "metadata": {
                        "source_document_name": filename,
                        "chunk_index": i,
                        "full_location": f"{filename}, Chunk {i+1}"
                    }
                }
                all_chunks_for_json.append(chunk_data)

            processed_files_count += 1
        else:
            logger.warning(f"Could not extract text from {filename}. Skipping.")

    if not all_chunks_for_json:
        logger.warning(f"No processable documents found or no text extracted in '{sources_dir}'. JSON file will be empty.")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_for_json, f, indent=2)

    logger.info(f"Chunking complete. Processed {processed_files_count} files.")
    logger.info(f"Created a total of {len(all_chunks_for_json)} chunks.")
    logger.info(f"Chunked JSON output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process source documents into a JSON file of text chunks for RAG.")
    parser.add_argument(
        '--sources-dir',
        type=str,
        required=True,
        help="The directory containing source files (PDFs, DOCX, TXT)."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help="The full path for the output JSON file containing the chunks."
    )
    # MODIFIED: Added new optional argument
    parser.add_argument(
        '--text-output-dir',
        type=str,
        default=None,
        help="Optional: The directory to save raw extracted text files for debugging."
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help="The character size for each text chunk."
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=150,
        help="The character overlap between consecutive chunks."
    )

    args = parser.parse_args()

    try:
        process_sources_and_create_chunks(
            sources_dir=args.sources_dir,
            output_file=args.output_file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            text_output_dir=args.text_output_dir  # MODIFIED: Pass argument
        )
    except Exception as e:
        logger.critical(f"A critical error occurred during the chunking process: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()