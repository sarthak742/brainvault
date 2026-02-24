import logging
import sys
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, TypedDict, Optional

import pypdf
from PIL import Image
import io

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_project_root, get_sarvam_api_key, is_sarvam_ocr_enabled, get

# Try to import pdf2image for PDF to image conversion
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("pdf2image not installed. Sarvam OCR may not work properly.")

# Try to import Sarvam AI SDK
SARVAM_AVAILABLE = False
try:
    from sarvamai import SarvamAI
    SARVAM_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("sarvamai not installed. Sarvam OCR disabled.")

# --- Type Definitions ---
class PageRecord(TypedDict):
    source: str       # Relative path from project root
    page: Optional[int]
    text: str

# --- Module-Level Logger (Configured by consumer, not here) ---
logger = logging.getLogger(__name__)


def _to_relative_path(file_path: Path) -> str:
    """
    Convert absolute path to relative path from project root.
    """
    try:
        # Try to make it relative to project root
        rel_path = file_path.resolve().relative_to(get_project_root().resolve())
        return str(rel_path)
    except ValueError:
        # If not under project root, return as-is but try to make it portable
        return str(file_path)


# --- Core Functions ---

def load_documents(data_root: Path) -> List[Path]:
    """
    Recursively discovers PDF, TXT, and MD files in a directory.
    """
    if not data_root.exists():
        logger.error(f"Data root directory does not exist: {data_root}")
        return []

    files = []
    # Explicitly supported extensions
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    for path_obj in data_root.rglob("*"):
        if path_obj.is_file():
            if path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path_obj)

    if not files:
        logger.warning(f"No supported files found in {data_root}")
        return []

    files.sort()
    return files


def extract_text_from_txt(path: Path) -> List[PageRecord]:
    """Extracts raw text from a .txt file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        logger.exception(f"Failed to read text file: {path}")
        return []

    if not text:
        logger.warning(f"File is empty: {path}")

    return [{
        "source": _to_relative_path(path),
        "page": None,
        "text": text
    }]


def extract_text_from_markdown(path: Path) -> List[PageRecord]:
    """
    Extracts raw text from a .md file.
    Currently identical to .txt, but separated for future parsing logic (headers, frontmatter).
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        logger.exception(f"Failed to read markdown file: {path}")
        return []

    if not text:
        logger.warning(f"File is empty: {path}")

    return [{
        "source": _to_relative_path(path),
        "page": None,
        "text": text
    }]


def _get_sarvam_client():
    """
    Initialize and return the Sarvam AI client.
    """
    if not SARVAM_AVAILABLE:
        return None

    api_key = get_sarvam_api_key()
    if not api_key:
        logger.warning("SARVAM_API_KEY not set. Sarvam OCR disabled.")
        return None

    try:
        client = SarvamAI(api_subscription_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to configure Sarvam AI client: {e}")
        return None


def _extract_text_with_sarvam_from_pdf(file_path: Path) -> List[str]:
    """
    Extract text from PDF using Sarvam AI Vision API.
    Returns a list of extracted texts, one per page.
    """
    if not is_sarvam_ocr_enabled():
        logger.debug("Sarvam OCR is not enabled")
        return []

    client = _get_sarvam_client()
    if client is None:
        logger.warning("Sarvam AI client not available")
        return []

    try:
        # Get language from config
        language = get('sarvam_language', 'en-IN')

        logger.info(f"Starting Sarvam OCR for {file_path.name}...")

        # Create document intelligence job
        job = client.document_intelligence.create_job(
            language=language,
            output_format="md"  # Get markdown output
        )
        logger.info(f"Job created: {job.job_id}")

        # Upload the PDF file
        job.upload_file(str(file_path))
        logger.info("File uploaded")

        # Start processing
        job.start()
        logger.info("Job started")

        # Wait for completion
        logger.info("Waiting for job to complete...")
        status = job.wait_until_complete()
        logger.info(f"Job completed with state: {status.job_state}")

        if status.job_state.lower() != "completed":
            logger.warning(f"Job did not complete successfully: {status.job_state}")
            return []

        # Get page metrics
        metrics = job.get_page_metrics()
        logger.info(f"Page metrics: {metrics}")

        # Download output to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        job.download_output(tmp_path)
        logger.info(f"Output downloaded to {tmp_path}")

        # Extract the markdown content from the ZIP file
        page_texts = []
        with zipfile.ZipFile(tmp_path, 'r') as zf:
            # List all files in the ZIP
            file_list = zf.namelist()
            logger.info(f"Files in ZIP: {file_list}")

            # Read each markdown file (pages are usually named 1.md, 2.md, etc.)
            for i in range(1, len(file_list) + 1):
                page_file = f"{i}.md"
                if page_file in zf.namelist():
                    with zf.open(page_file) as f:
                        content = f.read().decode('utf-8')
                        page_texts.append(content)
                        logger.info(f"Extracted page {i}: {len(content)} chars")
                else:
                    # If no file found, try other patterns
                    for name in zf.namelist():
                        if name.endswith('.md'):
                            with zf.open(name) as f:
                                content = f.read().decode('utf-8')
                                page_texts.append(content)

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except:
            pass

        return page_texts

    except Exception as e:
        logger.error(f"Sarvam OCR processing failed for {file_path}: {e}")
        return []


def extract_text_from_pdf(file_path: Path, use_sarvam_fallback: bool = True) -> List[PageRecord]:
    """
    Extracts text from a PDF file page by page.
    Uses pypdf for text extraction first, then falls back to Sarvam Vision API if enabled
    and no text was extracted (common for scanned PDFs).

    Args:
        file_path: Path to the PDF file
        use_sarvam_fallback: If True, attempt Sarvam OCR when no text is extracted
    """
    records: List[PageRecord] = []

    try:
        reader = pypdf.PdfReader(str(file_path), strict=False)
    except Exception:
        logger.exception(f"Failed to open PDF file: {file_path}")
        return []

    # First pass: try standard text extraction
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        try:
            raw_text = page.extract_text() or ""
            records.append({
                "source": _to_relative_path(file_path),
                "page": page_num,
                "text": raw_text
            })
        except Exception:
            logger.exception(
                f"CRASH during extraction in {file_path} on Page {page_num}. Skipping page."
            )
            records.append({
                "source": _to_relative_path(file_path),
                "page": page_num,
                "text": ""
            })

    # Check if we need Sarvam OCR fallback
    if use_sarvam_fallback and is_sarvam_ocr_enabled():
        # Check if any pages have empty text
        total_chars = sum(len(r['text']) for r in records)

        if total_chars < 50:  # Very little text extracted, likely scanned
            logger.info(f"Low text extraction from {file_path}. Attempting Sarvam OCR...")
            sarvam_texts = _extract_text_with_sarvam_from_pdf(file_path)

            if sarvam_texts:
                # Update records with Sarvam OCR text
                for i, sarvam_text in enumerate(sarvam_texts):
                    if i < len(records):
                        if not records[i]['text'].strip() and sarvam_text.strip():
                            records[i]['text'] = sarvam_text
                            logger.info(f"Sarvam OCR successful for page {i+1}: {len(sarvam_text)} chars")

    # Log any pages that still have no text
    for record in records:
        if not record['text'].strip():
            logger.warning(
                f"Empty extraction in {record['source']} on Page {record['page']}. "
                f"No text could be extracted (may be a scanned image)."
            )

    return records
