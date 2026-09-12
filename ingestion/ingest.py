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

from config import get_project_root, get_sarvam_api_key, is_sarvam_ocr_enabled, get, get_ocr_engine

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


def _extract_text_with_tesseract_from_pdf(file_path: Path) -> List[str]:
    """
    Rasterise each page and OCR it locally with Tesseract.

    Free, offline, no API key. Weaker than a hosted vision model on messy
    handwriting, but it costs nothing per page, which matters when you are
    re-ingesting the same corpus repeatedly while tuning.

    Returns a list of page texts, index-aligned with the PDF's pages.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning("pytesseract/pdf2image not installed. Tesseract OCR unavailable.")
        return []

    dpi = get('ocr_dpi', 200)
    try:
        logger.info(f"Tesseract OCR starting for {file_path.name} at {dpi} DPI...")
        images = convert_from_path(str(file_path), dpi=dpi)
    except Exception as e:
        logger.error(f"Could not rasterise {file_path.name}: {e}")
        return []

    texts = []
    for i, img in enumerate(images, start=1):
        try:
            # osd auto-rotates pages photographed sideways. Without this, the
            # rotated page in a scanned notebook OCRs to noise.
            osd_img = img
            try:
                osd = pytesseract.image_to_osd(img)
                rot = int([l for l in osd.splitlines() if "Rotate:" in l][0].split(":")[1])
                if rot:
                    osd_img = img.rotate(-rot, expand=True)
                    logger.info(f"Page {i}: auto-rotated {rot} degrees")
            except Exception:
                pass  # OSD fails on sparse pages; fall back to original

            text = pytesseract.image_to_string(osd_img)
            texts.append(text)
            logger.info(f"Tesseract page {i}: {len(text.strip())} chars")
        except Exception as e:
            logger.error(f"Tesseract failed on page {i}: {e}")
            texts.append("")

    return texts


def _extract_text_with_nvidia_vlm(file_path: Path, pages: List[int] = None) -> List[str]:
    """
    OCR pages using a vision-language model on NVIDIA NIM.

    Why a VLM rather than Tesseract: Tesseract matches glyph shapes and has no
    language model, so cursive handwriting photographed at an angle produces
    either nothing or noise. A VLM reads in context -- it can infer
    "insulation resistance" from ambiguous strokes because the surrounding
    words make it the only sensible reading.

    Args:
        pages: zero-based page indices to OCR. None means all pages.
               Passing only the pages that need it roughly halves the request
               count on a typical mixed corpus.

    Returns a list index-aligned with the PDF's pages; pages that were not
    requested come back as empty strings.
    """
    import base64, io, requests
    from config import get_llm_api_key, get

    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning("pdf2image not installed. NVIDIA VLM OCR unavailable.")
        return []

    key = get_llm_api_key()
    if not key:
        logger.warning("No NVIDIA API key set. Skipping VLM OCR.")
        return []

    model = get('ocr_vlm_model', 'meta/llama-3.2-11b-vision-instruct')
    url = get('ocr_vlm_url', 'https://integrate.api.nvidia.com/v1/chat/completions')
    dpi = get('ocr_dpi', 200)

    prompt = (
        "Transcribe all handwritten and printed text in this page image "
        "exactly as written. Preserve tables as rows. Include labels on any "
        "diagrams. Output plain text only: no markdown, no bold, no headings, "
        "no commentary, no preamble."
    )

    try:
        images = convert_from_path(str(file_path), dpi=dpi)
    except Exception as e:
        logger.error(f"Could not rasterise {file_path.name}: {e}")
        return []

    texts = []
    wanted = set(range(len(images))) if pages is None else set(pages)
    skipped = 0
    for i, img in enumerate(images, start=1):
        if (i - 1) not in wanted:
            texts.append("")
            skipped += 1
            continue
        try:
            # Downscale: VLMs cap input resolution and large payloads get rejected.
            img.thumbnail((1600, 1600))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = requests.post(
                url,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}],
                    "max_tokens": 2048,
                    "temperature": 0.0,
                },
                timeout=180,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            texts.append(text)
            logger.info(f"VLM OCR page {i}: {len(text.strip())} chars")
        except Exception as e:
            logger.error(f"VLM OCR failed on page {i}: {str(e)[:160]}")
            texts.append("")

    if skipped:
        logger.info(f"Skipped {skipped} pages that already had extractable text.")

    return texts


def _run_ocr(file_path: Path, pages: List[int] = None) -> List[str]:
    """Dispatch to the OCR engine named in config.yaml."""
    engine = get_ocr_engine()
    if engine == "sarvam":
        if is_sarvam_ocr_enabled():
            return _extract_text_with_sarvam_from_pdf(file_path)
        logger.warning("ocr_engine is 'sarvam' but no key is set. Skipping OCR.")
        return []
    if engine == "nvidia":
        return _extract_text_with_nvidia_vlm(file_path, pages=pages)
    return _extract_text_with_tesseract_from_pdf(file_path)


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

    # Check if we need an OCR fallback
    if use_sarvam_fallback and get_ocr_engine():
        # Decide PER PAGE, not on the document total.
        #
        # Previously this summed chars across the whole document and skipped OCR
        # if the total exceeded 50. That meant a single page with a text
        # watermark could disable OCR for every other page in the file, and
        # slide decks with extractable titles but image-only diagrams were
        # silently ingested with their real content missing.
        MIN_CHARS_PER_PAGE = 100
        sparse_pages = [
            i for i, r in enumerate(records)
            if len(r['text'].strip()) < MIN_CHARS_PER_PAGE
        ]

        if sparse_pages:
            logger.info(
                f"{len(sparse_pages)}/{len(records)} pages below "
                f"{MIN_CHARS_PER_PAGE} chars in {file_path.name}. Running OCR "
                f"({get_ocr_engine()})..."
            )
            ocr_texts = _run_ocr(file_path, pages=sparse_pages)

            if ocr_texts:
                # Only overwrite the pages that were actually text-poor.
                for i in sparse_pages:
                    if i < len(ocr_texts) and ocr_texts[i].strip():
                        records[i]['text'] = ocr_texts[i]
                        logger.info(
                            f"OCR recovered page {i+1}: {len(ocr_texts[i])} chars"
                        )

    # Log any pages that still have no text
    for record in records:
        if not record['text'].strip():
            logger.warning(
                f"Empty extraction in {record['source']} on Page {record['page']}. "
                f"No text could be extracted (may be a scanned image)."
            )

    return records
