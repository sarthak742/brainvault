"""
Shared helper for pulling a JSON object out of an LLM response.

LLMs frequently wrap JSON in prose or ```json fences, or emit trailing
commentary. We locate the first balanced {...} block and parse that.
Returns None on any failure so callers can fail-open.
"""
import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Grab the first {...} block (non-greedy across newlines).
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a single JSON object from raw LLM text.

    Returns the parsed dict, or None if nothing parseable was found.
    """
    if not text or not text.strip():
        return None

    # 1. Fast path: the whole thing is valid JSON.
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Slow path: find the first balanced-looking {...} block.
    match = _JSON_BLOCK.search(text)
    if not match:
        logger.debug("extract_json: no JSON block found in response.")
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        logger.debug("extract_json: found a block but it did not parse.")
        return None
