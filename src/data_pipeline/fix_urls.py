"""
URL Fixer — JoshTalks ASR Research
====================================
Corrects broken dataset URLs by replacing the storage prefix
with the working upload_goai format.

URL Pattern:
    https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_recording.wav
    https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json
    https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_metadata.json
"""

import re
import requests
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger

logger = setup_logger("fix_urls", log_to_file=False)

# The correct base URL prefix
CORRECT_BASE_URL = "https://storage.googleapis.com/upload_goai"


def fix_url(url: str) -> str:
    """
    Fix a broken dataset URL by replacing the storage prefix.

    Args:
        url: Original (possibly broken) URL.

    Returns:
        Corrected URL with upload_goai prefix.
    """
    if not url or not isinstance(url, str):
        return url

    # If already correct, return as-is
    if CORRECT_BASE_URL in url:
        return url

    # Extract the path component (user_id/recording_id_suffix)
    # Try to parse out the meaningful parts
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # Reconstruct with correct base
    fixed = f"{CORRECT_BASE_URL}/{path}"
    return fixed


def build_urls(user_id: str, recording_id: str) -> Dict[str, str]:
    """
    Build all three URLs for a recording from user_id and recording_id.

    Args:
        user_id: Speaker/user identifier.
        recording_id: Unique recording identifier.

    Returns:
        Dictionary with 'recording', 'transcription', 'metadata' URLs.
    """
    base = f"{CORRECT_BASE_URL}/{user_id}/{recording_id}"
    return {
        "recording": f"{base}_recording.wav",
        "transcription": f"{base}_transcription.json",
        "metadata": f"{base}_metadata.json",
    }


def validate_url(url: str, timeout: int = 10) -> bool:
    """
    Check if a URL is accessible via HEAD request.

    Args:
        url: URL to validate.
        timeout: Request timeout in seconds.

    Returns:
        True if URL returns 200 status.
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.warning(f"URL validation failed for {url}: {e}")
        return False


def fix_dataset_urls(
    records: list,
    url_fields: tuple = ("rec_url_gcp", "transcription_url", "metadata_url"),
    validate: bool = False,
) -> list:
    """
    Fix URLs in an entire dataset of records.

    Args:
        records: List of record dictionaries.
        url_fields: Tuple of field names containing URLs.
        validate: Whether to validate each fixed URL.

    Returns:
        Records with corrected URLs.
    """
    fixed_count = 0
    invalid_count = 0

    for record in records:
        user_id = record.get("user_id", "")
        recording_id = record.get("recording_id", "")

        if user_id and recording_id:
            urls = build_urls(str(user_id), str(recording_id))
            record["rec_url_gcp"] = urls["recording"]
            record["transcription_url"] = urls["transcription"]
            record["metadata_url"] = urls["metadata"]
            fixed_count += 1
        else:
            # Fallback: fix individual URL fields
            for field in url_fields:
                if field in record and record[field]:
                    record[field] = fix_url(record[field])
                    fixed_count += 1

        if validate:
            for field in url_fields:
                if field in record and not validate_url(record[field]):
                    invalid_count += 1
                    logger.warning(f"Invalid URL after fix: {record[field]}")

    logger.info(f"Fixed {fixed_count} URLs, {invalid_count} invalid after fix")
    return records


if __name__ == "__main__":
    # Quick test
    test_urls = build_urls("967179", "825780")
    print("Generated URLs:")
    for key, url in test_urls.items():
        print(f"  {key}: {url}")

    # Validate one
    print(f"\nValidating recording URL: {validate_url(test_urls['recording'])}")
