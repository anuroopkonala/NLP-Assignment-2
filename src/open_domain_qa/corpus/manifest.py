"""
Corpus manifest generation and cryptographic integrity verification.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()


def generate_corpus_manifest(
    corpus_path: str,
    output_manifest_path: str = "data/wikipedia/corpus_manifest.json",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates and writes a cryptographic manifest for the corpus file."""
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    sha = compute_sha256(corpus_path) if os.path.exists(corpus_path) else ""

    manifest = {
        "dataset": "google-research-datasets/nq_open",
        "corpus_file": corpus_path,
        "corpus_sha256": sha,
        "metadata": metadata or {},
    }

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_corpus_manifest(manifest_path: str, base_dir: str = ".") -> Tuple[bool, Dict[str, bool]]:
    """Verifies that files listed in the manifest exist and match expected SHA-256 hashes."""
    if not os.path.exists(manifest_path):
        return False, {"manifest_exists": False}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_manifest = manifest.get("file_manifest", manifest)
    results = {}
    all_ok = True

    c_path = os.path.join(base_dir, file_manifest.get("corpus_file", ""))
    expected_c_hash = file_manifest.get("corpus_sha256", "")
    if os.path.exists(c_path) and compute_sha256(c_path) == expected_c_hash:
        results["corpus_valid"] = True
    else:
        results["corpus_valid"] = False
        all_ok = False

    if "questions_file" in file_manifest:
        q_path = os.path.join(base_dir, file_manifest.get("questions_file", ""))
        expected_q_hash = file_manifest.get("questions_sha256", "")
        if os.path.exists(q_path) and compute_sha256(q_path) == expected_q_hash:
            results["questions_valid"] = True
        else:
            results["questions_valid"] = False
            all_ok = False

    return all_ok, results
