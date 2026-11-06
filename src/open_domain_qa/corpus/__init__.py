"""
Wikipedia Passage Corpus Management and Manifest Verification
"""

from .wikipedia import WikipediaCorpus
from .segmentation import truncate_tokens
from .manifest import verify_corpus_manifest

__all__ = ["WikipediaCorpus", "truncate_tokens", "verify_corpus_manifest"]
