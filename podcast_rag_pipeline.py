"""Compatibility entry point for the Podcast RAG pipeline.

The implementation now lives under ``src/podcast_rag``. Keeping this wrapper
preserves existing commands like ``python podcast_rag_pipeline.py`` while the
repository adopts a standard package layout.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from podcast_rag.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
