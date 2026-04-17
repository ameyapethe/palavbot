"""Build the FAISS index ahead of container image build / local first-run.

Run this once before `docker build` (in CI) or after changing
`palav_url_links.txt`. Requires OPENAI_API_KEY for embedding generation;
results are deterministic for a given links file + settings hash, so reruns
are skipped when a matching cache already exists.
"""

from __future__ import annotations

import argparse
import os
import sys

from palav.retrieval import DEFAULT_INDEX_DIR, DEFAULT_LINKS_FILE, build_or_load


def main() -> int:
    parser = argparse.ArgumentParser(description="Build palavbot FAISS index cache")
    parser.add_argument("--links-file", default=DEFAULT_LINKS_FILE)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required to build the index.", file=sys.stderr)
        return 2

    bundle = build_or_load(
        links_file=args.links_file,
        api_key=api_key,
        index_dir=args.index_dir,
        force_rebuild=args.force,
    )
    print(
        f"Index ready: chunks={len(bundle.chunks)} "
        f"ok={bundle.report.get('ok', 0)} "
        f"failed={len(bundle.report.get('failed', []))} "
        f"from_cache={bundle.loaded_from_cache}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
