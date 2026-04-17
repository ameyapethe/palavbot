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
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.2,
        help=(
            "Max tolerated fraction of source URLs that may fail to ingest "
            "(0.0-1.0). Exits non-zero if exceeded, so a flaky scheduled "
            "rebuild won't ship a degraded index over a healthy one."
        ),
    )
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
    total = bundle.report.get("total_urls", 0)
    failed = len(bundle.report.get("failed", []))
    print(
        f"Index ready: chunks={len(bundle.chunks)} "
        f"ok={bundle.report.get('ok', 0)} "
        f"failed={failed} "
        f"from_cache={bundle.loaded_from_cache}"
    )

    if not bundle.loaded_from_cache and total > 0:
        fail_ratio = failed / total
        if fail_ratio > args.fail_threshold:
            for item in bundle.report.get("failed", []):
                print(f"  FAILED: {item.get('url')} -> {item.get('error')}", file=sys.stderr)
            print(
                f"ERROR: source failure rate {fail_ratio:.0%} exceeds "
                f"threshold {args.fail_threshold:.0%}; aborting so a "
                f"degraded index does not replace the current one.",
                file=sys.stderr,
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
