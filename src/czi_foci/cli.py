"""Command line interface for generic CZI nuclear foci analysis."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .analysis import analyse_files
from .config import load_config
from .io import selected_files_from_manifest, selected_files_from_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="czi-foci", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="Segment nuclei, two focus channels, and co-localised foci")
    source = analyse.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Root folder containing CZI files")
    source.add_argument("--manifest", type=Path, help="CSV manifest with a source_path column")
    analyse.add_argument("--config", type=Path, required=True, help="JSON analysis configuration")
    analyse.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    analyse.add_argument("--qc-title", default="CZI nuclear foci QC")
    analyse.add_argument("--limit", type=int, help="Process only the first N selected files")
    analyse.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyse":
        config = load_config(args.config)
        files = selected_files_from_root(args.root, config) if args.root else selected_files_from_manifest(args.manifest)
        if args.limit:
            files = files[: args.limit]
        if not files:
            parser.error("No CZI files selected")
        if args.output_dir.exists():
            if not args.overwrite:
                parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        analyse_files(files, config, args.config.resolve(), args.output_dir, args.qc_title)
        print(f"Wrote CZI foci analysis outputs to {args.output_dir}")
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
