"""Write the figures of a processed recording to its output folder.

    python tools/render_figures.py <output_root> <recording_id> [--stages 02 06] [--tracks 3] [--formats png pdf]

The recording must already have been run; its parameters are read from its
manifest. Figures go to ``<output_root>/<recording_id>/figures/<stage>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larval_explorer.core import manifest as M  # noqa: E402
from larval_explorer.core import pipeline as PL  # noqa: E402
from larval_explorer.plots import catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recording_id")
    parser.add_argument("--stages", nargs="*", default=list(catalog.STAGES_WITH_FIGURES))
    parser.add_argument("--tracks", type=int, default=None, help="limit per-track figures to the first N tracks")
    parser.add_argument("--formats", nargs="*", default=["png"])
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    manifest = M.load_manifest(args.output_root / args.recording_id)
    if manifest is None:
        raise SystemExit(f"No manifest in {args.output_root / args.recording_id}; run the recording first.")
    pipeline = PL.RecordingPipeline(args.recording_id, Path(manifest["source_file"]), args.output_root)
    pipeline.set_params(pipeline.saved_params())

    for stage in args.stages:
        if pipeline.manifest["stages"].get(stage, {}).get("status") != M.STATUS_OK:
            print(f"stage {stage}: no result, skipped")
            continue
        figures = catalog.stage_figures(pipeline, stage, max_tracks=args.tracks)
        written = pipeline.save_figures(stage, figures, formats=tuple(args.formats), dpi=args.dpi)
        print(f"stage {stage}: {len(written)} file(s) in {pipeline.figure_folder(stage)}")


if __name__ == "__main__":
    main()
