"""Run the stage 12 turn analysis over chosen recordings and export it (D-037).

    python tools/turn_analysis.py <output_root> <recording.csv> [...] [--condition RID=LABEL ...] [--windows 5 11]

Each recording is brought up to date through stage 09 with default parameters.
The end-of-turn threshold is then pooled across all of them and stage 12 runs
on each. The per-condition summary, refused or reported contrasts, grouping
report, curves and smoothing sensitivity are written to
``<output_root>/turn_analyses/turns_<timestamp>/``. Without ``--condition``,
all recordings form one condition: the right setting when they are samples of
one population, as with one recording per genotype.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larval_explorer.core import pipeline as PL  # noqa: E402
from larval_explorer.core import schema as S  # noqa: E402
from larval_explorer.core import turn_analysis as TA  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--condition", nargs="*", default=[], help="RECORDING_ID=LABEL")
    parser.add_argument("--windows", nargs="*", type=int, default=[5, 11])
    args = parser.parse_args()

    inputs = []
    for source in args.recordings:
        pipeline = PL.RecordingPipeline(source.stem, source, args.output_root)
        pipeline.run(["09"])
        inputs.append(TA.load_turn_inputs(pipeline))
    conditions = dict(item.split("=", 1) for item in args.condition) or None
    analysis = TA.turn_analysis(inputs, conditions, sensitivity_windows=args.windows)
    folder = TA.new_turn_analysis_folder(args.output_root)
    manifest = TA.export_turn_analysis(folder, analysis)

    summary = analysis.tables["summary"]
    headline = summary[(summary[S.SUBSET] == TA.SUBSET_ALL) & summary[S.MEASURE].isin(TA.SUMMARY_MEASURES[:4] + TA.OTHER_MEASURES[:3])]
    columns = [S.CONDITION, S.MEASURE, S.N_RECORDINGS, S.N_LARVAE, S.N_TURNS, S.TURN_Q25, S.TURN_MEDIAN, S.TURN_Q75,
               S.SHARE_NEGATIVE, S.LARVA_MEAN, S.LARVA_SD, S.LARVA_SE]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 3):
        print(f"pooled end_total_bend_deg {analysis.params.end_total_bend_deg:.2f}; per recording {analysis.end_threshold_per_recording}")
        print(headline[columns].to_string(index=False))
        print()
        sens = analysis.tables["smoothing_sensitivity"]
        print(sens[[S.SPINE_SG_WINDOW, S.END_TOTAL_BEND_DEG, S.CONDITION, S.MEASURE, S.N_TURNS, S.TURN_MEDIAN, S.SHARE_NEGATIVE]].to_string(index=False))
        print()
        contrasts = analysis.tables["contrasts"]
        if len(contrasts):
            print(contrasts[[S.CONDITION_A, S.CONDITION_B, S.CONTRAST_STATUS]].drop_duplicates().to_string(index=False))
        print(analysis.tables["grouping"].to_string(index=False))
    for warning in analysis.warnings:
        print("warning:", warning)
    print("written:", manifest)


if __name__ == "__main__":
    main()
