#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import h5py


DATE_COLUMN = "Date"
BLOCK_COLUMN = "Block number"
SENTENCE_COUNT_COLUMN = "Number of sentences"

SPLIT_COLORS = {
    "train": "#2E8B57",
    "val": "#E69F00",
    "test": "#D55E00",
}


@dataclass(frozen=True)
class SentenceRecord:
    date: str
    session_name: str
    block_number: int
    sentence_in_block: int
    sentence_index_in_day: int
    trial_idx: int
    trial_num: int
    source_hdf5: str
    source_row_index: int
    source_split: str
    corpus: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create 5 rotating train/val/test splits per day from the actual HDF5 trial inventory. "
            "Each day is shuffled once with a global seed, partitioned into 5 folds, "
            "and then rotated so each fold serves once as test and once as validation."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        default="data/hdf5_data_final",
        help="Directory containing per-session HDF5 files.",
    )
    parser.add_argument(
        "--source_filenames",
        nargs="+",
        default=["data_train.hdf5", "data_val.hdf5"],
        help=(
            "HDF5 files to use as the labeled trial pool for fold construction. "
            "Defaults to data_train.hdf5 and data_val.hdf5, excluding unlabeled data_test.hdf5."
        ),
    )
    parser.add_argument(
        "--csv_path",
        default="data/t15_copyTaskData_description.csv",
        help="Optional metadata CSV used only for mismatch reporting.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/daywise_sentence_splits",
        help="Directory where split CSV/JSON/SVG artifacts will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed used for deterministic day-wise shuffling.",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds to create per day. Default: 5.",
    )
    parser.add_argument(
        "--line_height",
        type=int,
        default=3,
        help="Pixel height per sentence line in the SVG audit plot.",
    )
    parser.add_argument(
        "--date_panel_width",
        type=int,
        default=96,
        help="Width of each per-date rectangle in the SVG audit plot.",
    )
    return parser.parse_args()


def session_name_to_date(session_name: str) -> str:
    parts = session_name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Session name does not match expected pattern: {session_name}")
    return f"{parts[1]}-{parts[2]}-{parts[3]}"


def load_sentence_records_from_hdf5(dataset_dir: Path, source_filenames: Sequence[str]) -> Dict[str, List[SentenceRecord]]:
    by_date: Dict[str, List[SentenceRecord]] = defaultdict(list)
    grouped_by_date_block: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for session_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        session_name = session_dir.name
        date = session_name_to_date(session_name)

        for hdf5_name in source_filenames:
            hdf5_path = session_dir / hdf5_name
            if not hdf5_path.exists():
                continue

            with h5py.File(hdf5_path, "r") as handle:
                for trial_key in handle.keys():
                    trial_idx = int(trial_key.split("_")[-1])
                    trial_group = handle[trial_key]
                    block_number = int(trial_group.attrs["block_num"])
                    trial_num = int(trial_group.attrs["trial_num"])
                    grouped_by_date_block[date][block_number].append(
                        {
                            "session_name": session_name,
                            "trial_idx": trial_idx,
                            "trial_num": trial_num,
                            "source_hdf5": str(hdf5_path),
                            "source_split": hdf5_path.stem.replace("data_", ""),
                        }
                    )

    for date in sorted(grouped_by_date_block.keys()):
        sentence_index_in_day = 0
        for block_number in sorted(grouped_by_date_block[date].keys()):
            block_records = sorted(
                grouped_by_date_block[date][block_number],
                key=lambda item: (
                    item["trial_num"],
                    Path(item["source_hdf5"]).name,
                    item["trial_idx"],
                ),
            )
            for sentence_in_block, item in enumerate(block_records):
                by_date[date].append(
                    SentenceRecord(
                        date=date,
                        session_name=item["session_name"],
                        block_number=block_number,
                        sentence_in_block=sentence_in_block,
                        sentence_index_in_day=sentence_index_in_day,
                        trial_idx=item["trial_idx"],
                        trial_num=item["trial_num"],
                        source_hdf5=item["source_hdf5"],
                        source_row_index=-1,
                        source_split=item["source_split"],
                        corpus="",
                    )
                )
                sentence_index_in_day += 1

    return dict(sorted(by_date.items()))


def compare_csv_against_hdf5(csv_path: Path, by_date: Dict[str, List[SentenceRecord]]) -> dict:
    csv_counts = defaultdict(lambda: defaultdict(int))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {DATE_COLUMN, BLOCK_COLUMN, SENTENCE_COUNT_COLUMN}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            csv_counts[row[DATE_COLUMN].strip()][int(row[BLOCK_COLUMN])] += int(row[SENTENCE_COUNT_COLUMN])

    hdf5_counts = defaultdict(lambda: defaultdict(int))
    for date, records in by_date.items():
        for rec in records:
            hdf5_counts[date][rec.block_number] += 1

    mismatches = []
    all_dates = sorted(set(csv_counts.keys()) | set(hdf5_counts.keys()))
    for date in all_dates:
        all_blocks = sorted(set(csv_counts[date].keys()) | set(hdf5_counts[date].keys()))
        for block_number in all_blocks:
            csv_count = int(csv_counts[date].get(block_number, 0))
            hdf5_count = int(hdf5_counts[date].get(block_number, 0))
            if csv_count != hdf5_count:
                mismatches.append(
                    {
                        "date": date,
                        "block_number": block_number,
                        "csv_sentence_count": csv_count,
                        "hdf5_trial_count": hdf5_count,
                        "delta_hdf5_minus_csv": hdf5_count - csv_count,
                    }
                )

    return {
        "num_dates_in_csv": len(csv_counts),
        "num_dates_in_hdf5": len(hdf5_counts),
        "num_mismatched_blocks": len(mismatches),
        "mismatches": mismatches,
    }


def write_mismatch_report(output_dir: Path, report: dict) -> None:
    json_path = output_dir / "csv_hdf5_mismatch_report.json"
    csv_path = output_dir / "csv_hdf5_mismatch_report.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "block_number",
                "csv_sentence_count",
                "hdf5_trial_count",
                "delta_hdf5_minus_csv",
            ],
        )
        writer.writeheader()
        for row in report["mismatches"]:
            writer.writerow(row)


def split_into_folds(items: Sequence[SentenceRecord], num_folds: int, seed: int, date: str) -> List[List[SentenceRecord]]:
    if len(items) < num_folds:
        raise ValueError(
            f"Date {date} has only {len(items)} sentences, which is fewer than num_folds={num_folds}."
        )

    shuffled = list(items)
    rng = random.Random(f"{seed}:{date}")
    rng.shuffle(shuffled)

    base_size = len(shuffled) // num_folds
    remainder = len(shuffled) % num_folds

    folds: List[List[SentenceRecord]] = []
    cursor = 0
    for fold_idx in range(num_folds):
        fold_size = base_size + (1 if fold_idx < remainder else 0)
        folds.append(shuffled[cursor : cursor + fold_size])
        cursor += fold_size

    return folds


def build_iteration_assignments(
    by_date: Dict[str, List[SentenceRecord]],
    num_folds: int,
    seed: int,
) -> tuple[
    Dict[str, List[List[SentenceRecord]]],
    List[Dict[str, Dict[str, List[SentenceRecord]]]],
]:
    iterations: List[Dict[str, Dict[str, List[SentenceRecord]]]] = []

    per_date_folds = {
        date: split_into_folds(records, num_folds=num_folds, seed=seed, date=date)
        for date, records in by_date.items()
    }

    for iteration_idx in range(num_folds):
        date_assignments: Dict[str, Dict[str, List[SentenceRecord]]] = {}
        test_fold = iteration_idx
        val_fold = (iteration_idx + 1) % num_folds

        for date, folds in per_date_folds.items():
            assigned = {"train": [], "val": [], "test": []}
            for fold_idx, fold_records in enumerate(folds):
                if fold_idx == test_fold:
                    split_name = "test"
                elif fold_idx == val_fold:
                    split_name = "val"
                else:
                    split_name = "train"
                assigned[split_name].extend(fold_records)

            for split_name in assigned:
                assigned[split_name].sort(key=lambda rec: rec.sentence_index_in_day)

            date_assignments[date] = assigned

        iterations.append(date_assignments)

    return per_date_folds, iterations


def write_iteration_csv(
    output_path: Path,
    iteration_idx: int,
    date_assignments: Dict[str, Dict[str, List[SentenceRecord]]],
) -> None:
    fieldnames = [
        "iteration",
        "date",
        "split",
        "session_name",
        "sentence_index_in_day",
        "block_number",
        "sentence_in_block",
        "trial_idx",
        "trial_num",
        "source_hdf5",
        "source_row_index",
        "source_split",
        "corpus",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for date, split_map in date_assignments.items():
            for split_name in ("train", "val", "test"):
                for rec in split_map[split_name]:
                    writer.writerow(
                        {
                            "iteration": iteration_idx,
                            "date": date,
                            "split": split_name,
                            "session_name": rec.session_name,
                            "sentence_index_in_day": rec.sentence_index_in_day,
                            "block_number": rec.block_number,
                            "sentence_in_block": rec.sentence_in_block,
                            "trial_idx": rec.trial_idx,
                            "trial_num": rec.trial_num,
                            "source_hdf5": rec.source_hdf5,
                            "source_row_index": rec.source_row_index,
                            "source_split": rec.source_split,
                            "corpus": rec.corpus,
                        }
                    )


def write_summary_json(
    output_path: Path,
    iterations: List[Dict[str, Dict[str, List[SentenceRecord]]]],
    seed: int,
    num_folds: int,
) -> None:
    payload = {
        "seed": seed,
        "num_folds": num_folds,
        "iterations": [],
    }

    for iteration_idx, date_assignments in enumerate(iterations):
        iter_payload = {
            "iteration": iteration_idx,
            "dates": {},
        }
        for date, split_map in date_assignments.items():
            iter_payload["dates"][date] = {
                split_name: {
                    "count": len(records),
                    "sentence_indices_in_day": [rec.sentence_index_in_day for rec in records],
                    "block_sentence_pairs": [
                        {
                            "session_name": rec.session_name,
                            "block_number": rec.block_number,
                            "sentence_in_block": rec.sentence_in_block,
                            "trial_idx": rec.trial_idx,
                            "trial_num": rec.trial_num,
                            "source_split": rec.source_split,
                        }
                        for rec in records
                    ],
                }
                for split_name, records in split_map.items()
            }
        payload["iterations"].append(iter_payload)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_iteration_svg(
    output_path: Path,
    iteration_idx: int,
    date_assignments: Dict[str, Dict[str, List[SentenceRecord]]],
    line_height: int,
    date_panel_width: int,
) -> None:
    dates = list(date_assignments.keys())
    max_sentences = max(
        sum(len(records) for records in split_map.values())
        for split_map in date_assignments.values()
    )

    margin_left = 26
    margin_right = 20
    margin_top = 86
    margin_bottom = 20
    panel_gap = 18
    label_height = 36

    plot_height = max_sentences * line_height
    width = margin_left + margin_right + (len(dates) * date_panel_width) + ((len(dates) - 1) * panel_gap)
    height = margin_top + margin_bottom + label_height + plot_height

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{margin_left}" y="28" font-family="monospace" font-size="18" fill="#111111">'
        f'Iteration {iteration_idx}: day-wise 5-fold train/val/test split'
        f"</text>"
    )
    parts.append(
        f'<text x="{margin_left}" y="50" font-family="monospace" font-size="12" fill="#444444">'
        f'Each colored line is one sentence. Order is the original within-day sentence order.'
        f"</text>"
    )

    legend_x = margin_left
    legend_y = 68
    for split_name in ("train", "val", "test"):
        color = SPLIT_COLORS[split_name]
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="10" fill="{color}" stroke="none"/>'
        )
        parts.append(
            f'<text x="{legend_x + 20}" y="{legend_y + 9}" font-family="monospace" font-size="12" fill="#111111">'
            f"{split_name}</text>"
        )
        legend_x += 86

    plot_y0 = margin_top + label_height

    for idx, date in enumerate(dates):
        x0 = margin_left + idx * (date_panel_width + panel_gap)
        split_map = date_assignments[date]
        sentence_to_split = {}
        for split_name, records in split_map.items():
            for rec in records:
                sentence_to_split[rec.sentence_index_in_day] = split_name

        parts.append(
            f'<text x="{x0 + date_panel_width / 2:.1f}" y="{margin_top - 16}" text-anchor="middle" '
            f'font-family="monospace" font-size="12" fill="#111111">{date}</text>'
        )
        counts_text = (
            f"T:{len(split_map['train'])} V:{len(split_map['val'])} E:{len(split_map['test'])}"
        )
        parts.append(
            f'<text x="{x0 + date_panel_width / 2:.1f}" y="{margin_top}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="#555555">{counts_text}</text>'
        )
        parts.append(
            f'<rect x="{x0}" y="{plot_y0}" width="{date_panel_width}" height="{plot_height}" '
            f'fill="none" stroke="#c7c7c7" stroke-width="1"/>'
        )

        for sentence_idx in range(max_sentences):
            y = plot_y0 + sentence_idx * line_height + (line_height / 2.0)
            if sentence_idx in sentence_to_split:
                color = SPLIT_COLORS[sentence_to_split[sentence_idx]]
            else:
                color = "#eaeaea"
            parts.append(
                f'<line x1="{x0 + 2}" y1="{y:.2f}" x2="{x0 + date_panel_width - 2}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="{max(1, line_height - 1)}" />'
            )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_fold_arranged_svg(
    output_path: Path,
    iteration_idx: int,
    date_assignments: Dict[str, Dict[str, List[SentenceRecord]]],
    per_date_folds: Dict[str, List[List[SentenceRecord]]],
    line_height: int,
    date_panel_width: int,
) -> None:
    dates = list(date_assignments.keys())
    max_sentences = max(
        sum(len(records) for records in split_map.values())
        for split_map in date_assignments.values()
    )

    margin_left = 26
    margin_right = 20
    margin_top = 86
    margin_bottom = 20
    panel_gap = 18
    label_height = 36

    plot_height = max_sentences * line_height
    width = margin_left + margin_right + (len(dates) * date_panel_width) + ((len(dates) - 1) * panel_gap)
    height = margin_top + margin_bottom + label_height + plot_height

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{margin_left}" y="28" font-family="monospace" font-size="18" fill="#111111">'
        f"Iteration {iteration_idx}: fold-arranged day-wise split view"
        f"</text>"
    )
    parts.append(
        f'<text x="{margin_left}" y="50" font-family="monospace" font-size="12" fill="#444444">'
        f"Sentences are reordered by fold 0..4. Colors still reflect this iteration's split assignment."
        f"</text>"
    )

    legend_x = margin_left
    legend_y = 68
    for split_name in ("train", "val", "test"):
        color = SPLIT_COLORS[split_name]
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="10" fill="{color}" stroke="none"/>'
        )
        parts.append(
            f'<text x="{legend_x + 20}" y="{legend_y + 9}" font-family="monospace" font-size="12" fill="#111111">'
            f"{split_name}</text>"
        )
        legend_x += 86

    plot_y0 = margin_top + label_height

    for idx, date in enumerate(dates):
        x0 = margin_left + idx * (date_panel_width + panel_gap)
        split_map = date_assignments[date]
        folds = per_date_folds[date]

        sentence_to_split = {}
        for split_name, records in split_map.items():
            for rec in records:
                sentence_to_split[rec.sentence_index_in_day] = split_name

        ordered_records: List[SentenceRecord] = []
        fold_boundaries: List[int] = []
        cursor = 0
        for fold_idx, fold_records in enumerate(folds):
            ordered_records.extend(sorted(fold_records, key=lambda rec: rec.sentence_index_in_day))
            cursor += len(fold_records)
            if fold_idx < len(folds) - 1:
                fold_boundaries.append(cursor)

        parts.append(
            f'<text x="{x0 + date_panel_width / 2:.1f}" y="{margin_top - 16}" text-anchor="middle" '
            f'font-family="monospace" font-size="12" fill="#111111">{date}</text>'
        )
        counts_text = (
            f"T:{len(split_map['train'])} V:{len(split_map['val'])} E:{len(split_map['test'])}"
        )
        parts.append(
            f'<text x="{x0 + date_panel_width / 2:.1f}" y="{margin_top}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="#555555">{counts_text}</text>'
        )
        parts.append(
            f'<rect x="{x0}" y="{plot_y0}" width="{date_panel_width}" height="{plot_height}" '
            f'fill="none" stroke="#c7c7c7" stroke-width="1"/>'
        )

        for row_idx, rec in enumerate(ordered_records):
            y = plot_y0 + row_idx * line_height + (line_height / 2.0)
            color = SPLIT_COLORS[sentence_to_split[rec.sentence_index_in_day]]
            parts.append(
                f'<line x1="{x0 + 2}" y1="{y:.2f}" x2="{x0 + date_panel_width - 2}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="{max(1, line_height - 1)}" />'
            )

        for sentence_idx in range(len(ordered_records), max_sentences):
            y = plot_y0 + sentence_idx * line_height + (line_height / 2.0)
            parts.append(
                f'<line x1="{x0 + 2}" y1="{y:.2f}" x2="{x0 + date_panel_width - 2}" y2="{y:.2f}" '
                f'stroke="#eaeaea" stroke-width="{max(1, line_height - 1)}" />'
            )

        for fold_idx, boundary in enumerate(fold_boundaries):
            y = plot_y0 + boundary * line_height
            parts.append(
                f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + date_panel_width}" y2="{y:.2f}" '
                f'stroke="#666666" stroke-width="1" stroke-dasharray="3 2" />'
            )
            parts.append(
                f'<text x="{x0 + date_panel_width - 2}" y="{max(plot_y0 + 10, y - 3):.2f}" text-anchor="end" '
                f'font-family="monospace" font-size="9" fill="#666666">f{fold_idx}</text>'
            )

        first_y = plot_y0 + 10
        parts.append(
            f'<text x="{x0 + 3}" y="{first_y:.2f}" font-family="monospace" font-size="9" fill="#666666">f0</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_date = load_sentence_records_from_hdf5(dataset_dir, source_filenames=args.source_filenames)
    per_date_folds, iterations = build_iteration_assignments(
        by_date=by_date,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    if csv_path.exists():
        mismatch_report = compare_csv_against_hdf5(csv_path, by_date)
        write_mismatch_report(output_dir, mismatch_report)
        print(
            f"CSV/HDF5 mismatch report written with "
            f"{mismatch_report['num_mismatched_blocks']} mismatched block(s)"
        )

    write_summary_json(
        output_dir / "split_summary.json",
        iterations=iterations,
        seed=args.seed,
        num_folds=args.num_folds,
    )

    for iteration_idx, date_assignments in enumerate(iterations):
        write_iteration_csv(
            output_dir / f"iteration_{iteration_idx:02d}_splits.csv",
            iteration_idx=iteration_idx,
            date_assignments=date_assignments,
        )
        render_iteration_svg(
            output_dir / f"iteration_{iteration_idx:02d}_audit.svg",
            iteration_idx=iteration_idx,
            date_assignments=date_assignments,
            line_height=args.line_height,
            date_panel_width=args.date_panel_width,
        )
        render_fold_arranged_svg(
            output_dir / f"iteration_{iteration_idx:02d}_fold_arranged.svg",
            iteration_idx=iteration_idx,
            date_assignments=date_assignments,
            per_date_folds=per_date_folds,
            line_height=args.line_height,
            date_panel_width=args.date_panel_width,
        )

    print(f"Wrote {len(iterations)} iterations to: {output_dir}")
    print(f"Dates: {len(by_date)}")
    print(f"Total HDF5 trials: {sum(len(records) for records in by_date.values())}")
    print(f"Labeled source files used for splits: {list(args.source_filenames)}")


if __name__ == "__main__":
    main()
