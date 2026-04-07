import argparse
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    root = next((p for p in candidates if (p / "model_training").exists()), None)
    if root is None:
        raise RuntimeError(
            f"Could not locate repo root from cwd={cwd}. Expected a parent directory containing 'model_training'."
        )
    return root


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.evaluate_model.compare_snn_rnn_results import flatten_metrics, load_pickle


LEXICON_PATH = PROJECT_ROOT / "language_model" / "examples" / "speech" / "s0" / "dict.txt"


def normalize_phone(phone: str) -> str:
    return re.sub(r"\d", "", phone.strip())


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]+", " ", text)).strip()


def load_pronunciation_lexicon(path: Path):
    exact = {}
    by_first = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            if not re.search(r"[a-z]", word):
                continue
            phones = tuple(normalize_phone(p) for p in parts[1:] if normalize_phone(p))
            if not phones:
                continue
            exact.setdefault(phones, word)
            by_first[phones[0]].append((word, phones))
    return exact, by_first


EXACT_PRONUNCIATIONS, PRONUNCIATIONS_BY_FIRST = load_pronunciation_lexicon(LEXICON_PATH)


def split_phoneme_words(tokens):
    words = []
    current = []
    for token in tokens:
        t = token.strip()
        if t in {"|", "SIL"}:
            if current:
                words.append(tuple(current))
                current = []
            continue
        current.append(normalize_phone(t))
    if current:
        words.append(tuple(current))
    return words


def phoneme_edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, start=1):
        prev_diag = dp[0]
        dp[0] = i
        for j, tok_b in enumerate(b, start=1):
            prev = dp[j]
            cost = 0 if tok_a == tok_b else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev_diag + cost)
            prev_diag = prev
    return dp[-1]


def phoneme_words_to_text(tokens):
    chunks = split_phoneme_words(tokens)
    decoded_words = []
    for chunk in chunks:
        if not chunk:
            continue
        exact_word = EXACT_PRONUNCIATIONS.get(chunk)
        if exact_word is not None:
            decoded_words.append(exact_word)
            continue

        candidates = [
            candidate
            for candidate in PRONUNCIATIONS_BY_FIRST.get(chunk[0], [])
            if abs(len(candidate[1]) - len(chunk)) <= 2
        ]
        if not candidates:
            candidates = PRONUNCIATIONS_BY_FIRST.get(chunk[0], [])
        if not candidates:
            decoded_words.append("/".join(chunk).lower())
            continue

        best_word, _ = min(
            candidates,
            key=lambda candidate: (
                phoneme_edit_distance(chunk, candidate[1]),
                abs(len(candidate[1]) - len(chunk)),
                len(candidate[0]),
                candidate[0],
            ),
        )
        decoded_words.append(best_word)
    return " ".join(decoded_words)


def parse_session_names(config_path: Path):
    sessions = []
    in_sessions = False
    with config_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("sessions:"):
                in_sessions = True
                continue
            if in_sessions:
                if line.startswith("  - "):
                    sessions.append(line.split("-", 1)[1].strip())
                elif stripped.startswith("dataset_probability_val:") or not line.startswith("  "):
                    break
    return sessions


def build_rows(run_dir: Path):
    metrics_path = run_dir / "checkpoints" / "val_metrics.pkl"
    config_path = run_dir / "configs" / "trainer_config_input.yaml"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    metrics = load_pickle(metrics_path)
    flat_rows = flatten_metrics(metrics)
    sessions = parse_session_names(config_path)
    day_to_session = {day: session for day, session in enumerate(sessions)}

    rows = []
    for row in flat_rows:
        pred_text = phoneme_words_to_text(row["pred_tokens"])
        ref_text = normalize_text(row["transcription"])
        rows.append(
            {
                "day": row["key"].day_index,
                "session": day_to_session.get(row["key"].day_index, f"day_{row['key'].day_index}"),
                "block": row["key"].block_num,
                "trial": row["key"].trial_num,
                "ground_truth_text_raw": row["transcription"],
                "ground_truth_text": ref_text,
                "predicted_text": pred_text,
                "predicted_phonemes": " ".join(row["pred_tokens"]),
                "true_phonemes": " ".join(row["true_tokens"]),
                "sentence_per": row["per"],
            }
        )
    return rows


def compute_bert(rows, model_type: str, batch_size: int, device: str | None):
    refs = [row["ground_truth_text"] for row in rows]
    hyps = [row["predicted_text"] for row in rows]
    precision, recall, f1 = bert_score_fn(
        hyps,
        refs,
        model_type=model_type,
        lang="en",
        batch_size=batch_size,
        verbose=True,
        rescale_with_baseline=False,
        device=device,
    )
    for row, p, r, f in zip(rows, precision.tolist(), recall.tolist(), f1.tolist()):
        row["bert_precision"] = float(p)
        row["bert_recall"] = float(r)
        row["bert_f1"] = float(f)
    return rows


def compute_bleu(rows):
    smooth = SmoothingFunction().method1
    for row in rows:
        ref_tokens = row["ground_truth_text"].split()
        hyp_tokens = row["predicted_text"].split()
        if not hyp_tokens:
            row["text_bleu"] = 0.0
            continue
        row["text_bleu"] = float(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=smooth,
            )
        )
    return rows


def write_csv(rows, output_csv: Path):
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, output_json: Path):
    session_groups = defaultdict(list)
    for row in rows:
        session_groups[row["session"]].append(row)

    session_summary = []
    for session, items in sorted(session_groups.items()):
        session_summary.append(
            {
                "session": session,
                "n_utterances": len(items),
                "mean_per": mean(x["sentence_per"] for x in items),
                "mean_bert_f1": mean(x["bert_f1"] for x in items),
                "mean_bert_precision": mean(x["bert_precision"] for x in items),
                "mean_bert_recall": mean(x["bert_recall"] for x in items),
                "mean_text_bleu": mean(x["text_bleu"] for x in items),
            }
        )

    summary = {
        "n_utterances": len(rows),
        "mean_per": mean(x["sentence_per"] for x in rows) if rows else None,
        "mean_bert_f1": mean(x["bert_f1"] for x in rows) if rows else None,
        "mean_bert_precision": mean(x["bert_precision"] for x in rows) if rows else None,
        "mean_bert_recall": mean(x["bert_recall"] for x in rows) if rows else None,
        "mean_text_bleu": mean(x["text_bleu"] for x in rows) if rows else None,
        "session_summary": session_summary,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Load val_metrics.pkl, reconstruct predicted text from phonemes, and compute BERTScore to ground-truth text."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Model run directory containing checkpoints/val_metrics.pkl")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV path for per-utterance results")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON path for summary results")
    parser.add_argument("--model-type", type=str, default="bert-base-uncased", help="Hugging Face model for BERTScore")
    parser.add_argument("--batch-size", type=int, default=16, help="BERTScore batch size")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu or cuda:0")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = args.run_dir / "analysis_exports" / "text_bert_score_from_val_metrics.csv"
    if args.output_json is None:
        args.output_json = args.run_dir / "analysis_exports" / "text_bert_score_from_val_metrics_summary.json"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args.run_dir)
    rows = compute_bert(rows, model_type=args.model_type, batch_size=args.batch_size, device=args.device)
    rows = compute_bleu(rows)
    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_json)

    print(f"Wrote per-utterance results to: {args.output_csv}")
    print(f"Wrote summary results to: {args.output_json}")
    print(f"Utterances scored: {len(rows)}")
    if rows:
        print(f"Mean PER: {mean(x['sentence_per'] for x in rows):.4f}")
        print(f"Mean BERTScore F1: {mean(x['bert_f1'] for x in rows):.4f}")
        print(f"Mean text BLEU: {mean(x['text_bleu'] for x in rows):.4f}")


if __name__ == "__main__":
    main()
