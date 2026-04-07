import argparse
import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


LOGIT_TO_PHONEME = [
    "BLANK",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    " | ",
]


@dataclass(frozen=True)
class UtteranceKey:
    day_index: int
    block_num: int
    trial_num: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "day_index": self.day_index,
            "block_num": self.block_num,
            "trial_num": self.trial_num,
        }


def extract_transcription(arr) -> str:
    chars = []
    for x in arr:
        x_int = int(x)
        if x_int == 0:
            break
        chars.append(chr(x_int))
    return "".join(chars)


def ids_to_phonemes(ids: Sequence[int]) -> List[str]:
    return [LOGIT_TO_PHONEME[int(i)] for i in ids]


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def simple_bleu(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not hypothesis:
        return 0.0
    precisions = []
    for n in (1, 2):
        hyp = ngrams(hypothesis, n)
        ref = ngrams(reference, n)
        if not hyp:
            precisions.append(1e-9)
            continue
        overlap = sum(min(c, ref.get(g, 0)) for g, c in hyp.items())
        total = sum(hyp.values())
        precisions.append((overlap + 1e-9) / (total + 1e-9))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / max(hyp_len, 1)))
    return float(bp * geo_mean)


def levenshtein_alignment(
    ref: Sequence[str], hyp: Sequence[str]
) -> Tuple[int, List[Tuple[str, str, str]]]:
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = 0 if ref[i - 1] == hyp[j - 1] else 1
            cand = [
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
                (dp[i - 1][j - 1] + cost_sub, "M" if cost_sub == 0 else "S"),
            ]
            best_cost, op = min(cand, key=lambda x: x[0])
            dp[i][j] = best_cost
            bt[i][j] = op

    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("M", "S"):
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        elif op == "I":
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
        else:
            break
    ops.reverse()
    return dp[n][m], ops


def edit_similarity(ref: Sequence[str], hyp: Sequence[str]) -> float:
    denom = max(len(ref), len(hyp), 1)
    dist, _ = levenshtein_alignment(ref, hyp)
    return float(1.0 - (dist / denom))


def cosine_from_counts(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2:
        return 0.0
    keys = set(c1) | set(c2)
    dot = sum(c1.get(k, 0.0) * c2.get(k, 0.0) for k in keys)
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


def phoneme_embedding(tokens: Sequence[str]) -> Counter:
    uni = Counter(tokens)
    bi = ngrams(tokens, 2)
    feats = Counter()
    feats.update({("U", k): v for k, v in uni.items()})
    feats.update({("B",) + k: v for k, v in bi.items()})
    return feats


def flatten_metrics(metrics: Dict) -> List[Dict]:
    rows = []
    bleu_idx = 0
    bert_idx = 0
    for batch_idx in range(len(metrics["decoded_seqs"])):
        decoded_batch = metrics["decoded_seqs"][batch_idx]
        true_batch = metrics["true_seq"][batch_idx]
        lens_batch = metrics["phone_seq_lens"][batch_idx]
        trans_batch = metrics["transcription"][batch_idx]
        block_batch = metrics["block_nums"][batch_idx]
        trial_batch = metrics["trial_nums"][batch_idx]
        day_batch = metrics["day_indicies"][batch_idx]

        for item_idx in range(len(decoded_batch)):
            true_ids = [int(x) for x in true_batch[item_idx][: int(lens_batch[item_idx])]]
            pred_ids = [int(x) for x in decoded_batch[item_idx]]
            true_tokens = ids_to_phonemes(true_ids)
            pred_tokens = ids_to_phonemes(pred_ids)
            dist, ops = levenshtein_alignment(true_tokens, pred_tokens)
            true_len = max(len(true_tokens), 1)
            per = dist / true_len

            rows.append(
                {
                    "batch_idx": int(batch_idx),
                    "item_idx": int(item_idx),
                    "key": UtteranceKey(
                        int(day_batch[item_idx]),
                        int(block_batch[item_idx]),
                        int(trial_batch[item_idx]),
                    ),
                    "transcription": extract_transcription(trans_batch[item_idx]),
                    "true_ids": true_ids,
                    "pred_ids": pred_ids,
                    "true_tokens": true_tokens,
                    "pred_tokens": pred_tokens,
                    "edit_distance": int(dist),
                    "per": float(per),
                    "bleu_to_true": float(
                        metrics["bleu_scores"][bleu_idx]
                        if bleu_idx < len(metrics.get("bleu_scores", []))
                        else simple_bleu(true_tokens, pred_tokens)
                    ),
                    "bert_to_true": float(
                        metrics["bert_scores"][bert_idx]
                        if bert_idx < len(metrics.get("bert_scores", []))
                        else cosine_from_counts(
                            phoneme_embedding(true_tokens), phoneme_embedding(pred_tokens)
                        )
                    ),
                    "edit_similarity_to_true": float(edit_similarity(true_tokens, pred_tokens)),
                    "embedding_cosine_to_true": float(
                        cosine_from_counts(
                            phoneme_embedding(true_tokens), phoneme_embedding(pred_tokens)
                        )
                    ),
                    "ops": ops,
                }
            )
            bleu_idx += 1
            bert_idx += 1
    return rows


def confusion_and_error_structure(rows: List[Dict]) -> Dict:
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()
    totals = Counter()
    for r in rows:
        for op, ref_tok, hyp_tok in r["ops"]:
            if op == "M":
                totals["match"] += 1
            elif op == "S":
                totals["substitution"] += 1
                substitutions[(ref_tok, hyp_tok)] += 1
            elif op == "I":
                totals["insertion"] += 1
                insertions[hyp_tok] += 1
            elif op == "D":
                totals["deletion"] += 1
                deletions[ref_tok] += 1
        totals["true_tokens"] += len(r["true_tokens"])
        totals["pred_tokens"] += len(r["pred_tokens"])

    err_total = totals["substitution"] + totals["insertion"] + totals["deletion"]
    per_mean = sum(r["per"] for r in rows) / max(len(rows), 1)
    return {
        "aggregate": {
            "num_utterances": len(rows),
            "mean_sentence_per": per_mean,
            "total_substitutions": totals["substitution"],
            "total_insertions": totals["insertion"],
            "total_deletions": totals["deletion"],
            "total_matches": totals["match"],
            "error_total": err_total,
            "substitution_ratio": totals["substitution"] / max(err_total, 1),
            "insertion_ratio": totals["insertion"] / max(err_total, 1),
            "deletion_ratio": totals["deletion"] / max(err_total, 1),
        },
        "top_substitutions": [
            {"true": k[0], "pred": k[1], "count": v}
            for k, v in substitutions.most_common(40)
        ],
        "top_insertions": [{"pred": k, "count": v} for k, v in insertions.most_common(30)],
        "top_deletions": [{"true": k, "count": v} for k, v in deletions.most_common(30)],
    }


def paired_comparison(rnn_rows: List[Dict], snn_rows: List[Dict], top_k: int) -> Dict:
    rnn_map = {r["key"]: r for r in rnn_rows}
    snn_map = {r["key"]: r for r in snn_rows}
    shared_keys = sorted(set(rnn_map) & set(snn_map), key=lambda k: (k.day_index, k.block_num, k.trial_num))

    pairs = []
    for key in shared_keys:
        r = rnn_map[key]
        s = snn_map[key]
        r_vs_s_bleu = simple_bleu(r["pred_tokens"], s["pred_tokens"])
        r_vs_s_edit_sim = edit_similarity(r["pred_tokens"], s["pred_tokens"])
        r_vs_s_cos = cosine_from_counts(
            phoneme_embedding(r["pred_tokens"]), phoneme_embedding(s["pred_tokens"])
        )
        pairs.append(
            {
                "key": key.as_dict(),
                "transcription": r["transcription"] or s["transcription"],
                "rnn": {
                    "per": r["per"],
                    "edit_distance": r["edit_distance"],
                    "bleu_to_true": r["bleu_to_true"],
                    "bert_to_true": r["bert_to_true"],
                    "edit_similarity_to_true": r["edit_similarity_to_true"],
                    "embedding_cosine_to_true": r["embedding_cosine_to_true"],
                    "pred_phonemes": r["pred_tokens"],
                },
                "snn": {
                    "per": s["per"],
                    "edit_distance": s["edit_distance"],
                    "bleu_to_true": s["bleu_to_true"],
                    "bert_to_true": s["bert_to_true"],
                    "edit_similarity_to_true": s["edit_similarity_to_true"],
                    "embedding_cosine_to_true": s["embedding_cosine_to_true"],
                    "pred_phonemes": s["pred_tokens"],
                },
                "true_phonemes": r["true_tokens"],
                "delta": {
                    "snn_minus_rnn_per": s["per"] - r["per"],
                    "abs_per_gap": abs(s["per"] - r["per"]),
                    "snn_minus_rnn_bert_to_true": s["bert_to_true"] - r["bert_to_true"],
                    "snn_minus_rnn_bleu_to_true": s["bleu_to_true"] - r["bleu_to_true"],
                    "rnn_vs_snn_bleu": r_vs_s_bleu,
                    "rnn_vs_snn_edit_similarity": r_vs_s_edit_sim,
                    "rnn_vs_snn_embedding_cosine": r_vs_s_cos,
                },
            }
        )

    pairs_by_gap = sorted(pairs, key=lambda x: x["delta"]["abs_per_gap"], reverse=True)
    low_semantic_gap = [
        p
        for p in pairs_by_gap
        if p["delta"]["abs_per_gap"] >= 0.15
        and abs(p["delta"]["snn_minus_rnn_bert_to_true"]) <= 0.03
    ]

    avg_abs_per_gap = sum(p["delta"]["abs_per_gap"] for p in pairs) / max(len(pairs), 1)
    avg_abs_bert_gap = (
        sum(abs(p["delta"]["snn_minus_rnn_bert_to_true"]) for p in pairs) / max(len(pairs), 1)
    )

    interpretation = {
        "mean_abs_per_gap": avg_abs_per_gap,
        "mean_abs_bert_gap": avg_abs_bert_gap,
        "semantic_gap_small_flag": bool(avg_abs_bert_gap <= 0.03),
        "note": (
            "When semantic/embedding gap is small while PER gap is non-trivial, "
            "the PER gap can overstate practical semantic difference."
        ),
    }

    return {
        "num_shared_utterances": len(shared_keys),
        "num_rnn_only": len(rnn_map) - len(shared_keys),
        "num_snn_only": len(snn_map) - len(shared_keys),
        "summary": interpretation,
        "largest_decoding_differences": pairs_by_gap[:top_k],
        "high_per_gap_but_small_semantic_gap_examples": low_semantic_gap[:top_k],
    }


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare RNN vs SNN validation outputs and export rich JSON analysis."
    )
    parser.add_argument("--rnn-run-dir", type=Path, required=True)
    parser.add_argument("--snn-run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: model_training/logs/compare_snn_vs_rnn_<rnn>__<snn>.json",
    )
    args = parser.parse_args()

    rnn_pickle = args.rnn_run_dir / "checkpoints" / "val_metrics.pkl"
    snn_pickle = args.snn_run_dir / "checkpoints" / "val_metrics.pkl"
    if not rnn_pickle.exists():
        raise FileNotFoundError(f"Missing RNN metrics file: {rnn_pickle}")
    if not snn_pickle.exists():
        raise FileNotFoundError(f"Missing SNN metrics file: {snn_pickle}")

    rnn_metrics = load_pickle(rnn_pickle)
    snn_metrics = load_pickle(snn_pickle)

    rnn_rows = flatten_metrics(rnn_metrics)
    snn_rows = flatten_metrics(snn_metrics)

    output_json = args.output_json
    if output_json is None:
        output_json = (
            Path("model_training/logs")
            / f"compare_snn_vs_rnn_{args.rnn_run_dir.name}__{args.snn_run_dir.name}.json"
        )

    report = {
        "runs": {
            "rnn_run_dir": str(args.rnn_run_dir),
            "snn_run_dir": str(args.snn_run_dir),
        },
        "model_level": {
            "rnn": confusion_and_error_structure(rnn_rows),
            "snn": confusion_and_error_structure(snn_rows),
        },
        "pairwise": paired_comparison(rnn_rows, snn_rows, args.top_k),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote comparison report to: {output_json}")


if __name__ == "__main__":
    main()
