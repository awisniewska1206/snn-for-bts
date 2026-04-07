import re
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

BASE_PATH = "./logs"  # adjust if needed

def load_per(path):
    per_dict = {}
    pattern = re.compile(r"(t15\.\d{4}\.\d{2}\.\d{2}) val PER: ([0-9.]+)")

    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                session = match.group(1)
                per = float(match.group(2))
                per_dict[session] = per

    return per_dict


def compare_runs(snn_id, rnn_id):
    snn_path = f"{BASE_PATH}/snn/{snn_id}/training_log"
    rnn_path = f"{BASE_PATH}/rnn/{rnn_id}/training_log"

    snn = load_per(snn_path)
    rnn = load_per(rnn_path)

    # align sessions
    common = sorted(set(snn.keys()) & set(rnn.keys()))
    snn_vals = np.array([snn[s] for s in common])
    rnn_vals = np.array([rnn[s] for s in common])

    diff = snn_vals - rnn_vals

    print(f"#sessions: {len(common)}")
    print(f"SNN mean PER: {snn_vals.mean():.4f}")
    print(f"RNN mean PER: {rnn_vals.mean():.4f}")
    print(f"Mean diff (SNN-RNN): {diff.mean():.4f}")

    # Wilcoxon
    w_stat, w_p = wilcoxon(diff)
    print(f"Wilcoxon p-value: {w_p:.6f}")

    # paired t-test
    t_stat, t_p = ttest_rel(snn_vals, rnn_vals)
    print(f"Paired t-test p-value: {t_p:.6f}")

    # bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, len(diff), len(diff))
        boots.append(diff[idx].mean())
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    print(f"95% CI of mean diff: [{ci_low:.4f}, {ci_high:.4f}]")


# example
compare_runs(
    snn_id="2026-04-02_20-48-23_266921_snn_pid238600",
    rnn_id="2026-04-02_17-52-59_035938_rnn_pid238051"
)