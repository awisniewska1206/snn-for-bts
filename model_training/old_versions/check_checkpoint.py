import pickle
import numpy as np

path = "logs/snn/2026-03-20_18-14-36_538374_snn_pid1022490/checkpoints/val_metrics.pkl"

with open(path, "rb") as f:
    m = pickle.load(f)

print("keys:", m.keys())

if "avg_PER" in m:
    print("avg_PER:", m["avg_PER"])

if "avg_loss" in m:
    print("avg_loss:", m["avg_loss"])

if "day_PERs" in m:
    print("day_PERs:")
    for day, d in m["day_PERs"].items():
        per = d["total_edit_distance"] / d["total_seq_length"]
        print(day, per)

if "losses" in m:
    print("n_losses:", len(m["losses"]))
    print("mean(losses):", np.mean(m["losses"]))

import torch

ckpt_path = "logs/snn/2026-03-20_18-14-36_538374_snn_pid1022490/checkpoints/best_checkpoint"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("checkpoint keys:", ckpt.keys())
print("val_PER:", ckpt.get("val_PER"))
print("val_loss:", ckpt.get("val_loss"))

sd = ckpt["model_state_dict"]
print("\nstate_dict keys:")
for k in list(sd.keys())[:200]:
    print(k)