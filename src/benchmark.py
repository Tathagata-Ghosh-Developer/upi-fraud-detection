"""Optional: time the same XGBoost fit on the GPU and on all CPU cores.

Fixed number of boosting rounds and no early stopping, so both devices do
identical work. Writes results/gpu_vs_cpu.json.
"""

from __future__ import annotations

import os
import time

import xgboost as xgb

from .common import LABEL, RESULTS_DIR, base_parser, feature_list, load_config, load_split, role_feature_set, write_json
from .train import scale_pos_weight


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config(args.config)
    features = feature_list(cfg, role_feature_set(cfg, "primary"))
    train = load_split(cfg, "train", [LABEL, *features])

    results = {"rows": len(train), "features": len(features), "rounds": args.rounds, "cpu_threads": os.cpu_count()}
    for device in ("cuda", "cpu"):
        params = {**cfg["xgboost"]["params"], "device": device, "nthread": os.cpu_count(), "seed": cfg["seed"],
                  "scale_pos_weight": scale_pos_weight(train[LABEL].to_numpy(), "full")}
        start = time.perf_counter()
        dtrain = xgb.QuantileDMatrix(train[features], label=train[LABEL])
        xgb.train(params, dtrain, num_boost_round=args.rounds)
        results[f"{device}_seconds"] = round(time.perf_counter() - start, 1)
        print(f"{device}: {results[f'{device}_seconds']} s")
    results["speedup"] = round(results["cpu_seconds"] / results["cuda_seconds"], 1)
    write_json(RESULTS_DIR / "gpu_vs_cpu.json", results)
    print(results)


if __name__ == "__main__":
    main()
