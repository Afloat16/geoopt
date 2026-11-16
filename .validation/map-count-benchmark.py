import argparse
import hashlib
from pathlib import Path
import sys

import torch

import kornia

root = Path.cwd().resolve()
assert Path(kornia.__file__).resolve().parent == root / "kornia", kornia.__file__
print("kornia:", kornia.__file__, "python:", sys.executable, flush=True)
sys.path.insert(0, str(root / "benchmarks"))
from common import run_metadata, save_json, time_us, warm_up_cpu

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
args = parser.parse_args()
torch.manual_seed(0)
torch.set_num_threads(1)
warm_up_cpu()
results = []
for name, dtype, per_image in (
    ("ordinary_float32", torch.float32, 8),
    ("ordinary_bfloat16", torch.bfloat16, 8),
    ("near_full_recall_float32", torch.float32, 128),
    ("near_full_recall_bfloat16", torch.bfloat16, 128),
):
    indices = torch.arange(per_image)
    x, y = indices.remainder(16) * 8, indices.div(16, rounding_mode="floor") * 8
    boxes = torch.stack((x, y, x + 4, y + 4), -1).to(dtype)
    pred_boxes = [boxes.clone() for _ in range(4)] + [boxes[:0].clone()]
    gt_boxes = [boxes.clone() for _ in range(4)] + [boxes[:1].clone()]
    pred_labels = [torch.ones(len(b), dtype=torch.long) for b in pred_boxes]
    gt_labels = [torch.ones(len(b), dtype=torch.long) for b in gt_boxes]
    scores = [torch.ones(len(b), dtype=dtype) for b in pred_boxes]

    def metric():
        return kornia.metrics.mean_average_precision(pred_boxes, pred_labels, scores, gt_boxes, gt_labels, 2)

    actual, per_class = metric()
    median, iqr = time_us(metric, min_run_time=1.0)
    row = {
        "name": name, "dtype": str(dtype), "images": 5, "detected": 4 * per_image,
        "objects": 4 * per_image + 1, "median_us": median, "iqr_us": iqr,
        "actual": actual.item(), "per_class": per_class,
        "oracle_exact_fraction": "10/11", "oracle_in_output_dtype": torch.tensor(10 / 11, dtype=dtype).item(),
    }
    results.append(row)
    print(row, flush=True)
metadata = run_metadata(torch.device("cpu"))
metadata.update({
    "kornia_module": str(Path(kornia.__file__).resolve().relative_to(root)),
    "interpreter": sys.executable,
    "source_sha256": hashlib.sha256((root / "kornia/metrics/mean_average_precision.py").read_bytes()).hexdigest(),
})
save_json(args.json, metadata, results)
