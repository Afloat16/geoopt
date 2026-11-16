import gc
import hashlib
import json
import statistics
import sys
import time

import torch

import geoopt
from geoopt.datasets.stereographic import StereographicTreeDataset


torch.set_num_threads(1)
results = []
for children, depth in [(2, 8), (100000, 1), (300000, 1)]:
    elapsed = []
    digests = []
    for _ in range(3):
        gc.collect()
        torch.manual_seed(907)
        ball = geoopt.Stereographic(-1.0)
        start = time.perf_counter()
        dataset = StereographicTreeDataset(
            ball, dim=2, depth=depth, numberOfChildren=children,
            numberOfsiblings=1, dist_children=0.1, sigma_sibling=0.02,
        )
        elapsed.append(time.perf_counter() - start)
        digest = hashlib.sha256()
        for value in [dataset.origin_data, dataset.origin_labels, dataset.data, dataset.labels,
                      torch.get_rng_state()]:
            digest.update(value.contiguous().numpy().tobytes())
        for index in [0, len(dataset) // 2, len(dataset) - 1]:
            for value in dataset[index]:
                digest.update(value.contiguous().numpy().tobytes())
        digest.update(str(dataset.num_classes).encode())
        digests.append(digest.hexdigest())
        del dataset
    assert len(set(digests)) == 1
    record = {
        "children": children, "depth": depth,
        "nodes": (children ** (depth + 1) - 1) // (children - 1),
        "median_seconds": statistics.median(elapsed),
        "trials_seconds": elapsed, "sha256": digests[0],
        "torch": torch.__version__,
    }
    results.append(record)
    print(json.dumps(record), flush=True)
with open(sys.argv[1], "w") as stream:
    json.dump({"results": results}, stream, indent=2)
