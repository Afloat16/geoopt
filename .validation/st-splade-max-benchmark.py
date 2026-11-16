import hashlib
import json
import statistics
import sys
import time

import torch

from sentence_transformers.sparse_encoder.modules import SpladePooling


torch.set_num_threads(1)
prefix = sys.argv[1]
results = []
for flattened, activation, chunk_size in [(False, "relu", None), (False, "log1p_relu", 64),
                                           (True, "relu", None), (True, "log1p_relu", None)]:
    torch.manual_seed(47)
    logits = torch.randn(2, 256, 30522)
    mask = torch.ones(2, 256, dtype=torch.int64)
    mask[0, -32:] = 0
    pooling = SpladePooling("max", activation, chunk_size=chunk_size).eval()
    if flattened:
        logits = torch.cat([logits[0, :224], logits[1]], dim=0).unsqueeze(0)
        features = {"token_embeddings": logits, "cu_seq_lens_q": torch.tensor([0, 224, 480])}
    else:
        features = {"token_embeddings": logits, "attention_mask": mask}
    times = []
    with torch.inference_mode():
        for repeat in range(12):
            start = time.perf_counter()
            output = pooling(dict(features))["sentence_embedding"]
            if repeat >= 3:
                times.append(time.perf_counter() - start)
    result = {"flattened": flattened, "activation": activation, "chunk_size": chunk_size,
              "median_seconds": statistics.median(times), "shape": list(output.shape),
              "dtype": str(output.dtype), "sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest()}
    results.append(result)
    print(json.dumps({"revision": prefix, **result}), flush=True)
with open(prefix + "-benchmark.json", "w") as file:
    json.dump(results, file)
if len(sys.argv) > 2:
    with open(sys.argv[2] + "-benchmark.json") as file:
        before = json.load(file)
    for old, after in zip(before, results):
        assert old["sha256"] == after["sha256"]
        assert old["shape"] == after["shape"] and old["dtype"] == after["dtype"]
    print("ALL OUTPUT DIGESTS IDENTICAL", flush=True)
    print(json.dumps([{**after, "speed_ratio": old["median_seconds"] / after["median_seconds"]}
                      for old, after in zip(before, results)]), flush=True)
