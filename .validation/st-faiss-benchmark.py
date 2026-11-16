import argparse
import hashlib
import json
import statistics
import time
import tracemalloc

import faiss
import numpy as np

from sentence_transformers.util.quantization import quantize_embeddings, semantic_search_faiss

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
faiss.omp_set_num_threads(1)
rng = np.random.default_rng(20261116)
records = []
for size in (5000, 50000):
    corpus = rng.normal(size=(size, 128)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.normal(size=(128, 128)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    ranges = np.stack((corpus.min(axis=0), corpus.max(axis=0)))
    codes = quantize_embeddings(corpus, "uint8", ranges=ranges)
    index = faiss.IndexFlatIP(128)
    index.add(codes)

    def run():
        return semantic_search_faiss(
            queries, corpus_index=index, corpus_precision="uint8", ranges=ranges,
            top_k=50, rescore=True, rescore_multiplier=2,
        )[0]

    for _ in range(3):
        run()
    times = []
    for _ in range(9):
        start = time.perf_counter()
        outputs = run()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    records.append({
        "corpus": size, "queries": 128, "dimension": 128, "top_k": 50,
        "median_seconds": statistics.median(times), "samples_seconds": times,
        "tracemalloc_peak_bytes": peak,
        "result_sha256": hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest(),
    })
with open(args.output, "w") as f:
    json.dump({"faiss": faiss.__version__, "numpy": np.__version__, "records": records}, f, indent=2)
print(json.dumps(records, indent=2))
