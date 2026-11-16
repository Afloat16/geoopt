import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from qdrant_client import QdrantClient, models

from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant


class CountingClient(QdrantClient):
    single_calls = 0
    batch_calls = 0

    def query_points(self, *args, **kwargs):
        self.single_calls += 1
        return super().query_points(*args, **kwargs)

    def query_batch_points(self, *args, **kwargs):
        self.batch_calls += 1
        return super().query_batch_points(*args, **kwargs)


revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
url = os.environ.get('ST_QDRANT_URL')
client = CountingClient(url=url, timeout=60) if url else CountingClient(location=':memory:')
collection = 'st_batch_benchmark_' + uuid4().hex
results = []
try:
    client.create_collection(collection, vectors_config={}, sparse_vectors_config={'text': models.SparseVectorParams()})
    generator = np.random.default_rng(73)
    points = []
    for index in range(1000):
        indices = generator.choice(256, size=8, replace=False)
        values = generator.uniform(0.1, 1.0, size=8).astype(np.float32)
        points.append(models.PointStruct(id=index, vector={'text': models.SparseVector(indices=indices.tolist(), values=values.tolist())}))
    client.upsert(collection, points, wait=True)
    query_array = np.zeros((256, 256), dtype=np.float32)
    for row in query_array:
        indices = generator.choice(256, size=16, replace=False)
        row[indices] = generator.uniform(0.1, 1.0, size=16)
    for count in [32, 256]:
        queries = torch.from_numpy(query_array[:count]).to_sparse()
        timings = []
        for repeat in range(4):
            client.single_calls = client.batch_calls = 0
            start = time.perf_counter()
            hits, _ = semantic_search_qdrant(queries, corpus_index=(client, collection), top_k=5)
            elapsed = time.perf_counter() - start
            if repeat:
                timings.append(elapsed)
        digest = hashlib.sha256(json.dumps(hits, sort_keys=True).encode()).hexdigest()
        result = dict(revision=revision, backend='http-server' if url else 'in-memory', queries=count,
                      corpus=1000, vocabulary=256, top_k=5, median_seconds=statistics.median(timings),
                      single_calls=client.single_calls, batch_calls=client.batch_calls, results_sha256=digest)
        if reference:
            expected = reference[len(results)]
            assert expected['results_sha256'] == digest
            result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
        results.append(result)
        print(json.dumps(result), flush=True)
finally:
    client.delete_collection(collection)
    client.close()
Path(revision + '.json').write_text(json.dumps(results, indent=2))
