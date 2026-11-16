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
    corpus_array = np.zeros((1000, 256), dtype=np.float32)
    for index in range(1000):
        indices = generator.choice(256, size=8, replace=False)
        values = generator.uniform(0.1, 1.0, size=8).astype(np.float32)
        corpus_array[index, indices] = values
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
        # Independent float64 dot products establish the expected ranking and score.
        # Each sparse document has at most eight nonzero products, so gamma_16
        # bounds their float32 multiply-and-accumulate relative error.
        oracle = query_array[:count].astype(np.float64) @ corpus_array.astype(np.float64).T
        expected_ids = np.argsort(-oracle, axis=1)[:, :5]
        actual_ids = np.asarray([[hit['corpus_id'] for hit in row] for row in hits])
        actual_scores = np.asarray([[hit['score'] for hit in row] for row in hits])
        oracle_scores = np.take_along_axis(oracle, expected_ids, axis=1)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        epsilon = np.finfo(np.float32).eps
        gamma_16 = 16 * epsilon / (1 - 16 * epsilon)
        np.testing.assert_allclose(actual_scores, oracle_scores, rtol=gamma_16, atol=0)
        oracle_max_relative_error = float(np.max(np.abs(actual_scores / oracle_scores - 1)))
        digest = hashlib.sha256(json.dumps(hits, sort_keys=True).encode()).hexdigest()
        result = dict(revision=revision, backend='http-server' if url else 'in-memory', queries=count,
                      corpus=1000, vocabulary=256, top_k=5, median_seconds=statistics.median(timings),
                      single_calls=client.single_calls, batch_calls=client.batch_calls, results_sha256=digest, oracle_max_relative_error=oracle_max_relative_error)
        if reference:
            expected = reference[len(results)]
            previous_ids = np.asarray([[hit['corpus_id'] for hit in row] for row in expected['hits']])
            previous_scores = np.asarray([[hit['score'] for hit in row] for row in expected['hits']])
            np.testing.assert_array_equal(actual_ids, previous_ids)
            result['exact_digest_match'] = expected['results_sha256'] == digest
            result['changed_scores'] = int(np.sum(actual_scores != previous_scores))
            result['max_score_difference'] = float(np.max(np.abs(actual_scores - previous_scores)))
            result['max_relative_score_difference'] = float(np.max(np.abs((actual_scores - previous_scores) / oracle_scores)))
            result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
        print(json.dumps(result), flush=True)
        result['hits'] = hits
        results.append(result)
finally:
    client.delete_collection(collection)
    client.close()
Path(revision + '.json').write_text(json.dumps(results, indent=2))
