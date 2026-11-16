import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sentence_transformers.sparse_encoder.evaluation import ReciprocalRankFusionEvaluator


revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
out = Path('rrf-benchmark-output')
out.mkdir(exist_ok=True)
records = []
for num_documents, num_positives in [(512, 20), (2048, 512)]:
    rng = np.random.default_rng(73)
    dense, sparse = [], []
    corpus = [f'Document with shared textual prefix and unique content {i}.' for i in range(8192)]
    for q in range(20):
        positive = [corpus[i] for i in rng.choice(len(corpus), num_positives, replace=False)]
        samples = []
        for _ in range(2):
            docs = [corpus[i] for i in rng.choice(len(corpus), num_documents, replace=False)]
            samples.append({'query_id': str(q), 'query': f'query{q}', 'positive': positive, 'documents': docs})
        dense.append(samples[0])
        sparse.append(samples[1])
    evaluator = ReciprocalRankFusionEvaluator(dense, sparse, write_csv=False, write_predictions=True)
    times = []
    for repeat in range(6):
        start = time.perf_counter()
        metrics = evaluator(output_path=str(out))
        if repeat:
            times.append(time.perf_counter() - start)
    digest = hashlib.sha256((out / evaluator.predictions_file).read_bytes()).hexdigest()
    result = {'revision': revision, 'queries': 20, 'documents_per_retriever': num_documents,
              'known_positives': num_positives, 'median_seconds': statistics.median(times),
              'metrics': metrics, 'predictions_sha256': digest}
    if reference is not None:
        expected = reference[len(records)]
        assert result['metrics'] == expected['metrics']
        assert result['predictions_sha256'] == expected['predictions_sha256']
        result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
    records.append(result)
    print(json.dumps(result), flush=True)
Path(revision + '.json').write_text(json.dumps(records, indent=2))
