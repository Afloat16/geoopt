import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
results = []
for count in [100, 1000]:
    queries = {str(index): 'query' for index in range(200)}
    corpus = {str(index): 'document' for index in range(count)}
    relevant = {query: {str((int(query) * 7 + index * 13) % count) for index in range(20)} for query in queries}
    cutoffs = [cutoff for cutoff in [1, 3, 5, 10, 20, 50, 100, 500, 1000] if cutoff <= count]
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant, accuracy_at_k=cutoffs,
        precision_recall_at_k=cutoffs, mrr_at_k=cutoffs, ndcg_at_k=cutoffs, map_at_k=cutoffs, write_csv=False)
    generator = np.random.default_rng(32)
    rankings = [[dict(corpus_id=doc, score=float(score)) for doc, score in zip(corpus, generator.standard_normal(count))]
                for query in queries]
    timings = []
    for repeat in range(6):
        start = time.perf_counter()
        metrics = evaluator.compute_metrics(rankings)
        if repeat:
            timings.append(time.perf_counter() - start)
    result = dict(revision=revision, queries=len(queries), documents_per_query=count,
                  cutoffs=cutoffs, median_seconds=statistics.median(timings), metrics=metrics)
    if reference:
        expected = reference[len(results)]
        for metric, by_cutoff in metrics.items():
            for cutoff, value in by_cutoff.items():
                np.testing.assert_allclose(value, expected['metrics'][metric][str(cutoff)], rtol=1e-13, atol=1e-13)
        result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
    results.append(result)
    print(json.dumps(result), flush=True)
Path(revision + '.json').write_text(json.dumps(results, indent=2))
