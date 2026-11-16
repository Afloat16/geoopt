import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from sentence_transformers.sentence_transformer.evaluation import ParaphraseMiningEvaluator


revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
results = []
for size, shape in [(128, 'complete'), (384, 'complete'), (384, 'chain')]:
    nodes = [str(index) for index in range(size)]
    sentences = {node: 'sentence ' + node for node in nodes}
    times = []
    for repeat in range(4):
        if shape == 'complete':
            graph = {node: {other: True for other in nodes if other != node} for node in nodes}
        else:
            graph = {node: {} for node in nodes}
            for first, second in zip(nodes, nodes[1:]):
                graph[first][second] = graph[second][first] = True
        start = time.perf_counter()
        evaluator = ParaphraseMiningEvaluator(sentences, duplicates_dict=graph, add_transitive_closure=True)
        if repeat:
            times.append(time.perf_counter() - start)
        expected_pairs = size * (size - 1) // 2
        assert evaluator.total_num_duplicates == expected_pairs
        digest = hashlib.sha256(json.dumps(evaluator.duplicates, sort_keys=True).encode()).hexdigest()
    result = {'revision': revision, 'nodes': size, 'graph': shape, 'median_seconds': statistics.median(times),
              'duplicate_pairs': expected_pairs, 'closure_sha256': digest}
    if reference is not None:
        expected = reference[len(results)]
        assert result['closure_sha256'] == expected['closure_sha256']
        result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
    results.append(result)
    print(json.dumps(result), flush=True)
Path(revision + '.json').write_text(json.dumps(results, indent=2))
