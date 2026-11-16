import gc
import hashlib
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

import sentence_transformers.sentence_transformer.modules.word_embeddings as word_module
from sentence_transformers.sentence_transformer.modules import WordEmbeddings


word_module.tqdm = lambda values, **kwargs: values
torch.set_num_threads(1)
revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
results = []
for vocab_size, dimension in [(5000, 128), (20000, 300)]:
    path = Path(f'word-vector-benchmark-{vocab_size}-{dimension}.txt')
    if not path.exists():
        rng = np.random.default_rng(41)
        with path.open('w', encoding='utf-8') as handle:
            handle.write(f'{vocab_size} {dimension}\n')
            for index in range(vocab_size):
                values = rng.standard_normal(dimension)
                handle.write(f'word{index} ' + ' '.join(format(value, '.7e') for value in values) + '\n')
    times = []
    for repeat in range(4):
        gc.collect()
        start = time.perf_counter()
        model = WordEmbeddings.from_text_file(str(path))
        elapsed = time.perf_counter() - start
        if repeat:
            times.append(elapsed)
        digest = hashlib.sha256(model.emb_layer.weight.detach().numpy().tobytes()).hexdigest()
        vocabulary = hashlib.sha256('\0'.join(model.tokenizer.get_vocab()).encode()).hexdigest()
        del model
    gc.collect()
    tracemalloc.start()
    model = WordEmbeddings.from_text_file(str(path))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del model
    result = {'revision': revision, 'vocab_size': vocab_size, 'dimension': dimension,
              'median_seconds': statistics.median(times), 'peak_traced_bytes': peak,
              'weights_sha256': digest, 'vocabulary_sha256': vocabulary}
    if reference is not None:
        expected = reference[len(results)]
        assert result['weights_sha256'] == expected['weights_sha256']
        assert result['vocabulary_sha256'] == expected['vocabulary_sha256']
        result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
        result['peak_traced_ratio'] = expected['peak_traced_bytes'] / result['peak_traced_bytes']
    results.append(result)
    print(json.dumps(result), flush=True)
Path(revision + '.json').write_text(json.dumps(results, indent=2))
