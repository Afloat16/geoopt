import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import ADRMSELoss


class ScoreTable(CrossEncoder):
    def __init__(self, scores):
        torch.nn.Module.__init__(self)
        self.scores = torch.nn.Parameter(scores)

    @property
    def num_labels(self):
        return 1

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kwargs):
        return {'indices': torch.tensor([int(document) for _, document in pairs])}

    def forward(self, inputs):
        return {'scores': self.scores[inputs['indices']].unsqueeze(-1)}


revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
results = []
for documents in [128, 512, 2048]:
    generator = torch.Generator().manual_seed(29)
    lengths = [documents, documents // 2, documents - 1, documents // 4]
    count = sum(lengths)
    model = ScoreTable(torch.randn(count, generator=generator))
    loss = ADRMSELoss(model, mini_batch_size=-1)
    queries, rows, labels = [], [], []
    offset = 0
    for index, length in enumerate(lengths):
        queries.append(str(index))
        rows.append([str(i) for i in range(offset, offset + length)])
        labels.append(torch.randint(-2, 4, (length,), generator=generator).float())
        offset += length
    timings = []
    for repeat in range(7):
        model.scores.grad = None
        start = time.perf_counter()
        value = loss((queries, rows), labels)
        value.backward()
        if repeat > 1:
            timings.append(time.perf_counter() - start)
    result = dict(revision=revision, lengths=lengths, median_seconds=statistics.median(timings),
                  loss=value.item(), gradient_sha256=hashlib.sha256(model.scores.grad.numpy().tobytes()).hexdigest())
    if reference:
        expected = reference[len(results)]
        assert result['loss'] == expected['loss']
        assert result['gradient_sha256'] == expected['gradient_sha256']
        result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
    results.append(result)
    print(json.dumps(result), flush=True)
Path(revision + '.json').write_text(json.dumps(results, indent=2))
