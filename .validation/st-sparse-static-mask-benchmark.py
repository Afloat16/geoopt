import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding

revision = sys.argv[1]
reference = json.loads(Path(sys.argv[2] + '.json').read_text()) if len(sys.argv) > 2 else None
results = []
dimension = 32768
vocab = {'[PAD]': 0, '[UNK]': 1, **{str(index): index for index in range(2, dimension)}}
tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='[UNK]')),
                                  unk_token='[UNK]', pad_token='[PAD]', model_max_length=256)
for batch in [128, 512]:
    generator = torch.Generator().manual_seed(91)
    module = SparseStaticEmbedding(tokenizer, weight=torch.rand(dimension, generator=generator))
    ids = torch.randint(2, dimension, (batch, 128), generator=generator)
    mask = torch.ones_like(ids)
    ids[:, -32:] = 0
    mask[:, -32:] = 0
    timings = []
    for repeat in range(7):
        module.weight.grad = None
        start = time.perf_counter()
        output = module(dict(input_ids=ids, attention_mask=mask))['sentence_embedding']
        output.square().mean().backward()
        if repeat > 1:
            timings.append(time.perf_counter() - start)
    saved = {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        module(dict(input_ids=ids, attention_mask=mask))
    result = dict(revision=revision, batch=batch, vocabulary=dimension, median_seconds=statistics.median(timings),
                  saved_forward_storage_bytes=sum(saved.values()),
                  output_sha256=hashlib.sha256(output.detach().numpy().tobytes()).hexdigest(),
                  gradient_sha256=hashlib.sha256(module.weight.grad.numpy().tobytes()).hexdigest())
    if reference:
        expected = reference[len(results)]
        assert result['output_sha256'] == expected['output_sha256']
        assert result['gradient_sha256'] == expected['gradient_sha256']
        result['speed_ratio'] = expected['median_seconds'] / result['median_seconds']
    results.append(result)
    print(json.dumps(result), flush=True)
Path(revision + '.json').write_text(json.dumps(results, indent=2))
