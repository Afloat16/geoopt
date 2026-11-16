import json
import statistics
import sys
import time

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss, GISTEmbedLoss


class Tokenizer(PreTrainedTokenizerBase):
    def get_vocab(self):
        return {"token": 0}


class EmbeddingInput(nn.Module):
    max_seq_length = 16

    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.input_module = nn.Identity()

    def __getitem__(self, index):
        return self.input_module

    def forward(self, features):
        return {"sentence_embedding": features["sentence_embedding"]}


torch.set_num_threads(1)
prefix = sys.argv[1]
all_outputs = []
all_results = []
for loss_class, batch, dim in [(GISTEmbedLoss, 64, 384), (GISTEmbedLoss, 128, 768), (CachedGISTEmbedLoss, 128, 384)]:
    kwargs = {"mini_batch_size": 32} if loss_class is CachedGISTEmbedLoss else {}
    loss = loss_class(EmbeddingInput(), EmbeddingInput(), **kwargs)
    torch.manual_seed(1)
    columns = [torch.randn(batch, dim, requires_grad=True) for _ in range(3)]

    def calculate():
        if loss_class is CachedGISTEmbedLoss:
            reps = [list(column.split(32)) for column in columns]
            guides = [[part.detach() for part in column] for column in reps]
            return loss.calculate_loss(reps, guides)
        return loss([{"sentence_embedding": column} for column in columns], None)

    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        value = calculate()
    value.backward()
    all_outputs.append([value.detach()] + [column.grad.clone() for column in columns])
    times = []
    for repeat in range(9):
        for column in columns:
            column.grad = None
        start = time.perf_counter()
        calculate().backward()
        if repeat >= 2:
            times.append(time.perf_counter() - start)
    result = {"loss": loss_class.__name__, "batch": batch, "dim": dim, "median_seconds": statistics.median(times),
              "unique_saved_tensor_storage_bytes": sum(storages.values())}
    all_results.append(result)
    print(json.dumps({"revision": prefix, **result}), flush=True)
torch.save(all_outputs, prefix + "-outputs.pt")
with open(prefix + "-benchmark.json", "w") as file:
    json.dump(all_results, file)
if len(sys.argv) > 2:
    reference = torch.load(sys.argv[2] + "-outputs.pt", weights_only=True)
    for actual_case, expected_case in zip(all_outputs, reference):
        for actual, expected in zip(actual_case, expected_case):
            torch.testing.assert_close(actual, expected)
    with open(sys.argv[2] + "-benchmark.json") as file:
        before = json.load(file)
    print("ALL LOSS AND GRADIENT VALUES MATCH", flush=True)
    print(json.dumps([{"loss": after["loss"], "batch": after["batch"],
                       "speed_ratio": old["median_seconds"] / after["median_seconds"],
                       "saved_storage_ratio": old["unique_saved_tensor_storage_bytes"] / after["unique_saved_tensor_storage_bytes"]}
                      for old, after in zip(before, all_results)]), flush=True)
