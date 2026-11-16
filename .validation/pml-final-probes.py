import json
import math

import torch

from pytorch_metric_learning import distances, losses
from pytorch_metric_learning.evaluation.accuracy_calculator import RetrievalInputs, ndcg


records = []
for dtype, scale in [(torch.float32, 1e20), (torch.float64, 1e200), (torch.bfloat16, 1e20)]:
    query = torch.tensor([[3 * scale, 4 * scale]], dtype=dtype, requires_grad=True)
    reference = torch.tensor([[1., 0.]], dtype=dtype)
    value = distances.CosineSimilarity().pairwise(query, reference)
    gradient = torch.autograd.grad(value.sum(), query)[0]
    records.append(dict(case='normalization', dtype=str(dtype), value=value.tolist(), gradient=gradient.tolist(), expected=.6))

for scale in [1., 1e308, 1e-320]:
    def relevance(query_labels, references):
        return torch.tensor([1.5, 1.3, 1.], dtype=torch.float64)[references] * scale
    inputs = RetrievalInputs(torch.tensor([0]), torch.tensor([[1, 0, 2]]), torch.tensor([[False, True, False]]), torch.tensor([1]), torch.tensor([0, 1, 2]), False)
    actual = ndcg(inputs, relevance)
    expected = (1.3 + 1.5 / math.log2(3) + .5) / (1.5 + 1.3 / math.log2(3) + .5)
    records.append(dict(case='ndcg', scale=scale, value=actual.tolist(), expected=expected))

for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    matrix = torch.full((4, 4), .5, dtype=dtype, requires_grad=True)
    actual = losses.CircleLoss(margin=.5, gamma=1e5)(torch.zeros(4, 2, dtype=dtype), torch.tensor([0, 0, 1, 1]), mat=matrix)
    gradient = torch.autograd.grad(actual, matrix)[0]
    records.append(dict(case='circle', dtype=str(dtype), value=actual.item(), gradient=gradient.tolist(), expected=math.log(3)))

for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    model = losses.P2SGradLoss(1, 2).to(dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.]], dtype=dtype))
    query = torch.tensor([[1., .02]], dtype=dtype, requires_grad=True)
    value = model(query, torch.tensor([0]))
    gradient = torch.autograd.grad(value, (query, model.weight))
    tilt = float(query.detach()[0, 1])
    expected = (1 / math.hypot(1, tilt) - 1)**2
    records.append(dict(case='p2s', dtype=str(dtype), value=value.item(), gradient=[x.tolist() for x in gradient], expected=expected))

model = losses.P2SGradLoss(1, 2)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[1., 0.]]))
query = torch.tensor([[1., .02]], requires_grad=True)
with torch.autocast('cpu', dtype=torch.bfloat16):
    value = model(query, torch.tensor([0]))
gradient = torch.autograd.grad(value, (query, model.weight))
records.append(dict(case='p2s-autocast', value=value.item(), gradient=[x.tolist() for x in gradient]))
print('FINAL_PROBES=' + json.dumps(records))
