import json
import math

import torch
from pytorch_metric_learning import distances, evaluation, losses


def record(name, function):
    try:
        result = function()
    except Exception as error:
        result = {"exception": type(error).__name__, "message": str(error)}
    print("PROBE=" + json.dumps({"name": name, "result": result}), flush=True)


def nca(scale, dtype):
    x = (torch.eye(4, dtype=dtype) * scale).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    function = losses.NCALoss(distance=distances.LpDistance(normalize_embeddings=False, power=2))
    actual = function(x, labels)
    grad = torch.autograd.grad(actual, x)[0]
    matrix = -torch.cdist(x, x).square()
    eligible = ~torch.eye(4, dtype=torch.bool)
    log_probs = torch.log_softmax(matrix.masked_fill(~eligible, -torch.inf), dim=1)
    expected = -log_probs[torch.arange(4), torch.tensor([1, 0, 3, 2])].mean()
    reference = torch.autograd.grad(expected, x)[0]
    return {"value": actual.item(), "expected": expected.item(), "expected_closed_form": math.log(3), "gradient": grad.tolist(), "reference_gradient": reference.tolist()}


def proxy_nca(dtype):
    x = torch.zeros(1, 3, dtype=dtype, requires_grad=True)
    function = losses.ProxyNCALoss(3, 3, distance=distances.LpDistance(normalize_embeddings=False, power=2)).to(dtype=dtype)
    with torch.no_grad():
        function.weight.copy_(torch.eye(3, dtype=dtype) * 10000)
    actual = function(x, torch.tensor([0]))
    actual.backward()
    return {"value": actual.item(), "expected": math.log(3), "gradient": x.grad.tolist()}


def ranked_list(dtype, temperature):
    x = torch.zeros(4, 2, dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    matrix = torch.tensor([[0, 1.5, .5, .5], [1.5, 0, .5, .5], [.5, .5, 0, 1.5], [.5, .5, 1.5, 0]], dtype=dtype, requires_grad=True)
    function = losses.RankedListLoss(margin=.4, neg_temperature=temperature, pos_temperature=temperature)
    actual = function(x, labels, mat=matrix)
    actual.backward()
    return {"value": actual.item(), "expected": .7, "gradient_finite": bool(torch.isfinite(matrix.grad).all())}


def proxy_temperature(name, dtype):
    x = torch.tensor([[1., 1.]], dtype=dtype, requires_grad=True)
    function = (losses.SoftTripleLoss(2, 2, centers_per_class=2, gamma=1e-5, margin=0) if name == "soft-triple" else losses.NormalizedSoftmaxLoss(2, 2, temperature=1e-5)).to(dtype=dtype)
    with torch.no_grad():
        function.weight.copy_(torch.tensor([1., 0.], dtype=dtype).expand_as(function.weight))
    actual = function(x, torch.tensor([0]))
    actual.backward()
    return {"value": actual.item(), "expected": math.log(2), "gradient_finite": bool(torch.isfinite(x.grad).all()), "weight_gradient_finite": bool(torch.isfinite(function.weight.grad).all())}


def empty_accuracy(custom, self_search):
    query = torch.empty(0, 3)
    labels = torch.empty(0, dtype=torch.long)
    function = evaluation.AccuracyCalculator(label_comparison_fn=torch.eq if custom else None, custom_metrics={"ndcg": evaluation.ndcg_metric(lambda x, y: (x == y).double())})
    args = {} if self_search else {"reference": torch.eye(3), "reference_labels": torch.tensor([0, 1, 2])}
    return {"pooled": function.get_accuracy(query, labels, **args), "classes": function.get_accuracy_per_class(query, labels, **args)}


def dcl(offset, dtype):
    x = torch.zeros(4, 2, dtype=dtype)
    matrix = torch.full((4, 4), offset, dtype=dtype, requires_grad=True)
    function = losses.DecoupledContrastiveLoss(temperature=1, distance=distances.DotProductSimilarity(normalize_embeddings=False))
    actual = function(x, torch.tensor([0, 0, 1, 1]), mat=matrix)
    actual.backward()
    return {"value": actual.item(), "expected": math.log(2), "gradient": matrix.grad.tolist()}


for dtype in (torch.float32, torch.float64):
    record("nca-simplex-" + str(dtype), lambda dtype=dtype: nca(10000 if dtype == torch.float32 else 1e8, dtype))
    record("proxy-nca-" + str(dtype), lambda dtype=dtype: proxy_nca(dtype))
    record("dcl-offset-" + str(dtype), lambda dtype=dtype: dcl(1e8 if dtype == torch.float32 else 1e16, dtype))
for dtype in (torch.float16, torch.float32, torch.float64):
    record("ranked-list-" + str(dtype), lambda dtype=dtype: ranked_list(dtype, 1e5))
    for name in ("soft-triple", "normalized-softmax"):
        record(name + "-" + str(dtype), lambda name=name, dtype=dtype: proxy_temperature(name, dtype))
for custom in (False, True):
    for self_search in (False, True):
        record(f"empty-accuracy-custom{custom}-self{self_search}", lambda custom=custom, self_search=self_search: empty_accuracy(custom, self_search))
