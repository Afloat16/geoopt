import json

import torch

from pytorch_metric_learning import distances, losses, miners


records = []


def probe(name, function):
    try:
        records.append({"name": name, "result": function()})
    except Exception as error:
        records.append({"name": name, "exception": type(error).__name__, "message": str(error)})


def dynamic_margins(counts, power):
    model = losses.DynamicArcFaceLoss(counts, 2, power=power, sub_centers=1)
    value = model(torch.tensor([[1., .2], [.1, 1.], [-1., .3]], requires_grad=True), torch.tensor([0, 1, 2]))
    return {"margins": model.class_margins.tolist(), "loss": value.item(), "finite": bool(torch.isfinite(value))}


for counts, power in [([10, 100, 1000], 1000), ([10, 100, 1000], 1e-20), ([2**60, 2**60 + 1, 2**60 + 2], .25), ([10, 100, 1000], .25)]:
    probe(f"dynamic-{counts}-{power}", lambda c=counts, p=power: dynamic_margins(c, p))


def intra_variance(dtype, values, positive):
    matrix = torch.zeros(4, 4, dtype=dtype)
    matrix[0, 1], matrix[2, 3] = values
    matrix.requires_grad_()
    anchors = torch.tensor([0, 2])
    others = torch.tensor([1, 3])
    empty = torch.empty(0, dtype=torch.long)
    pairs = (anchors, others, empty, empty) if positive else (empty, empty, anchors, others)
    model = losses.IntraPairVarianceLoss(pos_eps=0, neg_eps=0, distance=distances.DotProductSimilarity(normalize_embeddings=False))
    value = model(torch.zeros(4, 2, dtype=dtype), indices_tuple=pairs, mat=matrix)
    gradient = torch.autograd.grad(value, matrix)[0]
    pairs_ref = matrix.detach().double()[anchors, others].requires_grad_()
    residual = pairs_ref.mean() - pairs_ref if positive else pairs_ref - pairs_ref.mean()
    expected = residual.clamp_min(0).square().mean()
    expected_gradient = torch.autograd.grad(expected, pairs_ref)[0]
    return {"value": value.item(), "expected": expected.item(), "gradient": gradient[anchors, others].tolist(), "expected_gradient": expected_gradient.tolist(), "dtype": str(value.dtype)}


for dtype, values in [(torch.float16, [.5, .50048828125]), (torch.bfloat16, [.5, .50390625]), (torch.float32, [.5, .50048828125]), (torch.float16, [0., 1000.])]:
    for positive in [False, True]:
        probe(f"intra-{dtype}-{values}-{positive}", lambda d=dtype, v=values, p=positive: intra_variance(d, v, p))


def weighted_mining(dtype, dimension):
    query = torch.zeros(4, dimension, dtype=dtype)
    matrix = torch.full((4, 4), .5, dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1])
    result = miners.DistanceWeightedMiner()(query, labels, mat=matrix)
    return {"tuple_lengths": [len(x) for x in result], "anchors": result[0].tolist()}


for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    for dim in [128, 131072]:
        probe(f"distance-weighted-{dtype}-{dim}", lambda d=dtype, n=dim: weighted_mining(d, n))


print("ADDITIONAL_PROBES=" + json.dumps(records))
