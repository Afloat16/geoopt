import inspect
import json
import traceback

import torch
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses, miners, regularizers
from pytorch_metric_learning.utils import indexing


def run(name, fn):
    try:
        print("PROBE=" + json.dumps({"name": name, "result": fn()}, default=str), flush=True)
    except Exception as exc:
        print("PROBE=" + json.dumps({"name": name, "error": repr(exc)}), flush=True)
        traceback.print_exc()


def raw_gradients():
    result = {"torch": torch.__version__, "source": inspect.getfile(losses.ArcFaceLoss)}
    for name, fn in {
        "clamp": lambda x: x.clamp(-1, 1),
        "cos_acos": lambda x: x.acos().cos(),
        "sqrt_zero": lambda x: x.sqrt(),
    }.items():
        x = torch.tensor([0.0] if name == "sqrt_zero" else [-1.0, -1 + 1e-10, -0.3, 0.2, 1 - 1e-10, 1.0], dtype=torch.float64, requires_grad=True)
        y = fn(x)
        y.sum().backward()
        result[name] = {"value": y.tolist(), "gradient": x.grad.tolist()}
    return result


def arcface_training():
    result = []
    for dtype in [torch.float16, torch.float32, torch.float64]:
        for margin in [0.0, 0.5]:
            for offset in [0.1, 0.01, 1e-5]:
                x = torch.tensor([[1.0, offset]], dtype=dtype, requires_grad=True)
                loss = losses.ArcFaceLoss(2, 2, margin=margin, scale=1).to(dtype)
                with torch.no_grad():
                    loss.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=dtype))
                y = loss(x, torch.tensor([1]))
                y.backward()
                actual_x = x.grad.detach().clone()
                actual_w = loss.weight.grad.detach().clone()
                xx = x.detach().double().requires_grad_()
                ww = loss.weight.detach().double().requires_grad_()
                cosine = F.normalize(xx, dim=1) @ F.normalize(ww, dim=1).T
                logits = cosine.clone()
                logits[:, 1] = cosine[:, 1] - margin * torch.sin(torch.tensor(margin, dtype=torch.float64))
                reference = F.cross_entropy(logits, torch.tensor([1]))
                reference.backward()
                result.append({"dtype": str(dtype), "margin": margin, "offset": offset, "loss": y.item(), "reference_loss": reference.item(), "embedding_gradient": actual_x.tolist(), "reference_embedding_gradient": xx.grad.tolist(), "weight_gradient": actual_w.tolist(), "reference_weight_gradient": ww.grad.tolist()})
    return result


def weighted_sampling():
    labels = torch.tensor([0, 0, 1, 1])
    result = []
    for scale in [1.0, 1e-50, 1e50]:
        weights = torch.ones(4, 4, dtype=torch.float64) * scale
        try:
            torch.manual_seed(0)
            out = indexing.get_random_triplet_indices(labels, weights=weights, t_per_anchor=2)
            result.append({"scale": scale, "indices": [v.tolist() for v in out]})
        except Exception as exc:
            result.append({"scale": scale, "error": repr(exc)})
    return result


def spherical_regularizer():
    result = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        x = torch.tensor([[0.0, 0.0], [1000.0, 0.0]], dtype=dtype, requires_grad=True)
        value = regularizers.SphericalEmbeddingRegularizer()(x)
        weighted = value * 0.0001
        weighted.backward()
        result.append({"dtype": str(dtype), "result_dtype": str(value.dtype), "loss": value.item(), "weighted": weighted.item(), "gradient": x.grad.tolist()})
    return result


def miner_extrema():
    result = []
    for dtype in [torch.float16, torch.float32, torch.float64]:
        for inverted in [False, True]:
            distance = distances.DotProductSimilarity(normalize_embeddings=False) if inverted else distances.LpDistance(normalize_embeddings=False)
            miner = miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard", distance=distance)
            labels = torch.tensor([0, 0, 1, 1])
            embeddings = torch.zeros(4, 2, dtype=dtype)
            extreme = torch.finfo(dtype).min if inverted else torch.finfo(dtype).max
            mat = torch.full((4, 4), extreme, dtype=dtype)
            out = miner(embeddings, labels, mat=mat)
            result.append({"dtype": str(dtype), "inverted": inverted, "indices": [v.tolist() for v in out]})
    return result


def multi_supcon_matrix():
    result = []
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool)
    matrix = torch.tensor([[0.1, 0.7, 0.2, 0.3], [0.6, 0.1, 0.3, 0.5], [0.1, 0.2, 0.1, 0.5], [0.4, 0.3, 0.8, 0.1]], dtype=torch.float64)
    for offset in [0.0, 1e8]:
        for dtype in [torch.float16, torch.float32, torch.float64]:
            mat = (matrix + offset).detach().requires_grad_()
            x = torch.ones(4, 2, dtype=dtype)
            value = losses.MultiSupConLoss(temperature=1)(x, labels, mat=mat)
            value.backward()
            result.append({"embedding_dtype": str(dtype), "offset": offset, "loss": value.item(), "gradient": mat.grad.tolist()})
    return result


run("raw_gradients", raw_gradients)
run("arcface_training", arcface_training)
run("weighted_sampling", weighted_sampling)
run("spherical_regularizer", spherical_regularizer)
run("miner_extrema", miner_extrema)
run("multi_supcon_matrix", multi_supcon_matrix)
