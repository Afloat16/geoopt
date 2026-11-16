import json

import torch
from torch import nn

from sentence_transformers.sentence_transformer.losses.gist_embed import _cosine_similarity_matrix


generator = torch.Generator().manual_seed(13)
left = (torch.randn(7, 3, generator=generator) * 1e-10).T
right = (torch.randn(7, 5, generator=generator) * 1e-10).T
weights = torch.randn(3, 5, generator=generator)
cosine = nn.CosineSimilarity(dim=-1)
grads = {}
for name, dtype in [("native_float32", torch.float32), ("candidate_float32", torch.float32),
                    ("native_float64", torch.float64)]:
    inputs = tuple(value.to(dtype).detach().requires_grad_() for value in (left, right))
    scores = (_cosine_similarity_matrix(*inputs, cosine) if name == "candidate_float32"
              else cosine(inputs[0][:, None], inputs[1][None, :]))
    grads[name] = torch.autograd.grad((scores * weights.to(dtype)).sum(), inputs)
for index, oracle in enumerate(grads["native_float64"]):
    result = {"input": index, "reference_gradient_norm": oracle.norm().item(),
              "reference_max_component": oracle.abs().max().item()}
    for name in ["native_float32", "candidate_float32"]:
        error = grads[name][index].double() - oracle
        result[name] = {"max_absolute_error": error.abs().max().item(),
                        "norm_relative_error": error.norm().item() / oracle.norm().item()}
    discrepancy = grads["native_float32"][index] - grads["candidate_float32"][index]
    result["native_vs_candidate_max_absolute"] = discrepancy.abs().max().item()
    print(json.dumps(result), flush=True)
