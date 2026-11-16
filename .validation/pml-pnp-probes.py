import json
from decimal import Decimal, localcontext

import torch
from pytorch_metric_learning import losses


def reference(matrix, variant):
    with localcontext() as context:
        context.prec = 100
        total = Decimal(0)
        for anchor in range(4):
            for positive in range(4):
                if anchor // 2 != positive // 2:
                    continue
                rank = Decimal(0)
                for negative in range(4):
                    if anchor // 2 == negative // 2:
                        continue
                    delta = (Decimal.from_float(matrix[anchor][negative]) - Decimal.from_float(matrix[anchor][positive])) / Decimal.from_float(.01)
                    rank += Decimal(1) / (Decimal(1) + (-delta).exp())
                if variant == "Ds":
                    value = (1 + rank).ln()
                elif variant == "Dq":
                    value = 1 - (1 + rank) ** -8
                elif variant == "Iu":
                    value = (1 + rank) * (1 + rank).ln()
                elif variant == "Ib":
                    value = (2 * rank - (1 + 2 * rank).ln()) / 4
                else:
                    value = rank
                total += value / 8
        return float(total)


for dtype in (torch.float16, torch.float32, torch.float64):
    for variant in ("Ds", "Dq", "Iu", "Ib", "O"):
        matrix = torch.tensor([[1, .95, .65, .65], [.95, 1, .65, .65], [.65, .65, 1, .95], [.65, .65, .95, 1]], dtype=dtype, requires_grad=True)
        value = losses.PNPLoss(alpha=8, variant=variant)(torch.zeros(4, 2, dtype=dtype), torch.tensor([0, 0, 1, 1]), mat=matrix)
        value.backward()
        print("PROBE=" + json.dumps({"dtype": str(dtype), "variant": variant, "value": value.item(), "reference": reference(matrix.detach().double().tolist(), variant), "gradient_finite": bool(torch.isfinite(matrix.grad).all()), "gradient_max": matrix.grad.abs().max().item()}), flush=True)
