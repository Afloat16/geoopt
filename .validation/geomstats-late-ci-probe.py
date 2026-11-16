"""Compare late upstream failures on recorded common inputs, without changing tests."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    ReparametrizationBundle,
)
from geomstats.geometry.positive_lower_triangular_matrices import UnitNormedRowsPLTMatrices
from geomstats.learning.aac import AACRegression
from geomstats.metric_geometry.graph_space import GraphSpace
from geomstats.test.random import RandomDataGenerator


def payload_for(kind, seed):
    gs.random.seed(seed)
    if kind == "curves":
        return {
            "base": [[0.1902, -0.5485], [0.6530, -0.8989], [1.2325, -0.9332]],
            "provenance": "Four-decimal base curve printed in official job 133188695757; original full precision was not logged.",
        }
    if kind == "aac":
        space = GraphSpace(3 + seed % 2)
        return {
            "point": gs.to_numpy(space.random_point()).tolist(),
            "n_samples": 2,
            "provenance": "Recorded deterministic draw matching the official fixture; original graph was not logged.",
        }
    space = UnitNormedRowsPLTMatrices(2 + seed % 4)
    generator = RandomDataGenerator(space, amplitude=5.0)
    base = generator.random_point(4)
    direction = generator.random_tangent_vec(base)
    tangent = generator.random_tangent_vec(base)
    return {
        "n": space.n,
        "base": gs.to_numpy(base).tolist(),
        "direction": gs.to_numpy(direction).tolist(),
        "tangent": gs.to_numpy(tangent).tolist(),
        "provenance": "Recorded deterministic draw matching the official fixture; original matrices were not logged.",
    }


def evaluate(kind, payload):
    if kind == "curves":
        space = DiscreteCurvesStartingAtOrigin(k_sampling_points=4)
        space.fiber_bundle = ReparametrizationBundle(
            space, aligner=DynamicProgrammingAligner(space)
        )
        base = gs.array(payload["base"])
        point = space.projection(space.interpolate(base)(gs.linspace(0.0, 1.0, 4) ** 2))
        aligned = space.fiber_bundle.align(point, base)
        difference = gs.abs(aligned - base)
        return {
            "passed": bool(gs.allclose(aligned, base, atol=0.1)),
            "values": gs.to_numpy(aligned).tolist(),
            "max_error": float(gs.amax(difference)),
        }
    if kind == "aac":
        point = gs.array(payload["point"])
        space = GraphSpace(point.shape[-1])
        space.equip_with_group_action()
        space.equip_with_quotient()
        space.aligner.set_alignment_algorithm("exhaustive")
        estimator = AACRegression(space, init_point=gs.zeros(point.shape))
        y = gs.broadcast_to(point, (payload["n_samples"],) + point.shape)
        x = gs.expand_dims(gs.linspace(0.0, 1.0, payload["n_samples"]), -1)
        prediction = estimator.fit(x, y).predict(x)
        distances = space.metric.dist(prediction, y)
        quotient = space.quotient.metric.dist(prediction, y)
        return {
            "passed": bool(gs.allclose(distances, gs.zeros_like(distances), atol=1e-6)),
            "values": gs.to_numpy(distances).tolist(),
            "quotient_distances": gs.to_numpy(quotient).tolist(),
        }
    space = UnitNormedRowsPLTMatrices(payload["n"])
    base, direction, tangent = [gs.array(payload[key]) for key in ("base", "direction", "tangent")]
    transported = space.metric.parallel_transport(tangent, base, direction=direction)
    end = space.metric.exp(direction, base)
    tangent_check = space.is_tangent(transported, end, atol=1e-6)
    return {
        "passed": bool(gs.all(tangent_check)),
        "values": gs.to_numpy(tangent_check).tolist(),
        "transported": gs.to_numpy(transported).tolist(),
        "end": gs.to_numpy(end).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["curves", "aac", "plt_tangent"])
    parser.add_argument("output")
    parser.add_argument("--replay")
    parser.add_argument("--max-seeds", type=int, default=500)
    args = parser.parse_args()
    touched = set()

    def profile(frame, event, arg):
        if event != "call":
            return
        filename = frame.f_code.co_filename
        if args.kind == "curves":
            if filename.endswith("geomstats/geometry/hypersphere.py") and frame.f_code.co_qualname == "_HypersphereVonMisesFisher._sample_at_first_axis":
                touched.add(frame.f_code.co_qualname)
        elif filename.endswith("geomstats/numerics/ivp.py"):
            touched.add(frame.f_code.co_qualname)

    # Verify the trace predicate with a real call before relying on zero hits.
    sys.setprofile(profile)
    if args.kind == "curves":
        from geomstats.geometry.hypersphere import Hypersphere, VonMisesFisher
        VonMisesFisher(Hypersphere(3), kappa=1.0).sample(1)
    else:
        from geomstats.numerics.ivp import ScipySolveIVP
        ScipySolveIVP().integrate_t(lambda state, time: state, gs.array([1.0]), gs.array([0.0, 1.0]))
    sys.setprofile(None)
    assert touched, "Trace predicate did not observe its positive control"
    touched.clear()

    if args.replay:
        baseline = json.loads(Path(args.replay).read_text())
        payload = baseline["payload"]
        seed = baseline["seed"]
    else:
        for seed in range(1 if args.kind == "curves" else args.max_seeds):
            payload = payload_for(args.kind, seed)
            result = evaluate(args.kind, payload)
            if not result["passed"]:
                break
        else:
            raise AssertionError("No matching failure found within the recorded seed bound")
    sys.setprofile(profile)
    result = evaluate(args.kind, payload)
    sys.setprofile(None)
    assert not result["passed"], result
    assert not touched, sorted(touched)
    if args.replay:
        assert result.keys() == baseline["result"].keys()
        for key, value in result.items():
            if key == "passed":
                assert value == baseline["result"][key]
            else:
                np.testing.assert_allclose(value, baseline["result"][key], rtol=0, atol=0, equal_nan=True)
    record = {
        "kind": args.kind, "seed": seed, "payload": payload, "result": result,
        "changed_functions_called": sorted(touched), "trace_positive_control_passed": True,
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
    }
    if "torch" in sys.modules:
        record["versions"]["torch"] = sys.modules["torch"].__version__
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    summary = {k: v for k, v in record.items() if k not in {"payload", "result"}}
    summary["result"] = {k: v for k, v in result.items() if k not in {"transported", "end"}}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
