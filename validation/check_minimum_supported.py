import json, subprocess, sys
cases = [{"pr":304,"sha":"a86617d9f6cd92dd723dabd40e6666e5362f0982","tests":["tests/test_sparse_sgd_first_momentum.py","tests/test_sparse_rsgd.py"]},{"pr":305,"sha":"5491ed5069a606bf40c2c9c81f46b8a66d9dc58c","tests":["tests/test_spectral_reconstruction.py","tests/test_manifold_basic.py","tests/test_cayley_transform.py"]},{"pr":306,"sha":"7367031575097c0479039c59cb02280e7a941cea","tests":["tests/test_spd_solve_operations.py","tests/test_manifold_basic.py"]},{"pr":307,"sha":"2846064db5d6d637cf683167fc257507d301a8af","tests":["tests/test_stiefel_cayley.py","tests/test_manifold_basic.py"]},{"pr":308,"sha":"5162a2a7cc5147545ddaa3ee5dc8fe98d8d965f3","tests":["tests/test_cayley_solve.py","tests/test_manifold_basic.py","tests/test_cayley_transform.py"]},{"pr":309,"sha":"8d0bbddf2f2d2fc748daea2b4500300b0840fab1","tests":["tests/test_spd_transport_sqrt.py","tests/test_manifold_basic.py"]},{"pr":310,"sha":"78c37b66257412094c6cc6d54d58bd55c4044947","tests":["tests/test_stiefel_expmap_blocks.py","tests/test_manifold_basic.py"]}]
failed = []
for case in cases:
    print("MINIMUM_VERSION_PR=" + str(case["pr"]), flush=True)
    subprocess.run(["git", "fetch", "--depth=1", "origin", case["sha"]], check=True)
    subprocess.run(["git", "checkout", "--detach", case["sha"]], check=True)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *case["tests"]])
    print("MINIMUM_VERSION_RESULT=" + json.dumps({"pr": case["pr"], "sha": case["sha"], "returncode": result.returncode}), flush=True)
    if result.returncode:
        failed.append(case["pr"])
sys.exit(bool(failed))
