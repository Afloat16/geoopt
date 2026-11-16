import json
import platform
import sys
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from mlxtend.frequent_patterns.apriori import generate_new_combinations_low_memory
from mlxtend.preprocessing import minmax_scaling
from mlxtend.evaluate import bias_variance_decomp

print("VERSIONS=" + json.dumps({"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}))
X = sparse.csc_matrix((np.ones(500, dtype=bool), (np.zeros(500, dtype=int), np.arange(500))), shape=(50000, 500))
tracemalloc.start()
start = time.perf_counter()
first = next(generate_new_combinations_low_memory(np.arange(500).reshape(-1, 1), X, 1 / 50000, True))
elapsed = time.perf_counter() - start
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print("APRIORI=" + json.dumps({"shape": X.shape, "nnz": X.nnz, "first_support": int(first), "peak_bytes": peak, "seconds": elapsed}))

for values, limits in [(np.array([[-1e308], [0.], [1e308]]), (0., 1.)),
                       (np.array([[0.], [0.5], [1.]]), (-1e308, 1e308))]:
    with warnings.catch_warnings(record=True) as caught:
        result = minmax_scaling(values, columns=[0], min_val=limits[0], max_val=limits[1])
    print("MINMAX=" + json.dumps({"input": values.tolist(), "limits": limits, "output": result.tolist(), "warnings": [str(w.message) for w in caught]}))

X_train = np.arange(12, dtype=float).reshape(-1, 1)
y_train = np.sin(X_train[:, 0]) + X_train[:, 0] / 4
X_test = np.linspace(0., 11., 21).reshape(-1, 1)
y_test = np.sin(X_test[:, 0]) + X_test[:, 0] / 4
template = RandomForestRegressor(n_estimators=3, max_depth=2, warm_start=True, random_state=17)
with warnings.catch_warnings(record=True) as caught:
    actual = bias_variance_decomp(template, X_train, y_train, X_test, y_test, loss="mse", num_rounds=10, random_seed=42)
rng = np.random.RandomState(42)
predictions = []
for _ in range(10):
    indices = rng.choice(np.arange(len(X_train)), size=len(X_train), replace=True)
    model = clone(template).fit(X_train[indices], y_train[indices])
    predictions.append(model.predict(X_test))
predictions = np.asarray(predictions)
expected = (np.mean((predictions-y_test)**2), np.mean((predictions.mean(axis=0)-y_test)**2), np.mean((predictions-predictions.mean(axis=0))**2))
print("BIAS_VARIANCE=" + json.dumps({"actual": actual, "fresh_fit_reference": expected, "warnings": [str(w.message) for w in caught]}))
