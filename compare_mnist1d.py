import hashlib
import pickle

import numpy as np

PKL_FILE = "mnist1d_data.pkl"

# Julia names for numpy dtypes, so output matches convert_mnist1d.jl
JULIA_TYPES = {
    "float64": "Float64", "float32": "Float32",
    "int64": "Int64", "int32": "Int32",
}


def fmt_shape(shape):
    # Julia prints 1-tuples as (n,), same as Python
    return str(tuple(shape))


def describe(name, arr):
    arr = np.asarray(arr)
    # Julia arrays are column-major, so hash bytes in Fortran order
    digest = hashlib.sha256(arr.tobytes(order="F")).hexdigest()
    eltype = JULIA_TYPES.get(str(arr.dtype), str(arr.dtype))
    print(f"{name}: {fmt_shape(arr.shape)}, {eltype}, {digest}")


with open(PKL_FILE, "rb") as f:
    data = pickle.load(f)

for name in ("x", "x_test", "y", "y_test", "t"):
    describe(name, data[name])

for k in sorted(data["templates"]):
    describe(f"templates[{k}]", data["templates"][k])
