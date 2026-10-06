using Serialization

"""
    load_mnist1d(path = joinpath(@__DIR__, "mnist1d.jls"))

Load the frozen MNIST-1D dataset as a `NamedTuple` with fields
`x`, `x_test`, `y`, `y_test`, `t` and `templates` (itself a `NamedTuple`
with fields `x`, `t`, `y`). Array shapes match the Python version,
e.g. `size(data.x) == (4000, 40)`.
"""
load_mnist1d(path = joinpath(@__DIR__, "mnist1d.jls")) = deserialize(path)
