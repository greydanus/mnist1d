# MNIST-1D for Julia

`mnist1d.jls` contains the same frozen dataset as `../mnist1d_data.pkl`,
saved with Julia's built-in `Serialization` standard library, so no packages
are needed to load it.

```julia
julia> include("julia/mnist1d.jl")

julia> data = load_mnist1d();

julia> size(data.x), size(data.y), size(data.templates.x)
((4000, 40), (4000,), (10, 12))
```

Or, without the helper: `using Serialization; data = deserialize("julia/mnist1d.jls")`.

Note: `Serialization` files are only guaranteed to load in the same (or a newer)
Julia version as the one that wrote them. This file was written with Julia 1.13.
