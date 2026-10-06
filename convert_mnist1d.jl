using PythonCall
using NPZ
using Serialization 
using Downloads
using SHA

const URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"

const PKL_FILE = "mnist1d_data.pkl"
const NPZ_FILE = "mnist1d_data.npz"
const JLS_FILE = "mnist1d.jls"

# Downloads = pyimport("urllib.request")
pickle = pyimport("pickle")
numpy = pyimport("numpy")

isfile(PKL_FILE) || Downloads.download(URL, PKL_FILE)

data = pickle.loads(pybytes(read(PKL_FILE)))

# Convert the Numpy arrays to native Julia arrays (Py objects can't be serialized)
x = pyconvert(Array, data["x"])
x_test = pyconvert(Array, data["x_test"])
y = pyconvert(Array, data["y"])
y_test = pyconvert(Array, data["y_test"])
t = pyconvert(Array, data["t"])
# templates is a dict: {'x': ..., 't': ..., 'y': ...}
templates = Dict(pyconvert(String, k) => pyconvert(Array, v) for (k, v) in data["templates"].items())

struct MNIST1D
    x::Array
    x_test::Array
    y::Array
    y_test::Array
    t::Array
    templates::Dict{String,Array}
end

dataset = MNIST1D(
    x,
    x_test,
    y,
    y_test,
    t, 
    templates
)


serialize(JLS_FILE, dataset)

# verify the dataset
data = deserialize("mnist1d.jls")

function describe(name, arr)
    println(
        name, ": ",
        size(arr), ", ",
        eltype(arr), ", ",
        bytes2hex(sha256(reinterpret(UInt8, vec(arr))))
    )
end

for name in (:x, :x_test, :y, :y_test, :t)
    describe(name, getfield(data, name))
end

for k in sort(collect(keys(data.templates)))
    describe("templates[$k]", data.templates[k])
end