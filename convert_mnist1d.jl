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

# Extract the Numpy arrays
x = data["x"]
x_test = data["x_test"]
y = data["y"]
y_test = data["y_test"]
t = data["t"]
templates = data["templates"]

struct MNIST1D 
    x 
    x_test
    y 
    y_test
    t 
    templates
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
