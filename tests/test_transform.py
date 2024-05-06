# The MNIST-1D dataset | 2024
# Peter Steinbach

from mnist1d.data import get_dataset, get_dataset_args
from mnist1d.utils import set_seed
from mnist1d.transform import pad

from tempfile import NamedTemporaryFile
import pytest

@pytest.fixture
def tmpfile():
    value = NamedTemporaryFile(delete=True)
    print(value, value.file)
    yield value
    value.close()

def test_padding(tmpfile):
    set_seed(13)
    
    defaults = get_dataset_args()
    data = get_dataset(args=defaults,download=False)
    x = data['x']
    obs = pad(x[0,...],(36,44))
    assert x.shape != obs.shape
    assert obs.shape == (82,)

    obs = pad(x,(36,44))
    assert x.shape != obs.shape
    assert obs.shape == (4000,78)

