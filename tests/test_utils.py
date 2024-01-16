# The MNIST-1D dataset | 2024
# Peter Steinbach

from mnist1d.data import get_dataset, get_dataset_args
from mnist1d.utils import set_seed, to_pickle, from_pickle, ObjectView
import numpy as np

from tempfile import NamedTemporaryFile
import pytest

@pytest.fixture
def tmpfile():
    value = NamedTemporaryFile(delete=True)
    print(value, value.file.name)
    yield value.file.name
    value.close()

def test_seed_is_fixed():

    set_seed(42)
    exp = np.random.randn(8)
    set_seed(42)
    obs = np.random.randn(8)

    assert np.allclose(exp,obs)    

def test_pickle_roundtrip(tmpfile):

    exp = dict(a=3,b=11.01,c="d",d="foobar")
    _ = to_pickle(exp,tmpfile)
    obs = from_pickle(tmpfile)

    assert type(exp) == type(obs)
    for k,v in exp.items():
        assert v == obs[k]

def test_object_view():
    #TODO: IIRC there are better ways to do this
    exp = dict(a=3,b=11.01,c="d",d="foobar")
    obs = ObjectView(exp)

    assert type(obs) != type(exp)
    assert obs.a == exp['a']