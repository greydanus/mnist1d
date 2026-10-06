# The MNIST-1D dataset | 2024
# Peter Steinbach
from mnist1d.data import get_templates, make_dataset, get_dataset, get_dataset_args
import numpy as np
import pytest

def test_get_templates():
    obs = get_templates()
    assert isinstance(obs, dict)
    assert 'x' in obs.keys()
    assert 'y' in obs.keys()
    assert 't' in obs.keys()

    x,y,t = obs['x'], obs['y'], obs['t']
    assert isinstance(y,np.ndarray)
    assert (y == np.asarray(list(range(10)))).all()

    assert x.shape == (10,12)
    assert y.shape == (10,)

def test_make_dataset():

    obs = make_dataset()
    assert isinstance(obs, dict)
    assert 'x' in obs.keys()
    assert 'y' in obs.keys()
    assert 't' in obs.keys()
    assert 'x_test' in obs.keys()
    assert 'y_test' in obs.keys()
    assert 'templates' in obs.keys()

    x,y,t = obs['x'], obs['y'], obs['t']
    x_test,y_test = obs['x_test'], obs['y_test']

    assert x.shape == (4000,40)
    assert y.shape == (4000,)

    assert x_test.shape == (1000,40)
    assert y_test.shape == (1000,)

def test_get_dataset_args():
    defaults = get_dataset_args(as_dict=True)
    assert 'seed' in defaults.keys()
    assert defaults['seed'] == 42

def test_get_dataset(tmp_path):

    defaults = get_dataset_args(as_dict=False)
    tmp = str(tmp_path / "mnist1d_data.pkl")

    obs = get_dataset(args=defaults,path=tmp)
    assert isinstance(obs, dict)
    assert 'x' in obs.keys()
    assert 'y' in obs.keys()
    assert 't' in obs.keys()
    assert 'x_test' in obs.keys()
    assert 'y_test' in obs.keys()
    assert 'templates' in obs.keys()
