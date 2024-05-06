# The MNIST-1D dataset | 2020
# Sam Greydanus

import numpy as np
import os
import requests
import mnist1d
from mnist1d.transform import transform
from mnist1d.utils import from_pickle, to_pickle, ObjectView, set_seed

def get_dataset_args(as_dict=False):
    """ Generate dictionary with dataset properties

    Parameters
    ----------
    as_dict : bool, optional
        if true, return the dataset properties as dictionary; if false, return an ObjectView, by default False

    Returns
    -------
    _type_
        _description_
    """
    arg_dict = {'num_samples': 5000,
            'train_split': 0.8,
            'template_len': 12,
            'padding': [36,60],
            'scale_coeff': .4, 
            'max_translation': 48,
            'corr_noise_scale': 0.25,
            'iid_noise_scale': 2e-2,
            'shear_scale': 0.75,
            'shuffle_seq': False,
            'final_seq_length': 40,
            'seed': 42,
            'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'}
    return arg_dict if as_dict else ObjectView(arg_dict)


# basic 1D templates for the 10 digits
def get_templates():
    d0 = np.asarray([5,6,6.5,6.75,7,7,7,7,6.75,6.5,6,5])
    d1 = np.asarray([5,3,3,3.4,3.8,4.2,4.6,5,5.4,5.8,5,5])
    d2 = np.asarray([5,6,6.5,6.5,6,5.25,4.75,4,3.5,3.5,4,5])
    d3 = np.asarray([5,6,6.5,6.5,6,5,5,6,6.5,6.5,6,5])
    d4 = np.asarray([5,4.4,3.8,3.2,2.6,2.6,5,5,5,5,5,5])
    d5 = np.asarray([5,3,3,3,3,5,6,6.5,6.5,6,4.5,5])
    d6 = np.asarray([5,4,3.5,3.25,3,3,3,3,3.25,3.5,4,5])
    d7 = np.asarray([5,7,7,6.6,6.2,5.8,5.4,5,4.6,4.2,5,5])
    d8 = np.asarray([5,4,3.5,3.5,4,5,5,4,3.5,3.5,4,5])
    d9 = np.asarray([5,4,3.5,3.5,4,5,5,5,5,4.7,4.3,5])
    
    x = np.stack([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9])
    x -= x.mean(1,keepdims=True) # whiten
    x /= x.std(1,keepdims=True)
    x -= x[:,:1]  # signal starts and ends at 0
    
    templates = {'x': x/6., 't': np.linspace(-5, 5, len(d0))/6.,
                 'y': np.asarray([0,1,2,3,4,5,6,7,8,9])}
    return templates


# make a dataset
def make_dataset(args=None, template=None, ):
    templates = get_templates() if template is None else template
    args = get_dataset_args() if args is None else args
    set_seed(args.seed) # reproducibility
    
    xs, ys = [], []
    samples_per_class = args.num_samples // len(templates['y'])
    for label_ix in range(len(templates['y'])):
        for example_ix in range(samples_per_class):
            x = templates['x'][label_ix]
            t = templates['t']
            y = templates['y'][label_ix]
            x, new_t = transform(x, t, args) # new_t transformation is same each time
            xs.append(x) ; ys.append(y)
    
    batch_shuffle = np.random.permutation(len(ys)) # shuffle batch dimension
    xs = np.stack(xs)[batch_shuffle]
    ys = np.stack(ys)[batch_shuffle]
    
    if args.shuffle_seq: # maybe shuffle the spatial dimension
        seq_shuffle = np.random.permutation(args.final_seq_length)
        xs = xs[...,seq_shuffle]
    
    new_t = new_t/xs.std()
    xs = (xs-xs.mean())/xs.std() # center the dataset & set standard deviation to 1

    # train / test split
    split_ix = int(len(ys)*args.train_split)
    dataset = {'x': xs[:split_ix], 'x_test': xs[split_ix:],
               'y': ys[:split_ix], 'y_test': ys[split_ix:],
               't':new_t, 'templates': templates}
    return dataset



# we'll cache the dataset so that it doesn't have to be rebuild every time
# args must not be a dict
def get_dataset(args, path=None, verbose=True, download=True, regenerate=False, **kwargs):
    if 'args' in kwargs.keys() and kwargs['args'].shuffle_seq:
        shuffle = "_shuffle"
    else:
        shuffle = ""
    path = './mnist1d_data{}.pkl'.format(shuffle) if path is None else path

    assert not (download and regenerate), "You can either download the o.g. MNIST1D dataset or generate your own - but not both"
    try:
        if regenerate:
            raise ValueError("Regenerating dataset") # yes this is hacky
        if download:
            if os.path.exists(path):
                if verbose:
                    print("File already exists. Skipping download.")
            else:
                print("Downloading MNIST1D dataset from {}".format(args.url))
                r = requests.get(args.url, allow_redirects=True)
                open(path, 'wb').write(r.content)
                print("Saving to {}".format(path))
        dataset = from_pickle(path)
        if verbose:
            print("Successfully loaded data from {}".format(path))
    except:
        if verbose:
            print("Did or could not load data from {}. Rebuilding dataset...".format(path))
        dataset = make_dataset(args, **kwargs)
        to_pickle(dataset, path)
    return dataset