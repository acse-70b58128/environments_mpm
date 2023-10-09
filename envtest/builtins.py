import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import misc
from sklearn.datasets import load_iris
import pandas as pd

__all__ = ['rand_array', 'smooth_image', 'my_mat_solve', 'iris_db']


def smooth_image(a, sigma=1):
    return gaussian_filter(a, sigma=sigma)

def rand_array(shape):
    return np.random.rand(*shape)

def my_mat_solve(A, b):
    return A.inv()*b


def iris_db():
    iris = load_iris()
    dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return dataset.head()
