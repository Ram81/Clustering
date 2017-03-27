from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)


digits = load_digits()
data = scale(digits.data)

#loading size of dataset
n_sample, n_feature = data.shape
print(n_sample," ", n_feature)

#obtaining total number of clusters based on the output i.e. target
n_digits = len(np.unique(digits.target))
labels = digits.target
print(n_digits)

samples = 300
