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
#print(n_sample," ", n_feature)

#obtaining total number of clusters based on the output i.e. target
n_digits = len(np.unique(digits.target))
labels = digits.target
#print(n_digits)

samples = 300
print("\n n_digits: %d, \t samples: %d, \t n_features: %d"%(n_digits, samples, n_feature))


print(79*'_')

print(' %9s '%'init      time inertia homo compl v-meas ARI AMI Silhouette\n')

#here estimator is the model being fitted to te data or applied to the data
def analyze_k_means(estimator, name, data):
	t0 = time()
	estimator.fit(data)	
	print(" %9s %.2fs %i %.3f %.3f %.3f %.3f %.3f %.3f"%( name, time()-t0, estimator.inertia_, metrics.homogeneity_score(labels,  estimator.labels_), metrics.completeness_score(labels, estimator.labels_), metrics.v_measure_score(labels, estimator.labels_), metrics.adjusted_rand_score(labels, estimator.labels_), metrics.adjusted_mutual_info_score(labels, estimator.labels_), metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size = samples) ))
	

analyze_k_means( KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 10), name = "k-means++", data = data)	


analyze_k_means( KMeans(init = 'random', n_clusters = n_digits, n_init = 10), name = "random", data = data)


pca = PCA(n_components = n_digits).fit(data)
# in this case the seeding of the centers is deterministic, hence we run the kmeans algorithm only once with n_init=1
analyze_k_means( KMeans(init = pca.components_, n_clusters = n_digits, n_init = 1), name = "PCA", data = data )

print('\n')

print(79*'_')	
	
	
	
	
	
	
	
	
	
	
	
	
