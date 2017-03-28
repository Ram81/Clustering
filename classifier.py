from time import time
import numpy as np
import matplotlib.pyplot as plt
import arff

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize

np.random.seed(42)

#{1 => TELNET, 2=> FTP, 3 => HTTP, 4 => DNS, 5 => lime, 6 => localForwarding, 7 => remoteForwarding, 8 => scp, 9 => sftp, 10 => x11, 11 = > shell}
 
dataset = arff.load(open('NIMS.arff','r'))
ds = np.array(dataset['data'], dtype = np.float64)

print(ds.shape)

data = normalize(ds[1:3001, 1:23], norm='l2')

#loading size of dataset
n_sample, n_feature = data.shape
print(n_sample," ", n_feature)

#obtaining total number of clusters based on the output i.e. target
n_digits = 11
labels = ds[1:3001, -1]
print(n_digits)

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
	
#Visualize Data	
reduced_data = PCA(n_components = 2).fit_transform(data)

kmeans = KMeans(init = "k-means++", n_clusters = n_digits, n_init = 10)
kmeans.fit(reduced_data)

#Step Size of mesh
h = .02	
	
#plot decision boundary
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1	
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
#Obtain Labels for each point in mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])	

Z = Z.reshape(xx.shape)

#plotting data
plt.figure(1)
plt.clf()

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)	

#plotting centroids	
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidths = 1, color = 'w', zorder = 10)	

plt.title('K-means clustering on the Traffic dataset (PCA-reduced data)\nCentroids are marked with white cross')
plt.xlim(x_min, x_max)	
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
	
	
	
