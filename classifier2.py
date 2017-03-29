from time import time
import numpy as np
import matplotlib.pyplot as plt
import arff

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize, StandardScaler
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

#{1 => TELNET, 2 => FTP, 3 => HTTP, 4 => DNS, 5 => lime, 6 => localForwarding, 7 => remoteForwarding, 8 => scp, 9 => sftp, 10 => x11, 11 = > shell}
 
dataset = arff.load(open('NIMS.arff','r'))
ds = np.array(dataset['data'], dtype = np.float64)

data = ds[0:700001, 0:22]

data = scale(data, axis = 1)

#loading size of dataset
n_sample, n_feature = data.shape

#obtaining total number of clusters based on the output i.e. target
n_digits = 11
labels = ds[0:700001, 22]

print('Data Set Shape : ',ds.shape)
print('Number of Output Classes : %d'%(n_digits))

samples = 300
print("\n n_digits: %d, \t samples: %d, \t n_features: %d"%(n_digits, samples, n_feature))
	
#Visualize Data	
reduced_data = PCA(n_components = 3).fit_transform(data)

kmeans = KMeans(init = 'k-means++', n_clusters = n_digits, n_init = 100)

kmeans.fit(reduced_data)

print(kmeans.labels_[0:10],labels[0:10])


predict_ds = ds[700001:700011,0:22]
lbls = ds[700001:700011, 22]

pds = scale(predict_ds, axis = 1)
reduced_pds = PCA(n_components = 3).fit_transform(pds)

print(reduced_pds.shape)

x = kmeans.predict(reduced_pds)
print(x[0:10],lbls[0:10])

#plotting data
fig = plt.figure(1)
plt.clf()

ax = Axes3D(fig, elev = -150, azim = 110)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c = labels, cmap = plt.cm.Paired)

#plotting centroids	
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidths = 2, color = 'w', zorder = 10)	

ax.set_title("K-means clustering on the Traffic dataset (PCA-reduced data)\nCentroids are marked with white cross")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
