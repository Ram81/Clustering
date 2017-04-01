import random
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
from mpl_toolkits.mplot3d import Axes3D


def readCsv(f_obj):
	'''
		Function for reading a csv file and obtaining dataset as list representation.
	'''
	dataset = csv.DictReader(f_obj, delimiter = ',')
	data = []
	labels = []
	for row in dataset:
		entry = []
		for i in row:
			if i!='Class':
				entry.append(row[i])
		data.append(list(entry))
		labels.append(row["Class"])

	data = np.array(data, dtype = np.float64)
	labels = np.array(labels, dtype = np.float64)	
	return (data, labels)
	
def kmeans(data, labels):
	'''
		Applying Kmeans on th dataset	
	'''
	kmeans = KMeans(n_clusters = 11, random_state = 0).fit(data)
	#print(kmeans.labels_)

def scaleDataset(data):
	'''
		Scaling the dataset between 1 to -1
	'''
	data = scale(data)
	
	return data

def plot3D(data, output_labels_3d, centroids):
	'''
		Creating a 3d Plot of the dataset
	'''	
	fig = plt.figure(3)
	ax = Axes3D(fig)
	
	for i in range(len(output_labels_3d)):
		if output_labels_3d[i] == 0:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'k')
		elif output_labels_3d[i] == 1:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'r')
		elif output_labels_3d[i] == 2:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'b')
		elif output_labels_3d[i] == 3:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'c')
		elif output_labels_3d[i] == 4:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'g')
		elif output_labels_3d[i] == 5:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'y')
		elif output_labels_3d[i] == 6:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 20, c = 'm')
		elif output_labels_3d[i] == 7:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 25, c = 'y')
		elif output_labels_3d[i] == 8:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 25, c = 'b')
		elif output_labels_3d[i] == 9:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 25, c = 'k')
		elif output_labels_3d[i] == 10:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 25, c = 'm')
		elif output_labels_3d[i] == 11:
			ax.scatter(data[i, 0], data[i, 1], data[i, 2], s = 25, c = 'g')
	
	ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s = 150, c = 'r', marker = 'x', linewidth = 5)
		
	plt.show()
	
	return

def plot2D(data):
	'''
		Create a 2D plot
	'''
	plt.figure(2)
	plt.clf()
	
	plt.plot(data[:, 0], data[:, 1], 'bo', markersize = 3)
	plt.show()
	
	plt.title('Plot of Dataset After Applying PCA')
	
	return

def colorClusters(reduced_data, output_labels):
	'''
		Color Cluster and associated points
	'''
	plt.figure(1)
	plt.clf()
	
	for i in range(len(output_labels)):
		if output_labels[i] == 0:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'k.', markersize = 5)
		elif output_labels[i] == 1:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'r.', markersize = 5)
		elif output_labels[i] == 2:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'b.', markersize = 5)
		elif output_labels[i] == 3:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'c.', markersize = 5)
		elif output_labels[i] == 4:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'g.', markersize = 5)
		elif output_labels[i] == 5:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'y.', markersize = 5)
		elif output_labels[i] == 6:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'm.', markersize = 5)
		elif output_labels[i] == 7:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'y.', markersize = 6)
		elif output_labels[i] == 8:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'b.', markersize = 6)
		elif output_labels[i] == 9:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'k.', markersize = 6)
		elif output_labels[i] == 10:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'm.', markersize = 6)
		elif output_labels[i] == 11:
			plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'g.', markersize = 6)
	return

def applyPCA(data, labels):
	'''
		Applying PCA for visualization.
	'''
	data = scaleDataset(data)	
	
	#Applying PCA for 2 dimensional reduction
	pca_2d = PCA(n_components = 2)
	reduced_data = pca_2d.fit_transform(data)
	
	plot2D(reduced_data)
	
	# Applying kmeans after dimension reduction to 2
	kmeans_2d = KMeans(init = "k-means++", n_clusters = 11, n_init = 10)
	output_labels_2d = kmeans_2d.fit_predict(reduced_data)
	
	colorClusters(reduced_data, output_labels_2d)
	
	centroids = kmeans_2d.cluster_centers_
	print(centroids)
	print('\n')
	plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 125, linewidths = 2, color = 'r', zorder = 10)
	plt.show()
	
	
	#Applying PCA for 3 dimensional reduction
	pca_3d = PCA(n_components = 3)
	reduced_data_3d = pca_3d.fit_transform(data)
	
	#Applying kmeans after dimension reduction to 3
	kmeans_3d = KMeans(init = "k-means++", n_clusters = 11, n_init = 10)
	output_labels_3d = kmeans_3d.fit_predict(reduced_data_3d)
	
	centroids_3d = kmeans_3d.cluster_centers_
	
	plot3D(reduced_data_3d, output_labels_3d, centroids_3d)
	print('\n')
	print(centroids_3d)
	
	dummyEntry1 = [59922,1160690,9285520,69244626,8,350224]
	dummyEntry2 = [32293,4968169,39745352,52902614,66,489930]
	
	while True:
		input_str = input().split(' ')
		predictFeature = np.array(input_str, dtype = np.float64)
	
		dummyEntry_2d = []
		dummyEntry_3d = []
	
		dummyEntry_2d.append(dummyEntry1)
		dummyEntry_2d.append(dummyEntry2)
	
		dummyEntry_3d.append(dummyEntry1)
		dummyEntry_3d.append(dummyEntry2)
	
		dummyEntry_2d.append(predictFeature)
		dummyEntry_3d.append(predictFeature)
		
		#Scaling the prediction feature vector
		dummyEntry_2d = scale(dummyEntry_2d)
		dummyEntry_3d = scale(dummyEntry_3d)
		
		#reducing feature vector using pca in 2 dimension
		reducedFeature_2d = pca_2d.transform(dummyEntry_2d)
		output_2d = kmeans_2d.predict(reducedFeature_2d)
		
		print(output_2d[2])
		
		#reducing feature vector using pca in 3 dimensions
		reducedFeature_3d = pca_3d.transform(dummyEntry_3d)
		output_3d = kmeans_3d.predict(reducedFeature_3d)
		
		print(output_3d[2])
	
	return
	
if __name__ == '__main__':
	with open("dataset.csv") as f_obj:
		data, labels = readCsv(f_obj)
	kmeans(data, labels)
	applyPCA(data,labels)
	
	

