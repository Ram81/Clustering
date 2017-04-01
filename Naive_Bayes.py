import arff
import numpy as np

from sklearn.naive_bayes import GaussianNB

dataset = arff.load(open('NIMS.arff','r'))
ds = np.array(dataset['data'], dtype = np.float64)
		
data = ds[0:710001, 0:22]
y = ds[0:710001, 22]
		
gnb = GaussianNB()

y1 = gnb.fit(data, y).predict(data)
print("Number of Mislabeled Outputs of total %d points: %d"%(data.shape[0], (y!=y1).sum()))
