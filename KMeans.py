import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import pyplot
style.use("ggplot")
import sklearn
from scipy.cluster.vq import kmeans,vq
from sklearn import cluster
from sklearn import preprocessing
from scipy.spatial import distance
import sklearn.datasets
import pandas as pd




f = '/Users/jeromescelza/Downloads/collection_id-ee03df8e-9f0a-4401-86c8-e12a5ff0493f-14575497083922701883.csv'

sample_df = pd.read_csv(f, skiprows=0)

#col = []
#
#
#for i in range(1,332):
#    col.append('S%s' % i) 
#    
#sample_df.columns=col



X = range(0,253)
time = []

for j in X:
    for i in sample_df.iloc[j]:
        time.append([j,i])

data_unscaled = np.asarray(time)

#adding a comment 

#Standardization, or mean removal and variance scaling¶

data = preprocessing.scale(data_unscaled)

X = data  
#
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = [(1.0 / (n[i] - m)) * sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2)  for i in range(m)]

    const_term = 0.5 * m * np.log10(N)

    BIC = np.sum([n[i] * np.log10(n[i]) -
           n[i] * np.log10(N) -
         ((n[i] * d) / 2) * np.log10(2*np.pi) -
          (n[i] / 2) * np.log10(cl_var[i]) -
         ((n[i] - m) / 2) for i in range(m)]) - const_term

    return(BIC)



# IRIS DATA
iris = sklearn.datasets.load_iris()
X = iris.data[:, :4]  # extract only the features
#Xs = StandardScaler().fit_transform(X)
Y = iris.target

ks = range(1,10)

# run 9 times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]

# now run for each cluster the BIC computation
BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]

plt.plot(ks,BIC,'r-o')
plt.title("iris data  (cluster vs BIC)")
plt.xlabel("# clusters")
plt.ylabel("# BIC")


#k = 3
#kmeans = cluster.KMeans(n_clusters=k)
#kmeans.fit(data)
#
#labels = kmeans.labels_
#centroids = kmeans.cluster_centers_
#
#
#for i in range(k):
#    # select only data observations with cluster label == i
#    ds = data[np.where(labels==i)]
#    # plot the data observations
#    pyplot.plot(ds[:,0],ds[:,1],'o')
#    # plot the centroids
#    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
#    # make the centroid x's bigger
#    pyplot.setp(lines,ms=15.0)
#    pyplot.setp(lines,mew=2.0)
#pyplot.show()












## data generation
#
## computing K-Means with K = 2 (2 clusters)
#centroids,_ = kmeans(data,3)
## assign each sample to a cluster
#idx,_ = vq(data,centroids)
#
## some plotting using numpy's logical indexing
#plot(data[idx==0,0],data[idx==0,1],'ob',
#     data[idx==1,0],data[idx==1,1],'or')
#plot(centroids[:,0],centroids[:,1],'sg',markersize=3)
#show()       
#        
        
      
# data generation




