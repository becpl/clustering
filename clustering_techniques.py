#Apply the clustering techniques [kmeans and Agglomerative] to identify the customer segments
# and identify how many data points are differently classified in both the methods.


import pandas as pd
import pandas_profiling as pp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


df = pd.read_csv("wholesale customers data.csv")
# print(df.info())
print("Given Dataset: \n")
print(df.head(5))
print("\n\n--------------------------------------------------------------------------")


# Pandas Profiling
# profile = pp.ProfileReport(df)
# profile.to_file("wholesale_customers_data_EdA.html")


# Dropping Categorical Columns
df_imp = df.drop(["Channel","Region"],axis=1)
# print(df_imp.info())
# print(df.head(5))


# Checking Optimal K value
Inertias = []
for i in range(2,10):
    model = KMeans(n_clusters=i).fit(df_imp.values)
    Inertia = model.inertia_
    Inertias.append(Inertia)
# print(Inertias)


# Creating Elbow Curves
# plt.plot(range(2,10), Inertias)
# plt.savefig("elbow_curve.png")
# plt.savefig("KMeans_elbowcurve.png")


# Creating a new "final" model based on the optimum k value (3/5 in this case)
KMeans_model = KMeans(n_clusters=5, init='k-means++').fit(df_imp.values)
clusters = KMeans_model.predict(df_imp.values)
df_imp['Cluster'] = clusters

# Reduce to 2 dimensions using PCA to be able to visualise
pca_k = PCA(n_components=2)
pc_k = pca_k.fit_transform(df_imp)
pc_k_df = pd.DataFrame(pc_k,columns=["pc1","pc2"])
# Check if dimension reduction has worked
# print(pc_k_df.head())


# Adding K-means Cluster Info for Comparison of All Numerical Columns Later
print("K-Means Clustering Dataset: \n")
print(df_imp.head())
print('\nCount in each K-Means Cluster: \n')
print(df_imp['Cluster'].value_counts())
print("\n\n--------------------------------------------------------------------------")


# Plotting K-means Clusters
X = pc_k_df.values
# #first number is cluster number, and second number is column number
# plt.scatter(X[clusters==0,0],X[clusters==0,1],c="red", label="Cluster1")
# plt.scatter(X[clusters==1,0],X[clusters==1,1],c="blue", label="Cluster2")
# plt.scatter(X[clusters==2,0],X[clusters==2,1],c="green", label="Cluster3")
# plt.scatter(X[clusters==3,0],X[clusters==3,1],c="cyan", label="Cluster4")
# plt.scatter(X[clusters==4,0],X[clusters==4,1],c="yellow", label="Cluster5")
# plt.title('Kmeans Clustering')
# plt.savefig("Clustering_wholesale_customers_k.png")



# Heirarchical/Agglomerative clustering technique
df_h = df_imp.copy()
# print(df_h.info())
X_h = df_h.values


# Dendogram
# den = dendrogram(linkage(X_h,method="ward"))
# plt.savefig("dendrogram")


# Modeling
finalmodel = AgglomerativeClustering(n_clusters=5).fit(X_h)
clusters1 = finalmodel.fit_predict(X_h)
df_h['Cluster'] = clusters1
print("Agglomerative Clustering Dataset: \n")
print(df_h.head())
# Check if the process of applying clustering worked
print("\n\n--------------------------------------------------------------------------")
# print(clusters)
print('\nCount in each Agglomerative Cluster: \n')
print(df_h['Cluster'].value_counts())
print("\n\n--------------------------------------------------------------------------")


#first number is cluster number, and second number is column number
# plt.scatter(X[clusters1==0,0],X[clusters1==0,1],c="red", label="Cluster1")
# plt.scatter(X[clusters1==1,0],X[clusters1==1,1],c="blue", label="Cluster2")
# plt.scatter(X[clusters1==2,0],X[clusters1==2,1],c="green", label="Cluster3")
# plt.scatter(X[clusters1==3,0],X[clusters1==3,1],c="cyan", label="Cluster4")
# plt.scatter(X[clusters1==4,0],X[clusters1==4,1],c="yellow", label="Cluster5")
# plt.title('Agglomerative Clustering')
# plt.savefig("Clustering_wholesale_customers_a.png")


# Sorting Before Comparison
df_imp = df_imp.sort_values('Fresh')
df_h = df_h.sort_values('Fresh')
print("K-Means Clustering Dataset after sorting: \n")
print(df_imp.head(5))
print("\n\n--------------------------------------------------------------------------")
print("Agglomerative Clustering Dataset after sorting: \n")
print(df_h.head(5))
df_imp['Comparison'] = df_imp['Cluster'] == df_h[ 'Cluster']
print("\n\n--------------------------------------------------------------------------")


# True/False Comparison Count
print("\nCluster Matching Statistics: \n")
print(df_imp['Comparison'].value_counts())

# print(df_imp1.head())
# print(df_imp1.shape)
# print(df_h.shape)

