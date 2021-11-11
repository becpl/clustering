#Apply the clustering techniques [kmeans and Agglomerative] to identify the customer segments
# and identify how many data points are differently classified in both the methods.

#import the data into a dataframe and have a quick look
import pandas as pd
df = pd.read_csv("wholesale customers data.csv")
print(df.info())
df_imp = df.drop(["Channel","Region"],axis=1)
print(df_imp.info())


#Kmeans clustering technique
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

Inertias = []
#for loop to automate testing of different k values
for i in range(2,10):
    model = KMeans(n_clusters=i).fit(df_imp.values)
    Inertia = model.inertia_
    Inertias.append(Inertia)
print("\n\n--------------------------------------------------------------------------")
print(Inertias)
from matplotlib import pyplot as plt
#plt.plot(range(2,10),Inertias)
#plt.savefig("KMeans_elbowcurve.png")
#create a new "final" model based on the optimum k values (5 in this case)
KMeans_model = KMeans(n_clusters=5).fit(df_imp.values)
clusters = KMeans_model.predict(df_imp.values)

#reduce to 2 dimensions using PCA to be able to visualise
pca_k = PCA(n_components=2)
pc_k = pca_k.fit_transform(df_imp)
pc_k_df = pd.DataFrame(pc_k,columns=["pc1","pc2"])
#check dimension reduction has worked
print("\n\n--------------------------------------------------------------------------")
print(pc_k_df.head())

X = pc_k_df.values
#first number is cluster number, and second number is column number
plt.scatter(X[clusters==0,0],X[clusters==0,1],c="red", label="Cluster1")
plt.scatter(X[clusters==1,0],X[clusters==1,1],c="blue", label="Cluster2")
plt.scatter(X[clusters==2,0],X[clusters==2,1],c="green", label="Cluster3")
plt.scatter(X[clusters==3,0],X[clusters==3,1],c="cyan", label="Cluster4")
plt.scatter(X[clusters==4,0],X[clusters==4,1],c="yellow", label="Cluster5")

plt.savefig("Clustering_wholesale_customers_k.png")

#heirarchical/agglomerative clustering technique
from matplotlib import pyplot as plt
df_h = df_imp
print("\n\n--------------------------------------------------------------------------")
print(df_h.info())
X_h = df_h.values
from scipy.cluster.hierarchy import dendrogram, linkage
#den = dendrogram(linkage(X_h,method="ward"))
#plt.savefig("dendrogram")
from sklearn.cluster import AgglomerativeClustering
finalmodel = AgglomerativeClustering(n_clusters=5).fit(X_h)
clusters = finalmodel.fit_predict(X_h)
#check process of applying clustering worked
print("\n\n--------------------------------------------------------------------------")
print(clusters)

#first number is cluster number, and second number is column number
plt.scatter(X[clusters==0,0],X[clusters==0,1],c="red", label="Cluster1")
plt.scatter(X[clusters==1,0],X[clusters==1,1],c="blue", label="Cluster2")
plt.scatter(X[clusters==2,0],X[clusters==2,1],c="green", label="Cluster3")
plt.scatter(X[clusters==3,0],X[clusters==3,1],c="cyan", label="Cluster4")
plt.scatter(X[clusters==4,0],X[clusters==4,1],c="black", label="Cluster5")

plt.savefig("Clustering_wholesale_customers_a.png")
