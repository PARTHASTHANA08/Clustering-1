import pandas as pd 
import csv 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Stars1.csv")
X = df.iloc[:[0,1]].values
print("The Values in X are :",X)
wcss = []
for i in range(1,11):
    kMeans = KMeans(n_clusters = i,init = "k-means++",random_state= 22)
    kMeans.fit(X)
    wcss.append(kMeans.inertia_)
plt.figure(figsize = (10,5))
sns.lineplot(range(1,11),wcss,marker = "o", color = "red")
plt.title("The Elbow Method")
plt.xlabel("Mass of Stars")
plt.ylabel("WCSS")
plt.show()    
kMeans = KMeans(n_clusters = 3 , init = "k-means++", random_state=20)
y_kMeans = kMeans.fit_predict(X)
plt.figure(figsize = (20,10))
sns.scatterplot(X[y_kMeans == 0,0], X[y_kMeans == 0,1] , color = "red" , label = "Cluster 2")
sns.scatterplot(X[y_kMeans == 1,0], X[y_kMeans == 1,1] , color = "yellow" , label = "Cluster 3")
sns.scatterplot(X[y_kMeans == 2,0], X[y_kMeans == 2,1] , color = "green" , label = "Cluster 1")
sns.scatterplot(kMeans.cluster_centers_[:,0], kMeans.cluster_centers_[:,1] , color = "blue" , label = "Centeroids" , s = 100, marker = ",")
plt.grid(True)
plt.title("Clusters of Stars")
plt.xlabel("Mass of the Stars")
plt.ylabel("Radius of the Stars")
plt.show()
