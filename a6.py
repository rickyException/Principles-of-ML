import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/Tourism/train.csv")
df.head()

nulls = df.isna().sum()
print(nulls)

df[df.isna().any(axis=1)]
print(df.shape)

df_new = df.drop(axis = 1,columns = ["subject","Activity"])
df_new.head()
print(df_new.shape)

pip install kneed

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,homogeneity_score,completeness_score
import seaborn as sns
from kneed import KneeLocator

kmeans_kwargs = {
   "init": "random",
   "n_init": 10,
   "max_iter": 300,
   "random_state": 42,
}
wcss = []
silhouette = []
for k in range(1,20):
  kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
  kmeans.fit(df_new)
  wcss.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1,20), wcss)
plt.xticks(range(1,20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(1, 20), wcss, curve="convex", direction="decreasing")

k_new=kl.elbow
print(k_new)

model = KMeans(n_clusters=k_new)
y_pred = model.fit_predict(df_new)
print(y_pred)
df_new['train_predict'] = y_pred

wcss = model.inertia_
print("Within-Cluster Sum of Squares (WCSS):", wcss)
klabels = model.labels_
silhouette=silhouette_score(df_new,klabels)
print("Silhouette Score ",silhouette)

df1 = pd.read_csv('/content/drive/MyDrive/Tourism/test.csv')
nulls = df1.isna().sum()
print(nulls)
print(df1.shape)
test_df = df1.drop(axis = 1,columns = ["subject","Activity"])
print(test_df.shape)

model_test = KMeans(n_clusters=k_new)
y_pred_test = model_test.fit_predict(test_df)
print(y_pred_test)
test_df['test_predict'] = y_pred_test

wcss_test = model_test.inertia_
print("Within-Cluster Sum of Squares (WCSS):", wcss_test)
klabels = model_test.labels_
silhouette=silhouette_score(test_df,klabels)
print("Silhouette Score ",silhouette)

