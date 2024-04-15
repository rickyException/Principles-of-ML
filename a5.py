import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_curve,auc

online = pd.read_csv("online_shoppers_intention.csv")
online.head()

result = online.isna().sum()
print(result)

counts = online["Revenue"].value_counts()
print(counts)

x = online.drop("Revenue",axis = 1)
x.shape
print(x.columns)

# Apply one-hot encoding to specific columns
x_encoded = pd.get_dummies(x, columns=['Month','VisitorType','Weekend'])

# Print column names
print(x_encoded.columns)
x_encoded

y = online["Revenue"]
#print(y)

x_train,x_test,y_train,y_test = train_test_split(x_encoded,y,test_size= 0.25)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def applyKNN(x_train,y_train,x_test,metric_p):
  knn = KNeighborsClassifier(n_neighbors = 5,metric = metric_p)
  knn.fit(x_train,y_train)
  y_pred = knn.predict(x_test)
  return y_pred

def apply_metrics(y_test,y_pred,metric_name):
  print("-----"+metric_name+"-----")
  print("Predicted y_pred:- \n")
  print(y_pred)
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  print("Classification report:- \n",classification_report(y_test,y_pred))
  print("Confusion matrix :- \n",confusion_matrix(y_pred,y_test))

def plotROC(y_test,y_pred):
  fpr, tpr, threshold = roc_curve(y_test, y_pred)
  roc_auc = auc(fpr, tpr)
  print(roc_auc)

  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.title('ROC Curve of kNN')
  plt.show()

vals = applyKNN(x_train,y_train,x_test,'manhattan')
apply_metrics(y_test,vals,'manhattan')
plotROC(y_test,vals)

vals = applyKNN(x_train,y_train,x_test,'euclidean')
apply_metrics(y_test,vals,'euclidean')
plotROC(y_test,vals)

vals=applyKNN(x_train,y_train,x_test,'minkowski')
apply_metrics(y_test,vals,'minkowski')
plotROC(y_test,vals)
