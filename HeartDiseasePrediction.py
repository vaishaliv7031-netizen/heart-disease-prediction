import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as  pd
import numpy as np
#Load Data
heartDiseace_cases=pd.read_csv(r"/home/sonu-nitu/Downloads/dataset.csv")
print(heartDiseace_cases)
df=heartDiseace_cases
x=df[['trestbps','chol']]
y=df['target']

# Train Model
model=LogisticRegression()
model.fit(x,y)
# Predict
y_pred=model.predict(x)
y_prob=model.predict_proba(x)[:,1]
#Metrics
print("Accuracy",accuracy_score(y,y_pred))
print("precison",precision_score(y,y_pred))
print("recall",recall_score(y,y_pred))
print("F1:",f1_score(y,y_pred))
#ConfussionMatix
cm=confusion_matrix(y,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()
#Roc_curve
fpr,tpr,_=roc_curve(y,y_prob)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'--')
plt.show()