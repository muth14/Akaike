import pandas as pd # pandas is used for reading the 
import numpy as np # numerical file handling
import matplotlib.pyplot as plt # visuvalization to understand the pattetrns
import seaborn as sns
from sklearn.model_selection import train_test_split #to split the train anad test data
from sklearn.preprocessing import StandardScaler # higher value is tuff for comparing it is used for changing to simple range values
from sklearn.linear_model import LogisticRegression # these model is classification and ensemble model has beeen used in this accuracy which is best that can taken and solved
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
# confusion matrix is used for checking correct and not correct records,auc-roc is basically used for binary classificaiton, 
#model paramters that haseen used to this model are penalty,maximum no of steps like  maxmimum no of iteration setting manual parameters  c--> how the model should be 
# random forest training phse --> selecting the randomsamples traiened on that data,feature selection training again
df = pd.read_csv(r"C:\Users\DELL\Downloads\heart_cleveland_upload.csv")
print(df.shape)
print(df.head())
print(df.describe())
print(df["condition"].value_counts())

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")#blue negative correlation ,red positive correlation
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
#correlation  how strongly two variables are correlated
# for example if age inc bp inc like it is highly correlated
#model building step condition mean --> heart diesase is there or not prediction
X = df.drop("condition", axis=1)
y = df["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

print("Logistic Regression")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

rf = RandomForestClassifier(random_state=42) # it is used to fit the random ness
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

print("Random Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression AUC=" + str(round(auc_lr,3)))
plt.plot(fpr_rf, tpr_rf, label="Random Forest AUC=" + str(round(auc_rf,3)))
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.title("ROC Curve")
plt.savefig("roccurve.png")
plt.show()

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values()

plt.figure()
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
