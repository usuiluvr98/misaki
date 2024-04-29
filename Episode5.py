from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,classification_report,accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(data = np.c_[iris['data'],iris['target']],
                  columns=features+['target'])
print(df.head())

X = df.drop(['target'],axis=1)
Y = df['target']
# print("X\n",X.head())
# print("Y\n",Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

print("Accuracy = ",accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

cm = confusion_matrix(Y_test,Y_pred)
disp = ConfusionMatrixDisplay(cm,display_labels=['SETOSA','VERSICOLR','VIRGINICA'])
disp.plot()
plt.show()

plt.figure(figsize=(8, 6))
sns.regplot(x=X_train.iloc[:, 0], y=model.predict_proba(X_train)[:, 1], logistic=True, ci = False)
plt.xlabel('Sepal length')
plt.ylabel('Probability')
plt.title('Logistic Regression Curve')
plt.show()
