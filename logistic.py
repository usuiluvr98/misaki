from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,classification_report,accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data = np.c_[iris['data'],iris['target']],
                  columns=iris['feature_names']+['target'])
# print(df.head())

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

