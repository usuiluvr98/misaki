from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=32)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(cm,display_labels=['has_cancer','not_cancer'])
disp.plot()
plt.show()