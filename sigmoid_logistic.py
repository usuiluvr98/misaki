import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt


X = np.array([3.78,2.44,2.09,0.14,1.72,1.65,4.92,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)

y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])

logreg = linear_model.LogisticRegression()
logreg.fit(X,y)


def logit2prob(logreg, X):
    log_odds = logreg.coef_ * X + logreg.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return probability

print(logit2prob(logreg,X))

plt.scatter(X, y, color='blue', label='Data Points')

# Plotting the logistic regression curve
X_values = np.linspace(0, 12, 100).reshape(-1, 1)
plt.plot(X_values, logreg.predict_proba(X_values)[:, 1], color='red', label='Logistic Regression Curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression')
plt.legend()

# Displaying the plot
plt.show()