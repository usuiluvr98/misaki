import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create sample data
np.random.seed(42)
data = np.random.randn(100, 2)  # 100 samples, 2 features

# Perform PCA
pca = PCA(n_components=1).fit(data)
transformed_data = pca.transform(data)

# Plot the original data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='red', label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Plot the transformed data
plt.subplot(1, 2, 2)
plt.scatter(transformed_data, np.zeros_like(transformed_data), c='blue', label='Transformed Data')
plt.title('Transformed Data after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()