from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)
print(X_pca.shape)