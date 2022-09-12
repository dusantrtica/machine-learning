from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

olivetti_data = fetch_olivetti_faces()
# there are 400 images (40 people - 1 person has 10 images). 1 image = 64x64px
features = olivetti_data.data
targets = olivetti_data.target

fig, sub_plots = plt.subplots(nrows=5, ncols=5, figsize=(14, 8))
sub_plots = sub_plots.flatten()

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, stratify=targets)
# lets try to find optimal number of eigenvectors (principle components)

# 4096 - to reduce to 100 features instead of 4096
pca = PCA(n_components=100, whiten=True)