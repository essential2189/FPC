import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


scaler1 = StandardScaler()
pca1 = PCA(n_components=500)


train_scaler = scaler1.fit_transform(result_list)
train_reduce = pca1.fit_transform(train_scaler)