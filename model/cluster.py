from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knn_cluster(result_list, label_list, normal_list, anomaly_list):
    # result_np = np.array(result_list)
    # label_np = np.array(label_list)

    # anomaly_np = result_np[label_np == 1]
    # normal_np = result_np[label_np == 0]
    #
    # label_anomaly_np = label_np[label_np == 1]
    # label_normal_np = label_np[label_np == 0]

    scaler1 = StandardScaler()
    pca1 = PCA(n_components=500)

    train_scaler = scaler1.fit_transform(result_list)
    train_reduce = pca1.fit_transform(train_scaler)

    knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)

    knn.fit(train_reduce, label_list)
    # y_ano = knn.predict(anomaly_reduce).tolist()
    # y_no = knn.predict(normal_reduce).tolist()
    y_all = knn.predict(train_reduce).tolist()

    return roc_auc_score(label_list, y_all)