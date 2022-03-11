import torch.optim as optim
import click
import torch
from data import LoadDataset
import torch.nn as nn
import random
import numpy as np
import os

from cluster import knn_cluster
from model import test_feature_extractor
from model import train_feature_extractor

ALLOWED_METHODS = ["FPC"]

model = train_feature_extractor()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    seed_everything(42)


def loss_():
    criterion = nn.MSELoss()
    return criterion


def optim_(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer


def run_model(method: str, data_path: str):
    train_ds, test_ds = LoadDataset(data_path, size=244).get_dataloaders()

    model.train(train_ds)

    MODEL_PATH = '/content/drive/MyDrive/resnet50_2.pth'

    # torch.save(model.state_dict(), MODEL_PATH)

    pre_net = train_feature_extractor()
    pre_net.load_state_dict(torch.load(MODEL_PATH))

    result_list, label_list, normal_list, anomaly_list = test_feature_extractor(pre_net, test_ds)

    rocauc_score = knn_cluster(result_list, label_list, normal_list, anomaly_list)

    print(rocauc_score)


@click.command()
@click.argument("method")
@click.option("--dataset", default="all", help="Dataset, defaults to all datasets.")
def cli_interface(method: str, dataset: str):
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        dataset = [dataset]

    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."

    run_model(method, dataset)

if __name__ == "__main__":
    cli_interface()