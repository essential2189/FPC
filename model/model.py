from tqdm import tqdm
import torch
from torch import tensor
import timm
from run import loss_, optim_

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

optimizer = optim_()
criterion = loss_()

class train_feature_extractor(torch.nn.Module):
    def __init__(self):
        super(train_feature_extractor, self).__init__()

        self.model = timm.create_model('resnet50', pretrained=True, num_classes=500)

        self.train_list = []
        self.test_list = []

        self.correct = 0
        self.total = 0
        self.result_list = []
        self.label_list = []
        self.anomaly_list = []
        self.normal_list = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def __call__(self, x: tensor):
        feature_maps = self.model(x.to(self.device))
        return feature_maps

    def train(self, train_dl):
        for epoch in range(15):
            i = 0
            running_loss = 0.0
            for inputs, labels in tqdm(train_dl):
                optimizer.zero_grad()
                labels = labels.to(self.device)

                outputs = self(inputs)

                loss = criterion(outputs, labels.to(torch.float32))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                i += 1

        print('Finished Training')


def test_feature_extractor(net, test_dl):
    result_list = []
    label_list = []
    anomaly_list = []
    normal_list = []
    with torch.no_grad():
        for sample, label in tqdm(test_dl):
            outputs = net(sample)

            label_list.append(label[0])
            result_list.append(outputs[0].tolist())
            if label[0] == 0:
                normal_list.append(outputs[0].tolist())
            elif label[0] == 1:
                anomaly_list.append(outputs[0].tolist())

    return result_list, label_list, normal_list, anomaly_list




