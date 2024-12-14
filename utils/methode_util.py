import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from torch import nn


def create_experiment_dir(base_dir, exp_name, added=None, with_timestamp=True):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_") + exp_name
    if with_timestamp:
        exp_dir = os.path.join(base_dir, timestamp)
    else:
        exp_dir = os.path.join(base_dir, exp_name + ' ' + str(added))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        return exp_dir
    return -1


def save_config_file(exp_dir, config):
    output_path = exp_dir + '/Train config.csv'
    dict = {}
    dict.update(config.training_parameters)
    dict.update(config.loss_parameters)
    df = pd.DataFrame(list(dict.items()), columns=['Attribute', 'Value'])
    df.to_csv(output_path)


def compute_iou(box1, box2):
    """Compute the Intersection Over Union (IOU) of two bounding boxes."""
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    inter_area = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)
    box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def plot_losses(output_path, list_loss_history, names_lost_history, graph_name):
    fig = plt.figure(figsize=(10, 8))
    for hist, name in zip(list_loss_history, names_lost_history):
        plt.plot(hist, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    # plt.grid()
    plt.legend()
    plt.title('Loss')
    # plt.show()
    plt.savefig(f'{output_path}/{graph_name}.png', dpi=fig.dpi)


class MSECSLoss(nn.Module):
    def __init__(self, w_mse, w_cs):
        super(MSECSLoss, self).__init__()
        self.w_mse = w_mse
        self.w_cs = w_cs
        self.mse_loss = nn.MSELoss()
        self.cs_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, pred, gt):
        return self.w_mse * self.mse_loss(pred, gt) + self.w_cs * (1 - self.cs_loss(pred, gt).mean())
