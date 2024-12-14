import numpy as np
import torch
from torch import nn


class Configs:

    def __init__(self):
        self.model_name = 'Faster'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.exp_name = 'KDAT training'
        self.dataset = 'coco'
        self.superstore_weights = ''
        self.verbos = False
        self.base_dir = 'Demo'

        self.data_parameters = {
            'clean_anno_file_path': 'Demo/clean.csv',
            'dirty_anno_file_path': 'Demo/dirty.csv',
            'masked_anno_file_path': 'Demo/masked.csv',
            'val_dirty_anno_file_path': 'Demo/val_dirty.csv',
            'val_clean_anno_file_path': 'Demo/val_clean.csv',
            'val_image_dir': 'Demo/Validation',
        }

        self.parameters = {'Faster_coco_training': {
            'Number_of_epochs': 25,
            'optimizer_type': 'SGD',
            'optimizer_lr': 0.0001,
            'backbone_lr': 0.00001,
            'optimizer_momentum': 0.9,
            'optimizer_weight_decay': 0.0005,
            'Number_of_dirty_images': 5,
            'lr_drop': 5,
            'batch_size': 1,
        },
            'Faster_SuperStore_training': {
                'Number_of_epochs': 25,
                'optimizer_type': 'SGD',
                'optimizer_lr': 0.00006,
                'backbone_lr': 0.0000015,
                'optimizer_momentum': 0.9,
                'optimizer_weight_decay': 0.0005,
                'Number_of_dirty_images': 5,
                'lr_drop': 5,
                'batch_size': 1,
            },
            'Faster_inria_training': {
                'Number_of_epochs': 25,
                'optimizer_type': 'AdamW',
                'optimizer_lr': 0.000075,
                'backbone_lr': 0.0000095,
                'optimizer_momentum': 0.9,
                'optimizer_weight_decay': 0.0005,
                'Number_of_dirty_images': 5,
                'lr_drop': 5,
                'batch_size': 1,
            },
            'DETR_coco_training': {
                'Number_of_epochs': 25,
                'optimizer_type': 'AdamW',
                'optimizer_lr': 0.00001,
                'backbone_lr': 0.000001,
                'optimizer_momentum': 0.9,
                'optimizer_weight_decay': 0.0001,
                'Number_of_dirty_images': 5,
                'lr_drop': 5,
                'batch_size': 1,
            },
            'Faster_coco_loss': {
                'loss_weights': np.array([0.1, 0.6, 0.6, 0.6]),  # w_od, w_back, w_cls, w_adjustable
                'w_clean': 0.75,
                'w_adv': 0.25,
                'freeze_backbone': False,
                'temperature_cls': 100,
                'feature_loss_function': MSECSLoss(0.5, 0.5),
                'adjustable_loss_function': nn.L1Loss(),
                'cls_loss_function': nn.KLDivLoss(),
                'use_od_loss': True,
                'use_feature_loss': True,
                'use_cls_loss': True,
                'use_adjustable_loss': True,
                'no_object_index': 0
            },
            'Faster_SuperStore_loss': {
                'loss_weights': np.array([0.07965, 0.17799, 0.72672, 0.58513]),  # w_od, w_back, w_cls, w_adjustable
                'w_clean': 0.81,
                'w_adv': 0.19,
                'freeze_backbone': False,
                'temperature_cls': 14,
                'feature_loss_function': MSECSLoss(0.5, 0.5),
                'adjustable_loss_function': MSECSLoss(0, 1),
                'cls_loss_function': nn.KLDivLoss(),
                'use_od_loss': True,
                'use_feature_loss': True,
                'use_cls_loss': True,
                'use_adjustable_loss': True,
                'no_object_index': 0
            },
            'Faster_inria_loss': {
                'loss_weights': np.array([0.09934, 0.7547, 0.88125, 0.90782]),  # w_od, w_back, w_cls, w_adjustable
                'w_clean': 0.86,
                'w_adv': 0.14,
                'freeze_backbone': False,
                'temperature_cls': 97,
                'feature_loss_function': nn.L1Loss(),
                'adjustable_loss_function': nn.L1Loss(),
                'cls_loss_function': MSECSLoss(0.5, 0.5),
                'use_od_loss': True,
                'use_feature_loss': True,
                'use_cls_loss': True,
                'use_adjustable_loss': True,
                'no_object_index': 0
            },
            'DETR_coco_loss': {
                'loss_weights': np.array([0.1, 0.6, 0.6, 0.6]),  # w_od, w_back, w_cls, w_adjustable
                'w_clean': 0.75,
                'w_adv': 0.25,
                'freeze_backbone': False,
                'temperature_cls': 50,
                'feature_loss_function': MSECSLoss(0.5, 0.5),
                'adjustable_loss_function': MSECSLoss(0, 1),
                'cls_loss_function': nn.KLDivLoss(),
                'use_od_loss': True,
                'use_feature_loss': True,
                'use_cls_loss': True,
                'use_adjustable_loss': True,
                'no_object_index': 91
            },
        }

        self.loss_parameters = self.parameters[f'{self.model_name}_{self.dataset}_loss']
        self.training_parameters = self.parameters[f'{self.model_name}_{self.dataset}_training']

        self.detr_args = DetrArgs()

    def __setitem__(self, key, value, conf_type='loss'):
        if conf_type == 'loss':
            self.loss_parameters[key] = value
        if conf_type == 'training':
            self.training_parameters[key] = value


class DetrArgs:
    def __init__(self):
        self.weights = ''
        self.lr = 0.0001
        self.lr_backbone = 1e-05
        self.batch_size = 1
        self.weight_decay = 0.0001
        self.epochs = 300
        self.lr_drop = 200
        self.clip_max_norm = 0.1
        self.frozen_weights = None
        self.backbone = 'resnet50'
        self.dilation = False
        self.position_embedding = 'sine'
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.num_queries = 100
        self.pre_norm = False
        self.masks = False
        self.aux_loss = True
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1
        self.dataset_file = 'coco'
        self.coco_path = None
        self.coco_panoptic_path = None
        self.remove_difficult = False
        self.output_dir = ''
        self.device = 'cuda'
        self.seed = 42
        self.resume = ''
        self.start_epoch = 0
        self.eval = False
        self.num_workers = 1
        self.world_size = 1
        self.dist_url = 'env://'
        self.distributed = False


class MSECSLoss(nn.Module):
    def __init__(self, w_mse, w_cs):
        super(MSECSLoss, self).__init__()
        self.w_mse = w_mse
        self.w_cs = w_cs
        self.mse_loss = nn.MSELoss()
        self.cs_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, pred, gt):
        return self.w_mse * self.mse_loss(pred, gt) + self.w_cs * (1 - self.cs_loss(pred, gt).mean())
