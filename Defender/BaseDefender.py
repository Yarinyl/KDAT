import random
from abc import ABC, abstractmethod

from torch import nn
import pandas as pd
import torch
from tqdm import tqdm

from ObjectDetection.DETR import DETR
from utils.custom_dataloaders import TrioCleanSetOfAdvAnno, custom_collate, StandardDat
from utils.methode_util import plot_losses, create_experiment_dir, save_config_file, compute_iou
from torch.utils.data import DataLoader


class BaseDefender(ABC, nn.Module):
    def __init__(self, device, config, exp_dir=None):
        super(BaseDefender, self).__init__()
        self.device = device
        self.config = config
        self._detr_args = None
        self._yolo_args = None
        if exp_dir:
            self._exp_dir = exp_dir
        else:
            self._exp_dir = create_experiment_dir(config.base_dir, config.exp_name)
        if self.config.model_name == 'DETR':
            self._detr_args = config.detr_args
        self.temperature_cls = config.loss_parameters['temperature_cls']
        self.teacher_model, self.student_model = self._load_base_models()
        self.start_epoch = 0
        self._loss_parameters = config.loss_parameters
        self._optimizer = self._init_optimizer()
        if isinstance(self.student_model, DETR):
            self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, self.config.training_parameters['lr_drop'])
        else:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self._optimizer, T_0=3, T_mult=2)
        self._loss_weights = torch.tensor(self.config.loss_parameters['loss_weights']).to(device)
        self._backbone_loss = self.config.loss_parameters['feature_loss_function']
        self._cls_loss_function = self.config.loss_parameters['cls_loss_function']
        self.verbos = self.config.verbos

    @abstractmethod
    def _load_base_models(self):
        pass

    @abstractmethod
    def _freeze_backbone(self):
        pass

    def _get_feature_loss(self, features_student_dirty, guiding_features_dirty, features_student_clean,
                          guiding_features_clean):
        back_loss_dirty = sum(
            [self._backbone_loss(fsd, gfd) for fsd, gfd in zip(features_student_dirty, guiding_features_dirty)])
        torch.cuda.empty_cache()
        back_loss_dirty = back_loss_dirty * 1 / len(guiding_features_dirty)
        back_loss_clean = sum(
            [self._backbone_loss(fsc, gfc) for fsc, gfc in zip(features_student_clean, guiding_features_clean)])
        return back_loss_dirty, back_loss_clean

    def _get_cls_loss_old(self, student_poa, teacher_poa, threshold=0.75):
        proposals_student, pred_vector_student = student_poa
        proposals_teacher, pred_vector_teacher = teacher_poa
        if isinstance(self._cls_loss_function, nn.KLDivLoss):
            pred_vector_student = torch.log(pred_vector_student)
        distance_score = torch.tensor(0.0).to(self.device)

        if len(proposals_teacher) > 0 and len(proposals_student) > 0:
            iou_matrix = torch.zeros((len(proposals_student), len(proposals_teacher)))

            for i, p_s in enumerate(proposals_student):
                iou_matrix[i, :] = torch.tensor([compute_iou(p_s, p_t) for p_t in proposals_teacher])

            max_iou_values, max_iou_indices = iou_matrix.max(dim=1)

            above_threshold_mask = max_iou_values > threshold
            below_threshold_mask = ~above_threshold_mask

            # For above threshold cases
            index_of_pred = torch.nonzero(above_threshold_mask).squeeze()
            if index_of_pred.numel() > 0:
                distance_score += self._cls_loss_function(pred_vector_student[index_of_pred],
                                                          pred_vector_teacher[max_iou_indices[above_threshold_mask]])

            # For below threshold cases
            if torch.sum(below_threshold_mask) > 0:
                tensor = torch.zeros(pred_vector_student.shape[1]).to(self.device)
                tensor[self._loss_parameters['no_object_index']] = 1
                tensor = tensor.expand(torch.sum(below_threshold_mask), -1)
                distance_score += self._cls_loss_function(pred_vector_student[below_threshold_mask], tensor)

        return distance_score

    def _get_cls_loss(self, student_poa, teacher_poa, threshold=0.75):
        proposals_student, pred_vector_student = student_poa
        proposals_teacher, pred_vector_teacher = teacher_poa
        if isinstance(self._cls_loss_function, nn.KLDivLoss):
            pred_vector_student = torch.log(pred_vector_student)
        distance_score = torch.tensor(0.0).to(self.device)

        if len(proposals_teacher) > 0 and len(proposals_student) > 0:
            proposals_student_tensor = torch.tensor(proposals_student, device=self.device)
            proposals_teacher_tensor = torch.tensor(proposals_teacher, device=self.device)

            # Use the vectorized IoU computation
            iou_matrix = self.compute_iou_matrix(proposals_student_tensor, proposals_teacher_tensor)

            max_iou_values, max_iou_indices = iou_matrix.max(dim=1)

            above_threshold_mask = max_iou_values > threshold
            below_threshold_mask = ~above_threshold_mask

            # For above threshold cases
            if above_threshold_mask.any():
                distance_score += self._cls_loss_function(
                    pred_vector_student[above_threshold_mask],
                    pred_vector_teacher[max_iou_indices[above_threshold_mask]]
                )

            # For below threshold cases
            if below_threshold_mask.any():
                no_object_tensor = torch.zeros_like(pred_vector_student[0]).to(self.device)
                no_object_tensor[self._loss_parameters['no_object_index']] = 1
                no_object_tensors = no_object_tensor.unsqueeze(0).expand(below_threshold_mask.sum(), -1)
                distance_score += self._cls_loss_function(pred_vector_student[below_threshold_mask], no_object_tensors)

        return distance_score

    @staticmethod
    def compute_iou_matrix(boxes1, boxes2):
        """Compute the Intersection Over Union (IOU) of two sets of bounding boxes."""
        boxes1 = boxes1.unsqueeze(1)  # Shape (N, 1, 4)
        boxes2 = boxes2.unsqueeze(0)  # Shape (1, M, 4)

        inter_xmin = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_ymin = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_xmax = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_ymax = torch.min(boxes1[..., 3], boxes2[..., 3])

        inter_area = (inter_xmax - inter_xmin + 1).clamp(min=0) * (inter_ymax - inter_ymin + 1).clamp(min=0)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0] + 1) * (boxes1[..., 3] - boxes1[..., 1] + 1)
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0] + 1) * (boxes2[..., 3] - boxes2[..., 1] + 1)

        iou = inter_area / (boxes1_area + boxes2_area - inter_area)
        return iou

    @abstractmethod
    def _get_adjustable_loss(self, student_values, teacher_values):
        pass

    @abstractmethod
    def _get_losses(self, clean_imgs, dirty_imgs, masked_imgs, targets):
        pass

    def compute_loss(self, clean_imgs, dirty_imgs, masked_imgs, targets):
        losses = self._get_losses(clean_imgs, dirty_imgs, masked_imgs, targets)

        torch.cuda.empty_cache()

        clean_losses, dirty_losses = self._normalize_losses(losses)

        torch.cuda.empty_cache()

        loss_clean = torch.sum(self._loss_weights * clean_losses)
        loss_dirty = torch.sum(self._loss_weights * dirty_losses)

        loss = self._loss_parameters['w_clean'] * loss_clean + self._loss_parameters['w_adv'] * loss_dirty

        return loss

    @staticmethod
    def _normalize_losses(losses):
        losses = [loss / (loss + 0.0001) * max(losses) for loss in losses]
        return torch.stack(losses[:4]), torch.stack(losses[4:])

    def _init_optimizer(self):
        param_dicts = [
            {"params": [p for n, p in self.student_model.model.named_parameters() if
                        "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.student_model.model.named_parameters() if
                           "backbone" in n and p.requires_grad],
                "lr": self.config.training_parameters['backbone_lr'],
            },
        ]

        if self.config.training_parameters['optimizer_type'] == 'SGD':
            optimizer = torch.optim.SGD(param_dicts,
                                        lr=self.config.training_parameters['optimizer_lr'],
                                        momentum=self.config.training_parameters['optimizer_momentum'],
                                        weight_decay=self.config.training_parameters['optimizer_weight_decay'])
        else:
            optimizer = torch.optim.AdamW(param_dicts,
                                          lr=self.config.training_parameters['optimizer_lr'],
                                          weight_decay=self.config.training_parameters['optimizer_weight_decay'])
        return optimizer

    def _create_train_data_loader(self):
        clean_anno_file = pd.read_csv(self.config.data_parameters['clean_anno_file_path'])
        dirty_anno_file = pd.read_csv(self.config.data_parameters['dirty_anno_file_path'])
        masked_anno_file = pd.read_csv(self.config.data_parameters['masked_anno_file_path'])

        unique_images = clean_anno_file.image_id.unique()
        train_dl = DataLoader(
            TrioCleanSetOfAdvAnno(self.student_model, clean_anno_file, dirty_anno_file,
                                  masked_anno_file, unique_images,
                                  list(range(len(unique_images))),
                                  number_of_dirty_images=self.config.training_parameters[
                                      'Number_of_dirty_images']), batch_size=1, shuffle=not self.verbos,
            collate_fn=custom_collate, pin_memory=True)
        return train_dl

    def _create_val_data_loader(self):
        images_anno_file = pd.read_csv(self.config.data_parameters['val_dirty_anno_file_path'])
        unique_images = images_anno_file.my_id.unique()
        val_dirty_dl = DataLoader(StandardDat(self.student_model, images_anno_file, unique_images,
                                              list(range(len(unique_images)))),
                                  batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, pin_memory=True)
        images_anno_file = pd.read_csv(self.config.data_parameters['val_clean_anno_file_path'])
        unique_images = images_anno_file.my_id.unique()
        val_clean_dl = DataLoader(StandardDat(self.student_model, images_anno_file, unique_images,
                                              list(range(len(unique_images)))),
                                  batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, pin_memory=True)
        return val_dirty_dl, val_clean_dl

    def _create_data_loaders(self):
        return self._create_train_data_loader(), self._create_val_data_loader()

    def trainer(self):
        train_dl, (val_dirty_dl, val_clean_dl) = self._create_data_loaders()
        early_stopping_counter = 0
        loss_history = []
        val_clean_loss_history = []
        val_dirty_loss_history = []
        save_config_file(self._exp_dir, self.config)
        if self.verbos:
            print('Start Training')
        for epoch in tqdm(range(self.start_epoch, self.config.training_parameters['Number_of_epochs']), desc='Epoch: ',
                          disable=not self.verbos):
            try:
                dirty_lose_on_val = self._loss_on_val(val_dirty_dl)
                clean_lose_on_val = self._loss_on_val(val_clean_dl)
                val_clean_loss_history.append(clean_lose_on_val)
                val_dirty_loss_history.append(dirty_lose_on_val)
                if epoch == 0:
                    best_val_loss = dirty_lose_on_val
                elif dirty_lose_on_val * 1.02 >= best_val_loss:
                    early_stopping_counter += 1
                    if early_stopping_counter > 6:
                        break
                else:
                    best_val_loss = dirty_lose_on_val
                    early_stopping_counter = 0
            except:
                pass
            epoch_loss = self._train_one_epoch(train_dl)
            loss_history.append(epoch_loss)
            save_path = self._exp_dir + '/epoch_' + str(epoch) + '.pt'
            torch.save(self.student_model.model.state_dict(), save_path)

            self._scheduler.step()

        plot_losses(self._exp_dir, [loss_history], ['Train Loss'], 'Train Loss')
        plot_losses(self._exp_dir, [val_clean_loss_history, val_dirty_loss_history], ['Val Clean', 'Val Dirty'],
                    'Validation Loss')

    def _train_one_epoch(self, train_dl):
        epoch_loss = 0
        batch_loss = 0
        j = 0
        for i, data in enumerate(tqdm(train_dl, desc=' training image: ', disable=not self.verbos)):
            clean_imgs = []
            dirty_imgs = []
            masked_imgs = []
            targets = []
            # try:
            for d in data:
                clean_imgs.append(d[0].to(self.device))
                dirty_imgs = [img.to(self.device) for img in d[1]]
                masked_imgs = [img.to(self.device) for img in d[2]]
                if isinstance(d[3], dict):
                    targ = {'boxes': d[3]['boxes'].to(self.device), 'labels': d[3]['labels'].to(self.device)}
                else:
                    targ = d[3].to(self.device)
                targets.append(targ)

            loss = self.compute_loss(clean_imgs, dirty_imgs, masked_imgs, targets)
            # print(loss.cpu().detach().numpy())
            torch.cuda.empty_cache()
            j += 1
            batch_loss += loss
            epoch_loss += loss.cpu().detach().numpy()

            if j == self.config.training_parameters['batch_size']:
                self._optimizer.zero_grad()
                batch_loss = batch_loss / j
                batch_loss.backward()
                if isinstance(self.student_model, DETR) and self.student_model.detr_args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(),
                                                   self.student_model.detr_args.clip_max_norm)
                self._optimizer.step()
                j = 0
                batch_loss = 0
            del loss, clean_imgs, masked_imgs, dirty_imgs, targets, targ
            # except Exception as e:
            #     print(f"An unexpected error occurred: {e}")
            #     print(f'Failed at iter {i}')
            #     continue
        if j > 0:
            self._optimizer.zero_grad()
            batch_loss = batch_loss / j
            batch_loss.backward()
            if isinstance(self.student_model, DETR) and self.student_model.detr_args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(),
                                               self.student_model.detr_args.clip_max_norm)
            self._optimizer.step()
        # print(f' Total Loss: {epoch_loss}')
        return epoch_loss

    def _loss_on_val(self, val_dl):
        loss = 0
        for data in tqdm(val_dl, desc=' validating image: ', disable=not self.verbos):
            imgs = []
            targets = []
            for d in data:
                imgs.append(d[0].to(self.device))
                if isinstance(d[1], dict):
                    targ = {'boxes': d[1]['boxes'].to(self.device), 'labels': d[1]['labels'].to(self.device)}
                else:
                    targ = d[1].to(self.device)
                targets.append(targ)
            with torch.no_grad():
                loss += self.student_model(imgs, targets, only_loss=True)
        loss = loss.cpu().detach().numpy()
        return loss

    def _init_params(self):
        return [0] + [torch.tensor(0.0).to(self.device) for _ in range(9)]
