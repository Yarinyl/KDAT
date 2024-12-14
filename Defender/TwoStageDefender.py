from Defender.BaseDefender import BaseDefender
from torch import nn
import torch
from ObjectDetection.FasterRCNN import FasterRCNN

class TwoStageDefender(BaseDefender, nn.Module):
    def __init__(self, device, config, exp_dir=None):
        super(TwoStageDefender, self).__init__(device, config, exp_dir)
        self._objectness_loss_function = self.config.loss_parameters['adjustable_loss_function']

    def _load_base_models(self):
        with torch.no_grad():
            t_model = FasterRCNN(self.device, self.config.dataset)
        s_model = FasterRCNN(self.device, self.config.dataset)
        s_model.model.roi_heads.temperature = self.temperature_cls
        t_model.model.roi_heads.temperature = self.temperature_cls
        return t_model, s_model

    def _freeze_backbone(self):
        for param in self.student_model.model.backbone.parameters():
            param.requires_grad = False

    def _get_losses(self, clean_imgs, dirty_imgs, masked_imgs, targets):
        objectness_student_dirty, loss, od_loss_clean_student, od_loss_dirty, back_loss_clean, back_loss_dirty, \
        objectness_loss_clean, objectness_loss_dirty, cls_loss_clean, cls_loss_dirty = self._init_params()

        objectness_student_clean, poa_student_clean, od_loss_clean_student = self.student_model(clean_imgs, targets)
        objectness_teacher_clean, poa_teacher_clean, od_loss_clean_teacher = self.teacher_model(clean_imgs, targets)
        torch.cuda.empty_cache()

        features_student_dirty = self._get_features(self.student_model.model, dirty_imgs)
        features_student_clean = self._get_features(self.student_model.model, clean_imgs)

        if od_loss_clean_teacher < od_loss_clean_student:
            guiding_objectness = objectness_teacher_clean[-1]
            guiding_poa = poa_teacher_clean
            guiding_features_clean = self._get_features(self.teacher_model.model, clean_imgs)
            guiding_features_dirty = self._get_features(self.teacher_model.model, masked_imgs)
        else:
            guiding_objectness = objectness_student_clean[-1]
            guiding_poa = poa_student_clean
            guiding_features_clean = features_student_clean
            guiding_features_dirty = self._get_features(self.student_model.model, masked_imgs)

        if self._loss_parameters['use_feature_loss']:
            back_loss_dirty, back_loss_clean = self._get_feature_loss(features_student_dirty, guiding_features_dirty,
                                                                      features_student_clean, guiding_features_clean)

        if self._loss_parameters['use_cls_loss']:
            cls_loss_clean = self._get_cls_loss(poa_student_clean, guiding_poa)
        del poa_student_clean

        for dirty_image in dirty_imgs:
            objectness_student_dirty_score, poa_student_dirty, od_loss_dirty_curr = self.student_model([dirty_image],
                                                                                                       targets)
            od_loss_dirty += 1 / len(dirty_imgs) * od_loss_dirty_curr
            if self._loss_parameters['use_cls_loss']:
                cls_loss_dirty += 1 / len(dirty_imgs) * self._get_cls_loss(poa_student_dirty, guiding_poa)
            if self._loss_parameters['use_adjustable_loss']:
                objectness_student_dirty += 1 / len(dirty_imgs) * objectness_student_dirty_score[-1]
            torch.cuda.empty_cache()
        del poa_student_dirty, guiding_poa

        if self._loss_parameters['use_adjustable_loss']:
            objectness_loss_dirty = self._get_adjustable_loss(objectness_student_dirty, guiding_objectness)
            objectness_loss_clean = self._get_adjustable_loss(objectness_student_clean[-1], guiding_objectness)

        if not self._loss_parameters['use_od_loss']:
            od_loss_clean_student = torch.tensor(0.0).to(self.device)
            od_loss_dirty = torch.tensor(0.0).to(self.device)

        return [od_loss_clean_student, back_loss_clean, objectness_loss_clean, cls_loss_clean,
                od_loss_dirty, back_loss_dirty, objectness_loss_dirty, cls_loss_dirty]

    def _get_adjustable_loss(self, student_values, teacher_values):
        return self._objectness_loss_function(student_values, teacher_values)

    def _get_features(self, model, images):
        if self._loss_parameters['use_feature_loss']:
            features = [model.backbone(image)['pool'] for image in images]
            if self._loss_parameters['hint_source'] == 'backbone':
                return features
            else:
                return [model.rpn.head.conv(feature) for feature in features]
