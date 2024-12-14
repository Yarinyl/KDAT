from Defender.BaseDefender import BaseDefender
from torch import nn
import torch.nn.functional as F
import torch
from ObjectDetection.DETR import DETR
from ObjectDetection.ModDETR.misc import nested_tensor_from_tensor_list


class TransformerBasedDefender(BaseDefender, nn.Module):
    def __init__(self, device, config, exp_dir=None):
        super(TransformerBasedDefender, self).__init__(device, config, exp_dir)
        self._embedding_loss_function = self.config.loss_parameters['adjustable_loss_function']

    def _load_base_models(self):
        with torch.no_grad():
            t_model = DETR(self.device, self._detr_args)
        s_model = DETR(self.device, self._detr_args)
        return t_model, s_model

    def _freeze_backbone(self):
        for param in self.student_model.model.backbone.parameters():
            param.requires_grad = False

    def _get_losses(self, clean_imgs, dirty_imgs, masked_imgs, targets):
        objectness_student_dirty, loss, od_loss_clean_student, od_loss_dirty, back_loss_clean, back_loss_dirty, \
        embedding_loss_clean, embedding_loss_dirty, cls_loss_clean, cls_loss_dirty = self._init_params()

        clean_output_student, clean_embedding_student, od_loss_clean_student = self.student_model(clean_imgs, targets)
        clean_output_teacher, clean_embedding_teacher, od_loss_clean_teacher = self.teacher_model(clean_imgs, targets)

        poa_student_clean = self._prepare_proposals_with_labels(clean_output_student, self.temperature_cls)
        poa_teacher_clean = self._prepare_proposals_with_labels(clean_output_teacher, self.temperature_cls)

        features_student_clean = self._get_features(self.student_model.model)

        if od_loss_clean_teacher < od_loss_clean_student:
            guiding_embedding = clean_embedding_teacher
            guiding_poa = poa_teacher_clean
            guiding_features_clean = self._get_features(self.teacher_model.model)
            _ = self.teacher_model(masked_imgs, targets)
            guiding_features_dirty = self._get_features(self.teacher_model.model)
        else:
            guiding_embedding = clean_embedding_student
            guiding_poa = poa_student_clean
            guiding_features_clean = self._get_features(self.student_model.model)
            _ = self.student_model(masked_imgs, targets)
            guiding_features_dirty = self._get_features(self.student_model.model)

        if self._loss_parameters['use_feature_loss']:
            back_loss_clean = self._backbone_loss(features_student_clean, guiding_features_clean)

        if self._loss_parameters['use_cls_loss']:
            cls_loss_clean = self._get_cls_loss(poa_student_clean, guiding_poa)
        del poa_student_clean

        for dirty_image in dirty_imgs:
            dirty_output_student, dirty_embedding_student, od_loss_dirty_student = self.student_model([dirty_image],
                                                                                                      targets)
            back_loss_dirty += self._backbone_loss(self._get_features(self.student_model.model), guiding_features_dirty)
            od_loss_dirty += 1 / len(dirty_imgs) * od_loss_dirty_student
            if self._loss_parameters['use_cls_loss']:
                poa_student_dirty = self._prepare_proposals_with_labels(dirty_output_student, self.temperature_cls)
                cls_loss_dirty += 1 / len(dirty_imgs) * self._get_cls_loss(poa_student_dirty, guiding_poa)
                del poa_student_dirty
            if self._loss_parameters['use_adjustable_loss']:
                embedding_loss_dirty += 1 / len(dirty_imgs) * self._get_adjustable_loss(dirty_embedding_student[-1],
                                                                                        guiding_embedding[-1])
            torch.cuda.empty_cache()
        del guiding_poa

        if self._loss_parameters['use_adjustable_loss']:
            embedding_loss_clean = self._get_adjustable_loss(clean_embedding_student[-1], guiding_embedding[-1])

        return [od_loss_clean_student, back_loss_clean, embedding_loss_clean, cls_loss_clean,
                od_loss_dirty, back_loss_dirty, embedding_loss_dirty, cls_loss_dirty]

    def _get_adjustable_loss(self, student_values, teacher_values):
        return self._embedding_loss_function(student_values, teacher_values)

    @staticmethod
    def _get_features(model):
        return model.transformer.memory

    @staticmethod
    def _prepare_proposals_with_labels(output, temperature):
        class_logits = output['pred_logits'].squeeze()
        boxes = output['pred_boxes'].squeeze()
        pred_scores = F.softmax(class_logits / temperature, -1)
        return boxes, pred_scores
