import torch

from ObjectDetection.ModFaster.faster import fasterrcnn_resnet50_fpn_cust, FasterRCNN_ResNet50_FPN_Weights, \
    FastRCNNPredictor
from ObjectDetection.Object_Detector import ObjectDetector
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class FasterRCNN(ObjectDetector):
    def __init__(self, device, dataset='coco'):
        self.dataset = dataset
        super().__init__(device, self.load_model(), dataset)
        self.transform = transforms.Compose([transforms.ToTensor(), ])
        self.detection_threshold = 0.8 if self.dataset != 'SuperStore' else 0.7

    def load_model(self):
        model = fasterrcnn_resnet50_fpn_cust(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        if self.dataset == 'SuperStore':
            model = fasterrcnn_resnet50_fpn_cust()
            num_classes = 21
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.load_state_dict(torch.load(self.superstore_weights))
        return model

    def load_image(self, image_path, resize=False):
        if image_path.endswith('.npy'):
            image = Image.fromarray(np.load(image_path))
        else:
            image = Image.open(image_path)
        if resize:
            image = image.resize((1500, 1500), Image.LANCZOS)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h = image.size
        orig_size = torch.as_tensor([int(h), int(w)])
        image_float_np = np.float32(image) / 255
        if len(image_float_np.shape) < 3:
            image_float_np = np.expand_dims(image_float_np, axis=2)
            image_float_np = np.repeat(image_float_np, 3, axis=2)
        if len(image_float_np.shape) > 3:
            image_float_np = image_float_np[0] * 255
        image = self.transform(image_float_np)
        return image, orig_size, (1, (0, 0))

    def load_targets(self, new_image_size, org_image_size, org_target, scale_params):
        return org_target

    def forward(self, x, y=None, only_loss=False):
        if y is None:
            output = self.model(x)
            return output[0]
        loss_dict, objectness, poa = self.model(x, y)
        loss = sum(v for v in loss_dict.values())
        if y is not None and only_loss:
            return loss
        return objectness, poa, loss

    def general_process_pred(self, input_tensor, outputs, orig_size, scale_params):
        return self.process_preds(outputs, detection_threshold=self.detection_threshold)
