import torch
from ObjectDetection.ModDETR.detr import build_model, PostProcess, get_criterion
from ObjectDetection.Object_Detector import ObjectDetector
from PIL import Image
import ObjectDetection.ModDETR.transforms as T
from ObjectDetection.ModDETR.util import box_xyxy_to_coco, box_xyxy_to_cxcywh
import numpy as np


class DETR(ObjectDetector):
    def __init__(self, device, detr_args, dataset='coco', num_classes=None):
        self.detr_args = detr_args
        self.dataset = dataset
        super().__init__(device, self.load_model(), dataset)
        self.transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor()])
        self.normalize = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.criterion = get_criterion(self.detr_args, num_classes).to(self.device).train()
        self.detection_threshold = 0.7

    def load_model(self):
        model = build_model(self.detr_args)
        model.load_state_dict(
            torch.load(self.detr_args.weights))
        return model

    def load_image(self, image_path, resize=False):
        if image_path.endswith('.npy'):
            image = Image.fromarray(np.load(image_path))
        else:
            image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if resize:
            image = image.resize((800, 800), Image.LANCZOS)
        w, h = image.size
        orig_size = torch.as_tensor([int(h), int(w)])
        image = self.transform(image, None)[0]
        if image.shape[0] == 1:
            image = image.expand(3, -1, -1)

        return image, orig_size, (1, (0, 0))

    def load_targets(self, new_image_size, org_image_size, org_target, scale_params):
        boxes = org_target['boxes']
        boxes = box_xyxy_to_coco(boxes)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=org_image_size[0])
        boxes[:, 1::2].clamp_(min=0, max=org_image_size[1])

        classes = org_target['labels']
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_image_size, org_image_size))
        ratio_width, ratio_height = ratios

        target = org_target.copy()
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])

        w, h = new_image_size
        boxes = box_xyxy_to_cxcywh(scaled_boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        target["boxes"] = boxes
        target["labels"] = classes

        return target

    def forward(self, x, y=None, only_loss=False):
        x = [self.normalize(x[i], None)[0] for i in range(len(x))]
        output = self.model(x)
        if not self.model.training:
            return output
        prediction, embedding = output
        loss_dict = self.criterion(prediction, y)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        if y is not None and only_loss:
            return loss
        return prediction, embedding, loss

    def general_process_pred(self, input_tensor, outputs, orig_size, scale_params):
        postprocessors = PostProcess()
        orig_target_sizes = torch.stack([orig_size], dim=0)
        results = postprocessors(outputs, orig_target_sizes)
        return self.process_preds(results, detection_threshold=self.detection_threshold)
