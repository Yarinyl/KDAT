import numpy as np
import torch
from torch.cuda.amp import autocast

from ObjectDetection.ModDETR.detr import PostProcess

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def predict_wrapper(model, image_dataloader, targets=None, detection_threshold=0.8, device=torch.device('cuda'), custmodel=False):
    prediction_dicts = []
    for i, input_tensor in enumerate(image_dataloader):
        with autocast(enabled=True):
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                outputs = model(input_tensor.unsqueeze(0))
                if custmodel:
                    outputs = outputs[0]
        if targets is None:
            prediction_dicts.append(process_preds(outputs, detection_threshold))
        else:
            prediction_dicts.append(process_preds_detr(outputs, targets[i], 0))
    # GPUtil.showUtilization()
    del outputs
    torch.cuda.empty_cache()
    # GPUtil.showUtilization()
    return np.array(prediction_dicts)


def process_preds_detr(outputs, targets, detection_threshold=0.7):
    postprocessors = PostProcess()
    orig_target_sizes = torch.stack([targets["orig_size"]], dim=0)
    results = postprocessors(outputs, orig_target_sizes)
    return process_preds(results, detection_threshold=detection_threshold)


def process_preds(outputs, detection_threshold=0.6):
    """
    Helper function to predict, that extract detection as dictionary from a raw prediction (from a single frame).
    :param outputs: required, a row tensor prediction.
    :param detection_threshold: required, int. The confidence threshold upon the detections is set.
    :return: Dictionary containing detections (bounding boxes, classes and labels).
    """
    pred_labels = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_classes = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes, classes, labels, indices, scores = filter_low_confidence_predictions(pred_classes, pred_labels,
                                                                                pred_scores, pred_bboxes,
                                                                                detection_threshold)
    return add_pred_dict(boxes, classes, labels, indices, scores)


def add_pred_dict(boxes, classes, labels, indices, scores):
    """
    Helper function to predict, that receives detection raw data (bounding box, class etc.) and form a dictionary.
    :param boxes: required, numpy array of lists representing the detection bounding box.
    :param classes: required, numpy array of strings representing the detection classes.
    :param labels: required, numpy array of ints representing the detection labels.
    :param indices: required, numpy array of ints representing image indices.
    :return: Dictionary that represent the prediction.
    """
    pred_dict = {
        'boxes': boxes,
        'classes': classes,
        'labels': labels,
        'indices': indices,
        'scores': scores
    }
    return pred_dict


def filter_low_confidence_predictions(pred_classes, pred_labels, pred_scores, pred_bboxes, detection_threshold):
    """
    Help function to 'extract_prediction' that filter low confidence predictions.
    :param pred_classes: required, numpy array of strings representing the detection classes.
    :param pred_labels: required, numpy array of ints representing the detection labels.
    :param pred_scores: required, numpy array of floats representing the detection confidence.
    :param pred_bboxes: required, numpy array of lists representing the detection bounding box.
    :param detection_threshold: required, int. The confidence threshold upon the detections is set.
    :return: The detections that their confidence was above the given threshold.
    """
    boxes, classes, labels, indices, scores = [], [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
            scores.append(pred_scores[index])
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices, scores
