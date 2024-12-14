from abc import ABC, abstractmethod

import os

import cv2
import random
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm


class ObjectDetector(ABC, nn.Module):
    def __init__(self, device, model, dataset):
        super(ObjectDetector, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_image(self, image_path, resize=False):
        pass

    @abstractmethod
    def load_targets(self, new_image_size, org_image_size, org_target, scale_params):
        pass

    def load_dataset(self, dataset_path, shuffle=False, n=-1, resize=False):
        images_paths = [os.path.join(dataset_path, img_file) for img_file in os.listdir(dataset_path)]
        images_paths = sorted(images_paths)
        if shuffle or n != -1:
            if shuffle:
                random.shuffle(images_paths)
            images_paths = images_paths[:n]
        images_with_sizes = [self.load_image(image, resize) for image in tqdm(images_paths)]
        images = [i[0] for i in images_with_sizes]
        org_sizes = [i[1] for i in images_with_sizes]
        pads = [i[2] for i in images_with_sizes]
        file_names = [image_path.split('/')[-1].split('.')[0] for image_path in images_paths]
        return images, file_names, org_sizes, pads

    def loaded_resize(self, image_path, target_size=(800, 800)):
        img = Image.open(image_path)
        img_resized = img.resize(target_size, Image.LANCZOS)
        return np.array(img_resized)

    @abstractmethod
    def forward(self, x, y=None, only_loss=False):
        pass

    def train(self, mode=True):
        self.model.train()

    def eval(self):
        self.model.eval()

    def predict_wrapper(self, image_dataloader, original_sizes, scale_params=None):
        if scale_params is None:
            scale_params = [None for _ in original_sizes]
        prediction_dicts = []
        for i, (input_tensor, org_size, sp) in enumerate(zip(image_dataloader, original_sizes, scale_params)):
            with autocast(enabled=True):
                with torch.no_grad():
                    input_tensor = input_tensor.to(self.device)
                    outputs = self([input_tensor])
                prediction_dicts.append(self.general_process_pred(input_tensor, outputs, org_size, sp))
        # GPUtil.showUtilization()
                del outputs
        torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        return np.array(prediction_dicts)

    @abstractmethod
    def general_process_pred(self, input_tensor, outputs, org_size, scale_params):
        pass

    def process_preds(self, outputs, detection_threshold=0.6):
        """
        Helper function to predict, that extract detection as dictionary from a raw prediction (from a single frame).
        :param outputs: required, a row tensor prediction.
        :param detection_threshold: required, int. The confidence threshold upon the detections is set.
        :return: Dictionary containing detections (bounding boxes, classes and labels).
        """
        pred_labels = [coco_names[int(i)] if i < len(coco_names) else 'skip' for i in outputs[0]['labels'].cpu().numpy()]
        pred_classes = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes, classes, labels, indices, scores = self.filter_low_confidence_predictions(pred_classes, pred_labels,
                                                                                         pred_scores, pred_bboxes,
                                                                                         detection_threshold)
        return self.add_pred_dict(boxes, classes, labels, indices, scores)

    @staticmethod
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

    def filter_low_confidence_predictions(self, pred_classes, pred_labels, pred_scores, pred_bboxes,
                                          detection_threshold):
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
                if self.dataset == 'inria' and pred_classes[index] != 1:
                    continue
                if pred_labels[index] == 'skip':
                    continue
                boxes.append(pred_bboxes[index].astype(np.int32))
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)
                scores.append(pred_scores[index])
        boxes = np.int32(boxes)
        return boxes, classes, labels, indices, scores

    @staticmethod
    def plot_image_with_boxes(img, boxes, pred_cls, confidence, output_path, image_id, save=True):
        """
        A function that plot the prediction on the input scene and saved it on
        a given output path.
        :param img: Required. 3D Numpy array. The input scene.
        :param boxes: Required. list of bounding boxes. Each bounding box is a list
        with 4 coordinated in Faster RCNN
        format (
        x1,y1,x2,y2).
        :param pred_cls: Required. List of strings represent the classification
        of the corresponding object.
        :param output_path: Required. String of the output path to save the plot.
        :param image_id: An id of the given img.
        :return: Saved the input image with the prediction in the given output
        path.
        """
        text_size = 0.6
        text_th = 2
        rect_th = 6
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0),
                  (128, 0, 128), (255, 105, 180), (0, 255, 0)]

        for i in range(len(boxes)):
            # img = img*255
            boxes_as_int = [int(coordinate) for coordinate in boxes[i]]
            start_point = (boxes_as_int[0], boxes_as_int[1])
            end_point = (boxes_as_int[2], boxes_as_int[3])
            # Draw Rectangle with the coordinates
            color_idx = random.randint(0, len(colors) - 1)
            cv2.rectangle(img, start_point, end_point, color=colors[color_idx], thickness=rect_th)

            # Write the prediction class

            curr_confidence = "%.2f" % confidence[i]
            prediction_text = f'{pred_cls[i]} {curr_confidence}'
            start_point = (boxes_as_int[0] + 5, boxes_as_int[1] + 20)
            cv2.putText(img, prediction_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                        (0, 150, 255), thickness=text_th)
        if save:
            fig = plt.figure(figsize=(10, 7))
            plt.axis("off")
            plt.imshow(img, interpolation="nearest")
            plt.gca().set_axis_off()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(f'{output_path}/{image_id}.jpg', dpi=fig.dpi, bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.close(fig)
        else:
            return img


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
