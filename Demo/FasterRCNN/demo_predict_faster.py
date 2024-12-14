import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_image_with_boxes(img, boxes, pred_cls, confidence):
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
    img = np.array(img)
    text_size = 1.5
    text_th = 5
    rect_th = 6
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0),
              (128, 0, 128), (255, 105, 180), (0, 255, 0)]

    for i in range(len(boxes)):
        boxes_as_int = [int(coordinate) for coordinate in boxes[i]]
        start_point = (boxes_as_int[0], boxes_as_int[1])
        end_point = (boxes_as_int[2], boxes_as_int[3])
        # Draw Rectangle with the coordinates
        color_idx = 0
        cv2.rectangle(img, start_point, end_point, color=colors[color_idx], thickness=rect_th)

        # Write the prediction class

        curr_confidence = "%.2f" % confidence[i]
        prediction_text = f'{pred_cls[i]} {curr_confidence}'
        start_point = (boxes_as_int[0] - 10, boxes_as_int[1] - 5)
        cv2.putText(img, prediction_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 120, 255), thickness=text_th)
    return img


def predict(input_tensor, model, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

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


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
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


def creat_pred(model, image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    boxes, classes, labels, indices, scores = predict(image_tensor, model, 0.8)
    pred_image = plot_image_with_boxes(image, boxes, classes, scores)
    return pred_image


def create_visu(clean_image_path, attacked_image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval().to(device)
    attacked_image = np.load(attacked_image_path)
    clean_image = Image.open(clean_image_path)
    pred_clean_image_undefended = creat_pred(model, clean_image)
    pred_attacked_image_undefended = creat_pred(model, attacked_image)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(torch.load(
        "KDAT/weights/KDAT_faster_coco.pt"))
    model.eval().to(device)
    pred_clean_image_defended = creat_pred(model, clean_image)
    pred_attacked_image_defended = creat_pred(model, attacked_image)

    # Display the original image and the saliency map
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0, 0].imshow(clean_image)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Input Image')
    ax[0, 1].imshow(pred_clean_image_undefended)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Prediction Vanilla')
    ax[0, 2].imshow(pred_clean_image_defended)
    ax[0, 2].axis('off')
    ax[0, 2].set_title('Prediction KDAT')
    ax[1, 0].imshow(attacked_image)
    ax[1, 0].axis('off')
    ax[1, 1].imshow(pred_attacked_image_undefended)
    ax[1, 1].axis('off')
    ax[1, 2].imshow(pred_attacked_image_defended)
    ax[1, 2].axis('off')
    plt.show()


create_visu('KDAT/Demo/Faster/000000308476.jpg',
            'KDAT/Demo/Faster/000000308476_PGD_303_189_431_317.npy')
