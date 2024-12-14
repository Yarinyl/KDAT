import cv2
import torch
import torchvision.transforms as T
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


def box_cxcywh_to_xyxy(x):
    # Convert [c_x, c_y, w, h] to [x_min, y_min, x_max, y_max]
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b


def predict(input_tensor, model, detection_threshold, image):
    outputs = model(input_tensor)
    probabilities = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Ignore the last class (no object)
    boxes = outputs['pred_boxes'][0]

    keep = probabilities.max(-1).values > detection_threshold

    probabilities = probabilities[keep]

    # Rescale bounding boxes to the image size
    bboxes_scaled = rescale_bboxes(boxes[keep], image.size)
    classes = [coco_names[p.argmax()] for p in probabilities]
    scores = [p.max() for p in probabilities]

    return bboxes_scaled, classes, scores


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
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(800),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    boxes, classes, scores = predict(image_tensor, model, 0.7, image)
    pred_image = plot_image_with_boxes(image, boxes, classes, scores)
    return pred_image


def create_visu(clean_image_path, attacked_image_path):
    # Load the Faster R-CNN model
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval().to(device)
    attacked_image = Image.fromarray(np.load(attacked_image_path))
    clean_image = Image.open(clean_image_path)
    pred_clean_image_undefended = creat_pred(model, clean_image)
    pred_attacked_image_undefended = creat_pred(model, attacked_image)

    model.load_state_dict(torch.load("/KDAT/weights//KDAT_detr_coco.pt"))
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
create_visu('/KDAT/Demo/DETR/000000174004.jpg',
            '/KDAT/Demo/DETR/000000174004_DPatch_524_336_656_468.npy')
