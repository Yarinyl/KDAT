This file describes the required adjustments to the original object detection model in order to apply KDAT.
The adjustments are essential for the KDAT training process, but after the training phase is over and the weights are saved, they can be loaded into the original model without any additional modifications.
The flowing changes are regarding the publicly available code repositories from the torch-vision library or the corresponding paper Git Hub.

Two-stage Object Detection Models:
The RPN. Originally, the RPN calculates the objectness values but returns only the proposal boxes and regression loss. 
In our method, we utilized the objectness values for the training. Thus, they need to return as well.

The ROI. Originally, the ROI receives the proposals from the RPN and calculates matching classification logits, which are converted to probabilities using softmax.
In our method, we add an internal method that 1) performs softmax with our selected temperature value and 2) filters the proposals and their matching probability vectors, eliminating proposals related to the background (class zero).
The final coupled boxes with matching probability vectors (POA) are returning from the ROI to the main detection process.

GeneralizedRCNN. Finally, all the collected values (objectness and POA) are then transferred at the end of the detection pipeline, together with the original values (losses and detections).


Transformer-based Object Detection Models:
Transformer. In the original architecture, the features of the image are forwarded to the transformer, which is composed of a transformer-encoder and a transformer-decoder.
The encoder output is a compressed representation of the features obtained from the image, which is utilized in our method's L_FM component.
Therefore, the embedded representation of a given image needs to be saved for further computations.

The output of the transformer component is an embedded representation of the detection features.
This embedded representation is then forwarded to create bounding boxes and class predictions of the given image.
Our method's L_FAL component utilizes the detection features embedded representation, thus needing to be returned from the detection process together with the original values (losses and detections).