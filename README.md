# Knowledge Distillation with Adversarial Tuning (KDAT)

Welcome to the official implementation for the AAAI-2025 paper KDAT: Inherent Adversarial Robustness via Knowledge Distillation with Adversarial Tuning for Object Detection Models.

<div align="center">
  <img src="images/faster_DEMO.png" alt="Project Screenshot">
</div>

Leveraging OD models' modularity (interchangeable components such as backbone and detection heads), KDAT transfers informative features from various processing stages in the model's architecture.

<div align="center">
  <img src="images/TransformerBasedPipeline.png" alt="Project Screenshot">
</div>

## Installation

The required packages are detailed in the requirement file and can be installed using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using pre-trained KDAT models:
We provide the checkpoints for both Faster R-CNN and DETR in the weights directory.

### Train KDAT:

The Defender directory is a versatile resource. It contains the abstract class BaseDefender, which serves as a foundation, and two successor classes that expand this class for two-stage detectors and transformer-based detectors. This versatility allows you to adapt the implementation to your specific needs.

The Train_KDAT.py file is the tool you'll use to fine-tune your model using KDAT. This process involves preparing an adversarial dataset and updating the config file accordingly.
Creating an adversarial dataset is a key step when fine-tuning your model using KDAT. This dataset should include both training and validation sets, as demonstrated in the Demo/Adv_Dataset directory, with CSV files for each subset of the examples. 

```bash
python main.py
```
