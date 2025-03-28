Skin Cancer Classification using YOLO
Project Banner

This repository contains an implementation of a YOLO (You Only Look Once) based model for detecting and classifying skin cancer lesions from dermatoscopic images. The project aims to assist in early detection of skin cancers, which is crucial for effective treatment.

Table of Contents
Features
Dataset
Installation
Usage
Training
Evaluation
Results
Contributing
License
Features
YOLOv5/YOLOv8 implementation for skin cancer detection
Support for multiple classes of skin lesions
Customizable model architecture
Training and evaluation scripts
Export to various formats (ONNX, TFLite, etc.)
Comprehensive metrics visualization
Dataset
The model is trained on the ISIC (International Skin Imaging Collaboration) dataset, which contains thousands of dermatoscopic images with various types of skin lesions. The dataset includes:

Melanoma
Melanocytic nevus
Basal cell carcinoma
Actinic keratosis
Benign keratosis
Dermatofibroma
Vascular lesion
Installation
Clone the repository:

git clone https://github.com/yourusername/SkinCancerClassification.git
cd SkinCancerClassification
Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

pip install -r requirements.txt
Usage
Inference on single image
python detect.py --weights best.pt --source test_image.jpg
Webcam demo
python detect.py --weights best.pt --source 0
Training
To train the model on your custom dataset:

Prepare your dataset in YOLO format
Modify the configuration file (data/skin_cancer.yaml)
Run the training script:
python train.py --img 640 --batch 16 --epochs 100 --data data/skin_cancer.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt
Evaluation
Evaluate the model performance:

python val.py --weights best.pt --data data/skin_cancer.yaml --img 640
Results
Performance metrics on the test set:

Metric	Value
Precision	0.89
Recall	0.85
mAP@0.5	0.87
mAP@0.5:0.95	0.62
Sample detection examples:

Sample Detection 1 Sample Detection 2

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.

Disclaimer: This project is for research purposes only and should not be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for medical concerns.
