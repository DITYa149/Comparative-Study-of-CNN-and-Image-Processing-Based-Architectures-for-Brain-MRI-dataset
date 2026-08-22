# Comparative-Study-of-CNN-and-Image-Processing-Based-Architectures-for-Brain-MRI-dataset
 Comparative study of CNN and Sigmoid–Otsu approaches for brain image analysis. CNN classifies brain tumor, stroke, MS, and normal cases, while the image-processing model uses Otsu thresholding, sigmoid classification, morphological operations, and contour detection for tumor localization.

Comparative-Study-of-CNN-and-Image-Processing-Based-Architectures-for-Brain-MRI-dataset/
│
├── README.md (Main documentation)
│
├── CNN_MODEL/
│   ├── preprocessing.py (Image normalization)
│   ├── model.py (CNN architecture)
│   ├── train.py (Training pipeline)
│   └── predict.py (Prediction with GUI)
│
├── SIGMOID_OTSU_MODEL/
│   ├── preprocessing.py (Image feature extraction)
│   ├── model.py (Sigmoid + Otsu model)
│   ├── train.py (Training with logistic regression)
│   └── predict.py (GUI with contour detection)
│
├── requirements.txt
└── dataset/ (to be filled with Keras dataset)
