# AI Frame work Robin Herickx 

This project combines InsightFace and YOLO models to detect and recognize faces in images. It processes images from an input folder, identifies people using facial embeddings, and generates annotated output images with bounding boxes and labels.

## Features

- **Multi-method face detection**: Uses InsightFace, flipped image detection, and YOLO for robust face detection
- **Face recognition**: Matches detected faces against a folder of embeddings
- **Prediction merging**: Combines results from multiple detection methods
- **CSV export**: Generates Kaggle-compatible CSV files with predictions

## Installation

### 1. Install PyTorch (Optional but Recommended for GPU)

For GPU acceleration, install the correct PyTorch package based on your CUDA version:

Visit: https://pytorch.org/get-started/locally/


### 2. Install Required Dependencies

Install all required packages from [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```



## Usage

### 1. Prepare Your Input

- Place all images you want to process in the `input/` folder

### 2. Run the Script

Execute the main script:

```bash
python faceappDetectionmodel.py
```

### 3. Check the Results

- **Annotated images**: Check the `output/` folder for processed images with bounding boxes, names, and confidence scores
- **Predictions CSV**: Find CSV files in the `kagle/` folder with format: `Robin_faceprodictions_X.csv`

## How It Works

The script uses three detection methods:

1. **InsightFace Detection**: Standard face detection using the buffalo_l model
2. **Flipped Detection**: Detects faces in horizontally flipped images to catch missed faces
3. **YOLO Detection**: Uses a custom-trained YOLO model ([`best.pt`](best.pt)) for additional detection

The [`merge_predictions`](faceappDetectionmodel.py) function combines results from all three methods, removes duplicates, and selects the best predictions based on confidence scores.

## Configuration

You can modify settings at the top of [`faceappDetectionmodel.py`](faceappDetectionmodel.py):

```python
embed_dir = r"finalProjeckt/embeddings_buffalo_l"  # Embedding database location
input_dir = Path(r"finalProjeckt/input")            # Input images folder
output_dir = r"finalProjeckt/output"                # Output images folder
yolo_model_path = r"finalProjeckt/best.pt"          # YOLO model path
```
