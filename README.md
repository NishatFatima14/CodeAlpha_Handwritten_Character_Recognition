# CodeAlpha_Handwritten_Character_Recognition
An advanced Handwritten Digit Recognition system using a Residual CNN on the MNIST dataset. Features data augmentation, model evaluation with confusion matrix, and Grad-CAM interpretability. Includes an interactive Tkinter GUI for real-time digit drawing and prediction.

# CodeAlpha_HandwrittenRecognition

This repository contains an end-to-end Handwritten Character Recognition project for CodeAlpha internship.

## Structure
```
CodeAlpha_HandwrittenRecognition/
  train_model.py        # training script - trains and saves model in saved_model/
  gui_predict.py        # Tkinter GUI to draw and predict using saved model
  saved_model/          # models (generated after training)
  confusion_matrix.png  # generated after training
  README.txt            # small artifact list
```

## Quick start

1. Install dependencies (recommended in a virtual environment):
```
pip install tensorflow scikit-learn numpy matplotlib pillow seaborn
```

2. Train the model (this creates `saved_model/final_model.h5`):
```
python train_model.py
```

3. Run GUI for live prediction:
```
python gui_predict.py
```

## Notes
- `train_model.py` uses MNIST by default. To use EMNIST, update the dataset loader accordingly.
- The GUI expects `saved_model/final_model.h5` (or `.keras`) file present.
- For faster experiments, reduce `EPOCHS` in `train_model.py`.

