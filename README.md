# Chest X-ray Pneumonia Detection | YOLOv8, PyTorch

Detection of pneumonia areas (bounding boxes) on chest X-rays from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge). A YOLOv8n detector is compared against no-learning baselines, and a ResNet18 classifier is used to test where the errors come from.

<a href="https://www.kaggle.com/code/ivanventyk/chest-x-ray-pneumonia-detection-yolov8-pytorch"><img src="https://img.shields.io/badge/View_notebook_on-Kaggle-F8C840?labelColor=1F1F1F&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAE%2BUlEQVR42u2bWYgcVRSGv7pVMzi45kHREAiiPhmCJCgKUVFx3xANuAzuiAriAoI7LqgRNIK7EEQRI4oLjlGEiA7og6D4oBjFgDD4IKKIYtA409Xlwz0HD9Izdaumum73TF8omO7pqr7nv%2Bee%2F5z%2F3E52f34KPUYCFMAq4HtgQl4n%2F%2FtcB8iAR4Db5O8OQzRcyf%2FngC5LeJQBkLDEh2OZjxEAIwBGAIwAGAEwAmD%2BUSx3AMaWupe4kgxwP6kDlmxWmA3JPNOaC9Atq2WyITE%2BX8T9yUKxbJABSGSL5sC5cnUDY9IcsA%2FwNfAYMDsfCFnAJGIbfxPweI1n7ADuFeNrb4FZmUTacmDuyvduBm4O2cuywjmwB%2FAKcDlenKm1BfSG3fKQlN6KUL%2F2%2BwTwErDRfH9WYnxXaPtR4FbjRbWCoKI2bla%2F38arnLYSeAM4hv8kt7JIn8g8b5HtkgZ6TakH%2FCFesFefPUCNXwe8DhwSaLxuz3%2BAS%2BXeTN4vQr%2B4bD%2B2tfJnAa8K2HnA3BSgX4DzgU%2BoIcrGTHPVbTvAtcCUMT4NNP47YENd42MC4IwLbwKeFZctKhj%2FEXAcXravLcfHSIQ00o8DLwCXmEifBNBcBrwMXGnu6yx2JdqsPXLgQGC7MT4LML6Qz22SgJebZGkoiiF10zXAW8BhFWhOg%2FGNwBNVaK4tAIpA408DtgIrKtLcX8Ak8DYNt9%2Ba2gJj87iwM3v0KmCbGB9KcynwE3CSGD9Gw71H19D9JxgvyORKTU7%2FALDFpKahkf4b4HjgM3k917YkFlq4rJMV7pgrBw4H3gPuMvvVBRq%2FHTgW2LnYSN%2FvGKB1wxZx1Y8lodkAnCHVWUhyowVNBrwIXG3uy%2BnTaAoAHRfJ1SuQlRmvSdCD4jFJEzTXNg3aIiQxQbDMeAXxOuC5JmmubQCqCicK1i5JiqZo%2BZRJTE2wMKu8EfhAaG6uzUnErAY1NmwW48fbNj4mAJrX%2F45XbR2RDlfFAkCD21ciaECkw1ix%2B35%2FCwNEa7vFBsCZHGBZAaArvobIzdeYklgOHARcbFLgZbUFtJB6GDhUKDBbTgCoy%2B8PfFhBIRpoALqmHO5W8ILVwKfA0W2D4Bo23hlBxAUWNBoPDsBL3ecQJpQOFACFPGsHvrvzJvCDvBdS0mr1NwG8I1VhhxY6U66hlU%2BAJ4G1EtUvEDXoQmAmUNFxpkB6RnSBvN%2BxyjXk9jP4gwwqdqb4puprwJF4MVTL3KIkMCbyuTvwDZAxwnTEaAAg1VzXGJmLIdq8PBvf%2Fsrkc2Ug6HMmqaYkR4sBv%2FXYq4XZxw64HrjTrGRZcFQQTgamgYP7wRBNAZAusKpdU%2F4%2BJMrPbGBwVBDWCk0e1TQITbJA2f914lvxHaKfA4Oj9hNXCk2e2SQIbWeCOvFpvOb%2FLWEaoNLknsC7wDWEdZQHMhVWEHbi%2B%2FvTgSA4s52eB%2B5rgiZj1QK6er8Cp%2BKPtYXQpCZGOXAPvoGSLoYmY4uiTgLiJL7vn5lkaCGa1NhxGfA%2BsC81zzPGVoTsEbfbgRsIPN%2FXgyZX18kVBuG3APboy1NSDP1ZkSaPEJpcX5UhBunHEDrxbcCJwI8VaXIVvjF7ehUQBu3XIDrxL4QmvzQGhtDk3kKTV5gsdKgAsAwxgz8cMWUMDKHJBH%2F67G7zeqgAwET0XcB5QpMuEASlyfuBp%2FEtt3nt%2FBfcU2ClwFaupAAAAABJRU5ErkJggg%3D%3D&style=flat" alt="View notebook on Kaggle" height="30"></a>

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge) ![Ultralytics](https://img.shields.io/badge/Ultralytics-042AFF?logo=ultralytics&logoColor=white&style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge) ![pydicom](https://img.shields.io/badge/pydicom-4B8BBE?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAACYElEQVR42u3dUU7DMBBFUWJl%2F1s2PyBVqAgqpfF43rkLQMRz%2FTyTJu3HBwAAAAAAACI4rvpDc84Zs2jHcRAgWIBOUhAgXAgChMswlOf9zC8IQIRyIiw%2FAn7Go2kiKAGeLUCnEWuHvmlUtJ8EegC9AQEyRSAACSYBSDAJQIJJABBACjQUIGnOry7BqfCOgFsKr%2Fg1U2DY9RIAwSkw7P5szg4XcYVoOzyHMOecV2%2Bq0x54LlHKgyljhcV%2FLW6FxU%2BZXMZdO2rXXdVdBFPAZg3t1ZuIAOFTzdjFVBI0OALcF9ADSIF0AaSABECh%2Fmqs%2BEelgARAigC%2FpcDuY2KXFJMAEmCvpqVjIxaRAL5CxhHwkgQmBT0ACToK8Oq78CR475qMtAvWzxgDsaMAXZ%2FIIYDZnwCKTwDFX3QUlhTgt4u8uw%2Bo%2FB2%2FV%2BHNoJCobyFA14%2BWV05CegD3AUAARMb%2FcgFeObs9T9CsCXzFaL9I9r7xdyh%2BbvGXCKD4wU2g4tfa%2FUt7AIVfX%2FySY6BHwO5dA%2FcBwjcAAcLTjwDhR1%2FZF0MU%2Fx48DxDe8C55MUTp60w7Z%2BXF6ShLtTF3iQDfhU2a%2Batea7kp4HHX7y7I8UDV%2F1ETGLLTCaDgewpwHMexshns3qecHXaUD5AaNYHuGxAABAAB0F%2BAHX49jABGLAKsKL7d31gAxZcAik8ARAqg8ZMAIAAIgEwBHrt8Hb8EQKIAdr4EQLoAUkACPKX6c%2FQEAAHuSgLlkQAgAAiAPAGMiRIAN%2BCbQiWA4hNA0R0BJJAAIAAIAAKAANvgU0MAAAAAAADgH3wCW1g9D8t2VS4AAAAASUVORK5CYII%3D&style=for-the-badge) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white&style=for-the-badge) ![matplotlib](https://img.shields.io/badge/matplotlib-11557C?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNTYgMjU2Ij48Y2lyY2xlIGN4PSIxMjgiIGN5PSIxMjgiIHI9IjExMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjE0Ii8%2BPGc%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0ibTEwNy44IDM2IC4xLjcgMjAuOCA5MS4xLTEuMy43LTU5LjktNzEuOC0uNS0uNi42LS41QTk0IDk0IDAgMCAxIDEwNyAzNi4xeiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Im01NCA5MS41LjYuMyA3My43IDM1LjUtLjIgMS40LTgwLjUgMTQuNi0uNy4yLS4xLS44YTgzIDgzIDAgMCAxIDYuOC01MC41eiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Im0xMjcuNyAxMjcuMy43IDEuM0w5MCAxNTUuMmwtLjYuNS0uNC0uNmE0OCA0OCAwIDAgMS0zLjgtNi41bC0uMy0uNy43LS4zeiIvPjxwYXRoIGZpbGw9IiNmZmYiIGQ9Im0xMjguNyAxMjcuOSA5LjggNTcuNi4yLjctLjguMmE1OSA1OSAwIDAgMS0yMy0uN2wtLjgtLjIuMi0uNyAxMy01N3oiLz48cGF0aCBmaWxsPSIjZmZmIiBkPSJtMTI4LjUgMTI3LjQgNzMuNyA1Ny41LjYuNC0uNS42YTk0IDk0IDAgMCAxLTE1LjUgMTUuOGwtLjYuNS0uNS0uNi01OC4zLTczLjF6Ii8%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0ibTE1MC43IDExOS44LjIuN2EyNCAyNCAwIDAgMSAxLjIgNy41di44aC0yMy42bC4xLTEuOCAyMS40LTd6Ii8%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0ibTE1Mi4yIDYxLjQuNy4yYTcxIDcxIDAgMCAxIDE5LjMgMTFsLjYuNS0uNS41LTQzLjcgNTQuOS0xLjMtLjggMjQuNi02NS42eiIvPjwvZz48Y2lyY2xlIGN4PSIxMjgiIGN5PSIxMjgiIHI9IjYiIGZpbGw9IiNmZmYiLz48L3N2Zz4%3D&style=for-the-badge)

## 📊 Results

Validation set: 5,337 images (1,203 with pneumonia, 1,898 true boxes). A predicted box counts as correct if IoU ≥ 0.5.

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Baseline: no boxes | – | 0.000 | 0.000 |
| Baseline: two fixed median boxes on every image | 0.028 | 0.159 | 0.048 |
| **YOLOv8n, confidence ≥ 0.10 (main)** | 0.361 | 0.448 | **0.400** |
| YOLOv8n, confidence ≥ 0.25 | 0.681 | 0.153 | 0.250 |

**Bold** = best result. F1 combines precision and recall into one number (computed from the two columns).

| Image-level classifier ("is there pneumonia?") | PR-AUC |
|---|---|
| Random guess | 0.225 |
| ResNet18, 256 px, seed 42 | 0.666 |
| ResNet18, 256 px, seed 43 | 0.691 |

- YOLOv8n beats the fixed-box baseline on precision and recall at the same time.
- The detector's confidences are low: at 0.25 it is precise but finds few boxes, and at 0.5 it keeps a single box on the whole validation set.
- Most false alarms come from images labeled "No Lung Opacity / Not Normal": 607 false boxes there vs 4 on Normal images. The classifier shows the same pattern, so it is a property of the data rather than of one model.
- The seed alone moves the classifier's PR-AUC by about 0.03, so smaller differences between single runs are treated as noise.
- Examples show duplicate overlapping boxes on correct areas and misses on small boxes.

<!-- >>> IMAGE: put det1_conf_tradeoff.png (from the Kaggle notebook's Output tab) into the images/ folder <<< -->
![Precision and recall vs confidence threshold](images/det1_conf_tradeoff.png)

*Detector precision and recall at different confidence thresholds (IoU 0.5); grey lines show the fixed-box baseline. At 0.4 and above only 1–2 boxes remain, so precision there is not meaningful.*

<!-- >>> IMAGE: put det1_curves.png (from the Kaggle notebook's Output tab) into the images/ folder <<< -->
![Training curves of the detector](images/det1_curves.png)

*Detector training: box loss (left) and YOLO's own validation metrics (right). Both losses fall together, and mAP50 is still rising at epoch 10.*

<!-- >>> IMAGE: put examples_det1.png (from the Kaggle notebook's Output tab) into the images/ folder <<< -->
![Detector examples: hits, misses and false alarms](images/examples_det1.png)

*Green: true boxes, red dashed: YOLO boxes with confidence ≥ 0.1. Two hits, two misses and two false alarms on "Not Normal" images.*

## 🛠️ Approach

1. **Data.** 26,684 DICOM images, 1024×1024. 22.5% have pneumonia (1–4 boxes each); the rest are Normal or "Not Normal". Stratified 80/20 split by these three classes.
2. **Preprocessing.** DICOM converted to 512×512 PNG with boxes scaled by 0.5. Boxes were checked on images and for bounds, and the YOLO label files were read back and drawn.
3. **Metric.** Hand-written IoU (checked on known cases) and per-class true/false positives at IoU ≥ 0.5, so baselines and models are scored the same way.
4. **Baselines.** "No boxes" and "two median boxes from the training set on every image".
5. **Classifier.** Pretrained ResNet18, 256 px, 3 epochs, two seeds: used to test which images cause false alarms and to measure run-to-run noise.
6. **Detector.** Pretrained YOLOv8n, 512 px, 10 epochs, evaluated with the same metric as the baselines.

## ⚠️ Limitations

- One detector run with the smallest model and 10 epochs; mAP50 was still rising slightly at the last epoch.
- Only 1,203 validation images contain pneumonia, so small differences may be noise.
- Confidence thresholds were compared on the validation set itself, with no separate test set and no Kaggle leaderboard submission.
- The competition metric is stricter (averaged over IoU 0.4–0.75); this project uses a single IoU threshold of 0.5.
- The split is by patientId; it cannot be verified that one person has no several IDs.
- The classifier and the detector were not combined (e.g. the classifier as a filter for false alarms).
- GPU training is not fully deterministic. The notes inside the notebook were written after an earlier run; this README uses the numbers of the published run.

## ▶️ How to run

Open the notebook on Kaggle with the competition data attached, enable GPU and Internet (pip install and pretrained weights), and Run All (about 1.5–2 hours, mostly DICOM conversion and YOLO training).
Outputs in `/kaggle/working`: `results.csv` and `results_boxes.csv` (box metrics per class), `results_classifier.csv`, the three figures above and `runs/` (model weights).
