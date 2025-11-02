# Object_Detection
Object Detection using YOLOv8 model with TensorRT

# 🚀 Real-Time Object Detection using YOLOv8 + TensorRT + CUDA GPU

This project demonstrates **real-time object detection** using the **YOLOv8 model optimized with TensorRT (INT8)** for **GPU acceleration**.  
It captures video frames from a **webcam**, runs inference on a **CUDA-enabled GPU**, and displays **detected objects** (like people, cars, laptops, etc.) in real time.

---

## 🧠 Features

- ⚡ High-speed inference using **TensorRT INT8 engine**
- 🎥 Real-time webcam feed detection
- 🧩 Customizable detection classes (e.g., person, car, laptop)
- 🧠 Fully utilizes **CUDA GPU acceleration**
- 🖼️ Live bounding box annotations with class labels and confidence scores

---

## 🛠️ Requirements

Before running the script, make sure you have the following installed:

- **Python 3.8+**
- **CUDA Toolkit** (compatible with your GPU)
- **cuDNN**
- **PyTorch** with CUDA support
- **Ultralytics YOLOv8**
- **OpenCV**

### Install Dependencies

```bash
pip install ultralytics torch torchvision torchaudio opencv-python
```

> ⚠️ Make sure your PyTorch installation supports **CUDA**.  
> You can check with:
> ```python
> import torch
> print(torch.cuda.is_available())
> ```

---

## 🚀 How to Run

1. **Clone this repository** (or copy the script into your working directory).

2. **Download or export the YOLOv8 TensorRT engine file**  
   You can export a YOLOv8 model to TensorRT using:
   ```bash
   yolo export model=yolov8m.pt format=engine int8=True
   ```
   This will create a file like `yolov8m.engine`.

3. **Run the script:**
   ```bash
   python detect_trt_yolov8.py
   ```

4. **Press `q` to exit** the live detection window.

---

## 📄 Code Overview

### Main Components

- **GPU Check:**
  Ensures CUDA GPU is available before running inference.

- **Class Selection:**
  You can modify the list:
  ```python
  classes_to_detect = ["person", "car", "laptop", "cell phone"]
  ```
  Only these objects will be detected and annotated.

- **Model Loading:**
  ```python
  model = YOLO("yolov8m.engine")
  ```
  Uses the **TensorRT-optimized YOLOv8m** model.

- **Inference Loop:**
  Captures frames from the webcam, resizes them, runs GPU inference, and overlays detections.

- **Visualization:**
  Draws bounding boxes and labels for detected objects.

---

## ⚙️ Customization

- **Change Camera Source**
  ```python
  cap = cv2.VideoCapture(1)  # use external USB camera
  ```

- **Modify Detection Classes**
  Change `classes_to_detect` to include or remove objects.

- **Model Variant**
  Replace `yolov8m.engine` with another TensorRT model like `yolov8n.engine` (smaller, faster) or `yolov8l.engine` (larger, more accurate).

---

## 🧩 Example Output

When running, you’ll see a real-time webcam window like this:

```
YOLOv8m INT8 GPU Detection
---------------------------
✅ Using GPU: NVIDIA GeForce RTX 3060
Detected: person 0.92, laptop 0.87
```

Objects in the frame will be highlighted with **bounding boxes and labels**.

---

## ⚠️ Troubleshooting

- **Error: GPU not available**
  > Make sure your system has a CUDA-compatible GPU and the drivers are properly installed.

- **Cannot open webcam**
  > Check your webcam index (`cv2.VideoCapture(0)`) or external camera permissions.

- **Slow performance**
  > Use a smaller model (e.g., `yolov8n.engine`) or reduce resolution.

---

##Important Files
Engine file : https://drive.google.com/file/d/1nWTHc-usCowljAso37nvB345YGxOq1Wx/view?usp=sharing

ONNX file : https://drive.google.com/file/d/192C1-Cudaslm_bHmEUn3CGSjE16Drh3m/view?usp=sharing

Pytorch file : https://drive.google.com/file/d/1DR1lX9zFOgvizskgidPqRJ0U7W2VTk69/view?usp=sharing

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)

---

## 🧑‍💻 Author

Mahesh Kachave  
📧 maheshk22310389@gmail.com  
💡 Passionate about computer vision, AI optimization, and embedded systems.

---
