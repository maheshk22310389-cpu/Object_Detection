import cv2                    #handles video input/output and drawing detections.
from ultralytics import YOLO  #interface to load and run YOLOv8 models.
import torch                  #checks CUDA GPU availability and manages device selection.
import time                   #used for timing inference if needed (though unused in your script)

# Ensure CUDA GPU is available
if not torch.cuda.is_available():
    raise SystemError("❌ GPU not available. Please enable a CUDA-supported GPU.")

# Select GPU automatically
device_index = 0  # use first GPU (cuda:0)
device = f"cuda:{device_index}"
print(f"✅ Using GPU: {torch.cuda.get_device_name(device_index)}")

# Classes to detect
classes_to_detect = [
    "person", "car", "bicycle", "sports ball", "motorbike",
    "book", "laptop", "cell phone", "bottle", "clock", "mouse"
]

# COCO class names
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# Get indices of selected classes
class_ids_to_detect = [COCO_CLASSES.index(cls) for cls in classes_to_detect]

# Load TensorRT engine model
model = YOLO("yolov8m.engine")  # INT8 TensorRT engine

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 640))

        # Run inference on GPU
        results = model.predict(source=frame_resized, device=device)

        # Annotate selected classes
        annotated_frame = frame_resized.copy()
        for box, cls_id, conf in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf
        ):
            cls_id = int(cls_id)
            if cls_id in class_ids_to_detect:
                x1, y1, x2, y2 = map(int, box)
                label = f"{COCO_CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        cv2.imshow("YOLOv8m INT8 GPU Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()