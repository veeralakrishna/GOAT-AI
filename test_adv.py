from ultralytics import YOLO, SAM
import numpy as np
import sys

def test():
    try:
        detect_model = YOLO("yolov8s-world.pt")
        detect_model.set_classes(["goat"])
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detect_model.predict(empty_frame)
        boxes = res[0].boxes.xyxy
        print("Detection boxes length:", len(boxes))
        
        seg_model = SAM("sam2_b.pt")
        # To test SAM with empty boxes might fail, let's inject a fake box
        fake_box = [[10, 10, 100, 100]]
        res_seg = seg_model(empty_frame, bboxes=fake_box)
        print("SAM returned masks:", res_seg[0].masks is not None)
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    test()
