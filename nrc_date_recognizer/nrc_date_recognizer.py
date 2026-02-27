import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from tensorflow.keras.models import load_model

detection_model = YOLO("trained_models/detection_model_v6/burmese_nrc_date_detector_v6.pt")
year_recognizer = load_model("trained_models/year_model/year_predictor.keras")
day_month_recognizer = load_model("trained_models/day_month_model/day_month_model_v2.keras")

burmese_digit = { 0: "၀", 1: "၁", 2: "၂", 3: "၃", 4: "၄", 5: "၅", 6: "၆", 7: "၇", 8: "၈", 9: "၉"}

class_names = ["၀", "၁", "၂", "၃", "၄", "၅", "၆", "၇", "၈", "၉", "၁၀", "၁၁", "၁၂", "၁၃", "၁၄",
               "၁၅", "၁၆", "၁၇", "၁၈", "၁၉", "၂၀", "၂၁", "၂၂", "၂၃", "၂၄", "၂၅", "၂၆", "၂၇", "၂၈",
               "၂၉", "၃၀", "၃၁"]

def predict_burmese_nrc_date(image):
    ind_p1, ind_p2, day_p1, day_p2, mon_p1, mon_p2, year_p1, year_p2 = get_detection_coordinates(image)
    d_x1, d_y1 = day_p1
    d_x2, d_y2 = day_p2
    m_x1, m_y1 = mon_p1
    m_x2, m_y2 = mon_p2
    y_x1, y_y1 = year_p1
    y_x2, y_y2 = year_p2
    day, month = predict_day_month(image, d_x1, d_y1, d_x2, d_y2, m_x1, m_y1, m_x2, m_y2)
    year = predict_year(image, y_x1, y_y1, y_x2, y_y2) 

    date_img = cv2.imread(image, cv2.IMREAD_COLOR_RGB)
    date_img = cv2.rectangle(date_img, pt1=day_p1, pt2=day_p2, color=(0, 255, 0), thickness=4)
    date_img = cv2.rectangle(date_img, pt1=mon_p1, pt2=mon_p2, color=(0, 255, 0), thickness=4)
    date_img = cv2.rectangle(date_img, pt1=year_p1, pt2=year_p2, color=(0, 255, 0), thickness=4)
    plt.imshow(date_img)

    print(f"Prediction: {day}.{month}.{year}")

def get_detection_coordinates(date_image):

    if isinstance(date_image, str):
        if not os.path.exists(date_image):
            raise ValueError("Image path not found")
        image = cv2.imread(date_image)
    else:
        image = date_image

    original_h, original_w = image.shape[:2]
    inference_img = cv2.resize(image, (512, 512))
    detected_result = detection_model.predict(
        source=inference_img,
        conf=0.1,
        imgsz=512,
        verbose=False
    )

    parts = []

    scale_x = original_w / 512
    scale_y = original_h / 512

    ind_p1, ind_p2 = (0, 0), (0, 0)

    for r in detected_result:
        for box in r.boxes:
            cords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, cords)
            label = r.names[int(box.cls[0])]
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            if label == "indicator":
                ind_p1, ind_p2 = (x1, y1), (x2, y2)
            else:
                parts.append({
                    "p1": (x1, y1),
                    "p2": (x2, y2),
                    "x_center": (x1 + x2) / 2
                })

    sorted_parts = sorted(parts, key=lambda x: x["x_center"])

    day_p1, day_p2 = (0, 0), (0, 0)
    mon_p1, mon_p2 = (0, 0), (0, 0)
    year_p1, year_p2 = (0, 0), (0, 0)

    if len(sorted_parts) >= 3:
        day_p1, day_p2 = sorted_parts[0]["p1"], sorted_parts[0]["p2"]
        mon_p1, mon_p2 = sorted_parts[1]["p1"], sorted_parts[1]["p2"]
        year_p1, year_p2 = sorted_parts[2]["p1"], sorted_parts[2]["p2"]

    return ind_p1, ind_p2, day_p1, day_p2, mon_p1, mon_p2, year_p1, year_p2

def predict_day_month(image_source, d_x1, d_y1, d_x2, d_y2, m_x1, m_y1, m_x2, m_y2):
    if isinstance(image_source, str):
        image = cv2.imread(image_source, cv2.IMREAD_COLOR_RGB)
    else:
        image = image_source

    h, w = image.shape[:2]
    pad = 0
    
    def process_and_predict(x1, y1, x2, y2):
        
        x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
        x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
        
        crop = image[y1_p:y2_p, x1_p:x2_p]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        resized = cv2.resize(thresh, (64, 64))
        rgb_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        img_array = tf.keras.preprocessing.image.img_to_array(rgb_input)
        normalized = img_array / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        pred = day_month_recognizer.predict(input_tensor, verbose=0)
        return class_names[np.argmax(pred)]

    day_result = process_and_predict(d_x1, d_y1, d_x2, d_y2)
    month_result = process_and_predict(m_x1, m_y1, m_x2, m_y2)

    return day_result, month_result

def predict_year(image_source, x1, y1, x2, y2):
    if isinstance(image_source, str):
        original_img = cv2.imread(image_source)
    else:
        original_img = image_source

    if original_img is None:
        raise ValueError("Image not found or empty!")

    h, w = original_img.shape[:2]
    pad = 0
    
    x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
    x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
    year_crop = original_img[y1_p:y2_p, x1_p:x2_p]

    gray = cv2.cvtColor(year_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    resized = cv2.resize(thresh, (128, 32))
    normalized = resized.astype("float32") / 255.0
    
    input_tensor = np.expand_dims(normalized, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    preds = year_recognizer.predict(input_tensor, verbose=0)

    d1 = int(np.argmax(preds[0], axis=1)[0])
    d2 = int(np.argmax(preds[1], axis=1)[0])
    d3 = int(np.argmax(preds[2], axis=1)[0])
    d4 = int(np.argmax(preds[3], axis=1)[0])

    burmese_year = "".join([burmese_digit[d] for d in [d1, d2, d3, d4]])

    return burmese_year
