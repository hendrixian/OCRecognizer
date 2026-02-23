import base64
import numpy as np
import cv2


def decode_base64_image(data_url):
    if not data_url:
        raise ValueError('Missing image data')

    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(data_url)
    except Exception as exc:
        raise ValueError('Invalid base64 image data') from exc

    return decode_image_bytes(img_bytes)


def decode_image_bytes(img_bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Failed to decode image')
    return image
