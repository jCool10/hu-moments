import cv2
import numpy as np

def preprocess_image(image, threshold=200):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh / 255  # Chuẩn hóa để giảm thiểu sai số số học

def compare_grids(img1, img2, threshold_scale=40):
    diff = np.sum(np.abs(img1.astype(int) - img2.astype(int)))
    threshold = threshold_scale * img1.size / 255
    return diff < threshold

def scan_image(template_thresh, img2, scale, original_height, original_width, step=5, threshold_scale=40):
    new_width = int(img2.shape[1] * scale)
    new_height = int(img2.shape[0] * scale)
    resized_img2 = preprocess_image(cv2.resize(img2, (new_width, new_height)), threshold=230)
    detected_points = []

    for i in range(0, resized_img2.shape[0] - original_height + 1, step):
        for j in range(0, resized_img2.shape[1] - original_width + 1, step):
            temp = resized_img2[i:i + original_height, j:j + original_width]
            if compare_grids(template_thresh, temp, threshold_scale=threshold_scale):
                detected_points.append((j, i, scale))

    return detected_points

def find_objects(template, img2, scales):
    template_thresh = preprocess_image(template, threshold=200)
    original_height, original_width = template_thresh.shape
    img2_copy = img2.copy()
    
    for scale in scales:
        points = scan_image(template_thresh, img2, scale, original_height, original_width)
        for (x, y, scale) in points:
            cv2.rectangle(img2_copy, (int(x / scale), int(y / scale)),
                          (int((x + original_width) / scale), int((y + original_height) / scale)),
                          (0, 255, 0), 2)

    return img2_copy

img = cv2.imread('images/tomato.png')
img2 = cv2.imread('images/find.png')
scales = np.arange(0.5, 1.5, 0.1)

result_img = find_objects(img, img2, scales)

cv2.imshow('Detected Objects', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

