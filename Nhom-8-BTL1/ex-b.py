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

def scan_image(template_thresh, img2, scale, original_height, original_width, step=5, threshold_scale=40, color=(0, 255, 0)):
    new_width = int(img2.shape[1] * scale)
    new_height = int(img2.shape[0] * scale)
    resized_img2 = preprocess_image(cv2.resize(img2, (new_width, new_height)), threshold=230)
    img2_copy = img2.copy()

    for i in range(0, resized_img2.shape[0] - original_height + 1, step):
        for j in range(0, resized_img2.shape[1] - original_width + 1, step):
            temp = resized_img2[i:i + original_height, j:j + original_width]
            if compare_grids(template_thresh, temp, threshold_scale=threshold_scale):
                cv2.rectangle(img2_copy, (int(j / scale), int(i / scale)),
                              (int((j + original_width) / scale), int((i + original_height) / scale)),
                              color, 2)

    return img2_copy

def find_objects(template, img2, scales, color=(0, 255, 0)):
    template_thresh = preprocess_image(template, threshold=200)
    original_height, original_width = template_thresh.shape

    for scale in scales:
        img2 = scan_image(template_thresh, img2, scale, original_height, original_width, color=color)

    return img2

# Load and preprocess images
cachua = cv2.imread('images/tomato.png')
ot = cv2.imread('images/ot.png')
img2 = cv2.imread('images/find2.png')

# Create scales by for loop
scales = np.arange(0.5, 1.5, 0.05)

# Find oranges
result_img = find_objects(ot, img2, scales, color=(0, 0, 255))

# Find bananas on the same image
result_img = find_objects(cachua, result_img, scales, color=(0, 255, 0))

cv2.imshow('Detected Objects', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
