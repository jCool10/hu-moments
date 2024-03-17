import cv2
import numpy as np
import matplotlib.pyplot as plt

def hu_moments(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to 128x128
    img = cv2.resize(img, (128, 128))

    # Get the dimensions of the image
    x, y = img.shape

    # Calculate the centroid of the image
    x_ = 0
    y_ = 0
    sum = 0

    for i in range(x):
        for j in range(y):
            x_ += (i+1) * img[i][j]
            y_ += (j+1) * img[i][j]
            sum += img[i][j]

    x_ = x_/sum
    y_ = y_/sum

    # Define the mpq and Mpq functions
    def mpq(p, q):
        sum = 0
        for i in range(x):
            for j in range(y):
                sum += (i+1-x_)**p * (j+1-y_)**q * img[i][j]
        return sum 
    
    def Mpq(p, q):
        return mpq(p, q)/(mpq(0, 0)**((p+q)/2+1))

    # Calculate the 7 Hu moments
    S1 = Mpq(2, 0) + Mpq(0, 2)
    S2 = (Mpq(2, 0) - Mpq(0, 2))**2 + 4*Mpq(1, 1)**2
    S3 = (Mpq(3, 0) - 3*Mpq(1, 2))**2 + (3*Mpq(2, 1) - Mpq(0, 3))**2
    S4 = (Mpq(3, 0) + Mpq(1, 2))**2 + (Mpq(2, 1) + Mpq(0, 3))**2
    S5 = (Mpq(3, 0) - 3*Mpq(1, 2))*(Mpq(3, 0) + Mpq(1, 2))*((Mpq(3, 0) + Mpq(1, 2))**2 - 3*(Mpq(2, 1) + Mpq(0, 3))**2) + (3*Mpq(2, 1) - Mpq(0, 3))*(Mpq(2, 1) + Mpq(0, 3))*(3*(Mpq(3, 0) + Mpq(1, 2))**2 - (Mpq(2, 1) + Mpq(0, 3))**2)
    S6 = (Mpq(2, 0) - Mpq(0, 2))*((Mpq(3, 0) + Mpq(1, 2))**2 - (Mpq(2, 1) + Mpq(0, 3))**2) + 4*Mpq(1, 1)*(Mpq(3, 0) + Mpq(1, 2))*(Mpq(2, 1) + Mpq(0, 3))
    S7 = (3*Mpq(2, 1) - Mpq(0, 3))*(Mpq(3, 0) + Mpq(1, 2))*((Mpq(3, 0) + Mpq(1, 2))**2 - 3*(Mpq(2, 1) + Mpq(0, 3))**2) + (3*Mpq(1, 2) - Mpq(3, 0))*(Mpq(2, 1) + Mpq(0, 3))*(3*(Mpq(3, 0) + Mpq(1, 2))**2 - (Mpq(2, 1) + Mpq(0, 3))**2)

    return S1, S2, S3, S4, S5, S6, S7

# Load the images
chilli_train = cv2.imread('images/ot.jpg')
tomato_train= cv2.imread('images/cachua.jpg')
tomato_test= cv2.imread('images/cachua2.jpg')
chilli_test = cv2.imread('images/ot2.jpg')

# Calculate the Hu moments
chilli_train_hu_moments = hu_moments(chilli_train)
tomato_train_hu_moments = hu_moments(tomato_train)
tomato_test_hu_moments = hu_moments(tomato_test)
chilli_test_hu_moments = hu_moments(chilli_test)

# print("chilli_train_hu_moments",chilli_train_hu_moments)
# print("tomato_train_hu_moments",tomato_train_hu_moments)
# print("tomato_test_hu_moments",tomato_test_hu_moments)
# print("chilli_test_hu_moments",chilli_test_hu_moments)


def manhattan_distance(humoments1, humoments2):
    return np.sum(np.abs(np.array(humoments1) - np.array(humoments2)))

def euclidean_distance(humoments1, humoments2):
    return np.sqrt(np.sum(np.square(np.array(humoments1) - np.array(humoments2))))

# template_matching can choice manhattan_distance or euclidean_distance
def template_matching(humoments, humoments_list, threshold, distance_type = 'manhattan'):
    if distance_type == 'manhattan':
        distance = manhattan_distance
    else:
        distance = euclidean_distance
    min_distance = distance(humoments, humoments_list[0])
    min_index = 0
    for i in range(1, len(humoments_list)):
        d = distance(humoments, humoments_list[i])
        if d < min_distance:
            min_distance = d
            min_index = i
    if min_distance < threshold:
        return min_index
    return -1

# Compare the Hu moments of the test images with the Hu moments of the training images
result_manhattan_distance = template_matching(chilli_test_hu_moments, [chilli_train_hu_moments, tomato_train_hu_moments], 0.1)
if result_manhattan_distance == 0:
    print('manhattan_distance: Chilli')
else:
    print('manhattan_distance: Tomato')

result_euclidean_distance = template_matching(tomato_test_hu_moments, [chilli_train_hu_moments, tomato_train_hu_moments], 0.1, 'euclidean')
if result_euclidean_distance == 0:
    print('euclidean_distance: Chilli')
else:
    print('euclidean_distance: Tomato')