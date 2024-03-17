import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lập trình tính đặc trưng Hu's moments cho ảnh nhị phân, với đối tượng là màu trắng trên nền đen.
def hu_moments(img):
    # convert img to gray and resize to 128x128
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    x, y = img.shape

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

    def mpq(p,q):
        sum = 0
        for i in range(x):
            for j in range(y):
                sum += (i+1-x_)**p * (j+1-y_)**q * img[i][j]
        return sum 
    
    def Mpq(p,q):
        return mpq(p,q)/(mpq(0,0)**((p+q)/2+1))

    S1 = Mpq(2,0) + Mpq(0,2)
    S2 = (Mpq(2,0) - Mpq(0,2))**2 + 4*Mpq(1,1)**2
    S3 = (Mpq(3,0) - 3*Mpq(1,2))**2 + (3*Mpq(2,1) - Mpq(0,3))**2
    S4 = (Mpq(3,0) + Mpq(1,2))**2 + (Mpq(2,1) + Mpq(0,3))**2
    S5 = (Mpq(3,0) - 3*Mpq(1,2))*(Mpq(3,0) + Mpq(1,2))*((Mpq(3,0) + Mpq(1,2))**2 - 3*(Mpq(2,1) + Mpq(0,3))**2) + (3*Mpq(2,1) - Mpq(0,3))*(Mpq(2,1) + Mpq(0,3))*(3*(Mpq(3,0) + Mpq(1,2))**2 - (Mpq(2,1) + Mpq(0,3))**2)
    S6 = (Mpq(2,0) - Mpq(0,2))*((Mpq(3,0) + Mpq(1,2))**2 - (Mpq(2,1) + Mpq(0,3))**2) + 4*Mpq(1,1)*(Mpq(3,0) + Mpq(1,2))*(Mpq(2,1) + Mpq(0,3))
    S7 = (3*Mpq(2,1) - Mpq(0,3))*(Mpq(3,0) + Mpq(1,2))*((Mpq(3,0) + Mpq(1,2))**2 - 3*(Mpq(2,1) + Mpq(0,3))**2) + (3*Mpq(1,2) - Mpq(3,0))*(Mpq(2,1) + Mpq(0,3))*(3*(Mpq(3,0) + Mpq(1,2))**2 - (Mpq(2,1) + Mpq(0,3))**2)

    return S1, S2, S3, S4, S5, S6, S7

# Đọc ảnh và chuyển sang ảnh nhị phân.

chilli_train = cv2.imread('images/ot.jpg')
tomato_train= cv2.imread('images/cachua.jpg')
tomato_test= cv2.imread('images/cachua2.jpg')
chilli_test = cv2.imread('images/ot2.jpg')

# Áp dụng code trên để tính Hu's moments cho 2 ảnh bất kỳ thuộc 2 class trong tập dữ liệu đã phân công cho nhóm.
chilli_train_hu_moments = hu_moments(chilli_train)
tomato_train_hu_moments = hu_moments(tomato_train)
tomato_test_hu_moments = hu_moments(tomato_test)
chilli_test_hu_moments = hu_moments(chilli_test)


def manhattan_distance(humoments1, humoments2):
    return np.sum(np.abs(np.array(humoments1) - np.array(humoments2)))

def euclidean_distance(humoments1, humoments2):
    return np.sqrt(np.sum(np.square(np.array(humoments1) - np.array(humoments2))))

# Áp dụng phương pháp template matching để tiến hành nhận dạng đối tượng. 
def template_matching(humoments, humoments_list, threshold):
    min_distance = manhattan_distance(humoments, humoments_list[0])
    min_index = 0
    for i in range(1, len(humoments_list)):
        distance = manhattan_distance(humoments, humoments_list[i])
        if distance < min_distance:
            min_distance = distance
            min_index = i
    if min_distance < threshold:
        return min_index
    return -1

# Nhận dạng ảnh 
threshold = 0.1
result = template_matching(chilli_test_hu_moments, [tomato_train_hu_moments, chilli_train_hu_moments], threshold)

if result == 0:
    print("Cà chua")
else:
    print("Ớt")