import cv2
import numpy as np

ot = cv2.imread('images/ot.jpg')
cachua = cv2.imread('images/cachua.jpg')
cachua2 = cv2.imread('images/cachua2.jpg')
ot2 = cv2.imread('images/ot2.jpg')


def hu_moments(img):
    # convert img to gray and resize to 128x128
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    x, y = img.shape

    x_ = np.sum(np.multiply(np.arange(1, x+1), img))
    y_ = np.sum(np.multiply(np.arange(1, y+1), img))
    sum_ = np.sum(img)

    x_ = x_/sum_
    y_ = y_/sum_

    def mpq(p, q):
        return np.sum(np.multiply(np.power(np.arange(1, x+1) - x_, p), np.power(np.arange(1, y+1) - y_, q), img))

    def Mpq(p, q):
        return mpq(p, q)/(mpq(0, 0)**((p+q)/2+1))

    S1 = Mpq(2, 0) + Mpq(0, 2)
    S2 = (Mpq(2, 0) - Mpq(0, 2))**2 + 4*Mpq(1, 1)**2
    S3 = (Mpq(3, 0) - 3*Mpq(1, 2))**2 + (3*Mpq(2, 1) - Mpq(0, 3))**2
    S4 = (Mpq(3, 0) + Mpq(1, 2))**2 + (Mpq(2, 1) + Mpq(0, 3))**2
    S5 = (Mpq(3, 0) - 3*Mpq(1, 2))*(Mpq(3, 0) + Mpq(1, 2))*((Mpq(3, 0) + Mpq(1, 2))**2 - 3*(Mpq(2, 1) + Mpq(0, 3))**2) + (3*Mpq(2, 1) - Mpq(0, 3))*(Mpq(2, 1) + Mpq(0, 3))*(3*(Mpq(3, 0) + Mpq(1, 2))**2 - (Mpq(2, 1) + Mpq(0, 3))**2)
    S6 = (Mpq(2, 0) - Mpq(0, 2))*((Mpq(3, 0) + Mpq(1, 2))**2 - (Mpq(2, 1) + Mpq(0, 3))**2) + 4*Mpq(1, 1)*(Mpq(3, 0) + Mpq(1, 2))*(Mpq(2, 1) + Mpq(0, 3))
    S7 = (3*Mpq(2, 1) - Mpq(0, 3))*(Mpq(3, 0) + Mpq(1, 2))*((Mpq(3, 0) + Mpq(1, 2))**2 - 3*(Mpq(2, 1) + Mpq(0, 3))**2) + (3*Mpq(1, 2) - Mpq(3, 0))*(Mpq(2, 1) + Mpq(0, 3))*(3*(Mpq(3, 0) + Mpq(1, 2))**2 - (Mpq(2, 1) + Mpq(0, 3))**2)

    return S1, S2, S3, S4, S5, S6, S7


ot_humoments = hu_moments(ot)
cachua_humoments = hu_moments(cachua)
cachua2_humoments = hu_moments(cachua2)
ot2_humoments = hu_moments(ot2)


def manhattan_distance(humoments1, humoments2):
    return np.sum(np.abs(humoments1 - humoments2))


def euclidean_distance(humoments1, humoments2):
    return np.sqrt(np.sum(np.square(humoments1 - humoments2)))


ot_cachua2_dis = manhattan_distance(ot_humoments, cachua2_humoments)
cachua_cachua2_dis = manhattan_distance(cachua_humoments, cachua2_humoments)
ot_ot2_dis = manhattan_distance(ot_humoments, ot2_humoments)
cachua_ot2_dis = manhattan_distance(cachua_humoments, ot2_humoments)

print("Train")
print("Ot:", ot_humoments)
print("Ca chua:", cachua_humoments)

print("Test")
print("Ot vs Ca chua 2:", ot_cachua2_dis)
print("Ca chua vs Ca chua 2:", cachua_cachua2_dis)
print("Ot vs Ot 2:", ot_ot2_dis)
print("Ca chua vs Ot 2:", cachua_ot2_dis)

if ot_ot2_dis < ot_cachua2_dis:
    print("Ot")
