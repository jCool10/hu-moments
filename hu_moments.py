import cv2
import numpy as np
import matplotlib.pyplot as plt

ot = cv2.imread('images/ot.jpg')
cachua = cv2.imread('images/cachua.jpg')
cachua2 = cv2.imread('images/cachua2.jpg')
ot2 = cv2.imread('images/ot2.jpg')



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


ot_humoments = hu_moments(ot)
cachua_humoments = hu_moments(cachua)
cachua2_humoments = hu_moments(cachua2)
ot2_humoments = hu_moments(ot2)



ot_cachua2_dis = sum([abs(ot_humoments[i] - cachua2_humoments[i]) for i in range(7)])
cachua_cachua2_dis = sum([abs(cachua_humoments[i] - cachua2_humoments[i]) for i in range(7)])
ot_ot2_dis = sum([abs(ot_humoments[i] - ot2_humoments[i]) for i in range(7)])
cachua_ot2_dis = sum([abs(cachua_humoments[i] - ot2_humoments[i]) for i in range(7)])

print("Train")
print("Ot:", ot_humoments)
print("Ca chua:",cachua_humoments)

print("Test")
print("Ot vs Ca chua 2:",ot_cachua2_dis)
print("Ca chua vs Ca chua 2:",cachua_cachua2_dis)
print("Ot vs Ot 2:",ot_ot2_dis)
print("Ca chua vs Ot 2:",cachua_ot2_dis)

if(ot_ot2_dis<ot_cachua2_dis):
    print("Ot")
