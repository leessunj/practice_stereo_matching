# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import cv2

def cv_basic(image):
    print(image.shape) #height, width, num_color_channel
    print(image.size) #h*w*c

    px = image[3, 3]
    print(px)
    #print(px[2])

def show_img(image):
    cv2.imshow('test', image)
    cv2.waitKey(0)  # 보여주자마자 꺼지지 않게. 입력받으면 꺼진다
    cv2.imwrite('result.png', image)
    cv2.destroyAllWindows()

def show_grayscale(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', img_gray)
    cv2.waitKey(0)

def change_img_px(image):
    for i in range(200,500):
        for j in range(200,400):
            image[i,j][2]=min(0,image[i,j][2]-50)

    image[:100,:100]=[0,0,0] #이 방법이 훨씬 빠르다
    return image

def roi(image):
    interest=image[300:500,300:500]
    image[:200,:200]=interest
    return image

def img_resize(image):
    expand = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    shrink = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

def transformation(image):
    # image[:,:,2]=0 #R값 색상만 다 0으로 바꾸기
    h, w = image.shape[:2]
    # M=np.float32([[1,0,100],[0,1,100]])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 60, 1)
    dst = cv2.warpAffine(image, M, (w, h))



if __name__ == '__main__':
    im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0)
    im2 = cv2.imread('tsukuba/scene1.row3.col3.ppm',0)
    optimal_result=cv2.imread('tsukuba/truedisp.row3.col3.pgm')
    #im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    h, w = im.shape #288 384
    d=[]
    for i in range(h):
        for j in range(w):
            candidates=[255]
            for k in range(1,16):
                if j+k>=w:
                    break
                candidates.append(abs(im[i][j+k]-im2[i][j]))
            d.append(min(candidates))

    disparity=np.array(d)
    disparity.shape=(h,w)
#    matcher = cv2.StereoBM_create(numDisparities=16, blockSize=9)
#    disparity=matcher.compute(im,im2)
    #show_img(disparity)
#    print(disparity)

#    print(disparity[144,192],h,w)
    plt.imshow(disparity,'gray')
    plt.show()

    #cv2.StereoBM.create


