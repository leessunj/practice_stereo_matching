import numpy as np
import matplotlib.pyplot as plt
import cv2

im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0)  # left
im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0)  # right
igt = cv2.imread('tsukuba/truedisp.row3.col3.pgm',0)

P1=5
P2=150
maxd=15

a=[1,2,3]
print(np.subtract(a,1))

# r=(1,1)
# p=(3,4)
# print(r+p)
# r=(-r[0],-r[1])
# print(r)
#matching cost
# MI
 #AD
def dis_AD(ib,im,maxd=13):
    h, w = im.shape  # 288 384
    d = []
    for i in range(h):
        for j in range(w):
            mdiff=255
            midx=-1
            for k in range(-maxd, maxd+1):
                if j + k < 0:
                    continue
                elif j + k >= w:
                    break
                diff=abs(int(ib[i][j])-int(im[i][j + k]))
                if diff<mdiff:
                    mdiff=diff
                    midx=k
                d.append(midx)
    disparity = np.array(d)
    disparity.shape = (h, w)
    return disparity #/ np.max(disparity)

def cost_AD(p,d,w=384):
    global im,im2
    if p[1]-d<0 or p[1]-d>=w:
        return 255
    return abs(int(im[p[0]][p[1]])-int(im2[p[0]][p[1] - d]))

def costvolume(maxd=15):
    global im,im2
    h, w = im.shape  # 288 384
    disparity = []
    for i in range(h):
        for j in range(w):
            mdiff = 255
            midx = -1
            d=[]
            for k in range(maxd):
                if j-k<0:
                    d.append(255)
                else:
                    d.append(abs(int(im[i][j]) - int(im2[i][j - k])))
            disparity+=d
    disparity = np.array(disparity)
    disparity.shape = (h, w,maxd)

    # d=[]
    # for i in range(h):
    #     for j in range(w):
    #         d.append(np.argmin(disparity[i,j,:]))
    # d = np.array(d)
    # d.shape = (h, w)
    # cv2.imshow('cv',d/np.max(d))
    # cv2.waitKey(0)

    return disparity  # / np.max(disparity)




def get_direction_pairs(p,r):
    dir_range = []
    y, x = p
    while 0 < y < h - 1 and 0 < x < w - 1:
        dir_range.append(((y, x)))
        y += r[0]
        x += r[1]
    return (y,x),dir_range[::-1]


def get_lr(cv, p, r): #d별 lr
    global maxd
    prev_i,dir_range = get_direction_pairs(p, r)
    prev = cv[prev_i[0],prev_i[1],:]
    for e in dir_range:
        cur=cv[e[0],e[1],:].astype('int64')
        ub=min(prev)
        cur[0]+=min(prev[1]+P1,ub+P2)
        for i in range(1,maxd-1):
            cur[i]+=min(prev[i],prev[i-1]+P1,prev[i+1]+P1,ub+P2)
        cur[-1]+=min(prev[-2]+P1,ub+P2)
        prev=np.subtract(cur,ub)
    return prev

cv=costvolume(maxd)

def agg_cost_disp(p): #s를 통해 구한 d
    global cv
    directions=[(0,1),(1,0),(1,1),(-1,1),]#(1,2),(2,1),(-2,1),(-1,2)]
    s=np.zeros(maxd)
    for r in directions:
        s+=get_lr(cv, p, r)
        r = (-r[0], -r[1])
        s+=get_lr(cv, p, r)
    return np.argmin(s)

h,w=im.shape
disparity=[]
for i in range(h):
    for j in range(w):
        disparity.append(agg_cost_disp((i,j)))
        print(i,j)
disparity=np.array(disparity)
disparity.shape=(h,w)
cv2.imshow("dis",disparity/np.max(disparity))
cv2.waitKey(0)



#cost aggregation

# dpr=dict()
# def cost_Lr(p,d,r):
#     global dpr,im,P1,P2
#     h,w=im.shape
#     if (p,d,r) in dpr:
#         return dpr[(p,d,r)]
#     elif p[0]==0 or p[0]==h-1 or p[1]==0 or p[1]==w-1:
#         return cost_AD(p,d)
#     if 0<=p[0]+r[0]<h and 0<=p[1]+r[1]<w and d>=0:
#         c1=cost_Lr((p[0]+r[0],p[1]+r[1]),d-1,r)+P1
#         c2=cost_Lr((p[0]+r[0],p[1]+r[1]),d+1,r)+P1
#         c3=cost_Lr((p[0]+r[0],p[1]+r[1]),d,r)
#         ub=cost_Lr((p[0]+r[0],p[1]+r[1]),255,r)
#         for i in range(1,255):
#             t=cost_Lr((p[0]+r[0],p[1]+r[1]),i,r)
#             if t<ub:
#                 ub=t
#         lr = cost_AD(p, d)+min(c1,c2,c3,ub+P2)-ub
#     else:
#         lr=cost_AD(p,d)
#     dpr[(p,d,r)]=lr
#     return lr
#
# def agg_cost(p,d):
#     global dpr
#     directions=[(0,1),(1,0),(1,1),(-1,1),]#(1,2),(2,1),(-2,1),(-1,2)]
#     s=0
#     for r in directions:
#         s+=cost_Lr(p,d,r)
#         # dpr.clear()
#         # print('--one dir clear--')
#         r = (-r[0], -r[1])
#         s+=cost_Lr(p,d,r)
#         # dpr.clear()
#         # print('--one dir clear--')
#     return s

#disparity computation
# def dis_compute(p):
#     s=agg_cost(p,1)
#     prop_d=1
#     for d in range(2,15):
#         t=agg_cost(p,d)
#         if t<s:
#             s=t
#             prop_d=d
#     return prop_d

# disparity=[]
# for i in range(h):
#     for j in range(w):
#         disparity.append(dis_compute((i,j)))
#         print(f'{i}, {j}')
#     dpr.clear()
#disparity refinement