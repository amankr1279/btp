import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#### Extraction from Mnist

df = pd.read_csv("train.csv")
#print(df.head(1))
mnist_img = [ [ 0 for i in range(28) ] for j in range(28) ] 

lst = df.loc[8,:]
img_name = '5' # can be changed for each digit
k=0
for i in range(28):
    for j in range(28):
        mnist_img[i][j] = lst[k]
        k = k+1


mnist_img = np.divide(mnist_img,255)
mnist_img = mnist_img.reshape(28,28,1)
mnist_img = np.multiply(mnist_img,255)
ret, bw_img = cv2.threshold(mnist_img,127,255,cv2.THRESH_BINARY)
cv2.namedWindow("output",cv2.WINDOW_NORMAL)
bw_img1 = cv2.resize(bw_img,(280,280))
#print(np.shape(bw_img))
cv2.imshow("Binary Image",bw_img1)

cv2.waitKey(1000)
cv2.destroyAllWindows

##### Spike Train generation

# Declaring rows 
N = 28

# Declaring columns 
M = 28

img = [ [ 0 for i in range(M) ] for j in range(N) ] 
#print(np.shape(img))
img = bw_img
scaled_down = np.array([ [ 0 for i in range(7) ] for j in range(7) ] )
#print(img)
def image_scaling(image, binary):
    (a1,b1) = np.shape(image)
    (a2,b2) = np.shape(binary)
    x1 = int(a1/a2)
    x2 = int(b1/b2)
    print(f"a = {a2} b = {b2}")
    for i in range(int(a2)):
        for j in range(int(b2)):
            r1 = x1*i
            r2 = x1*(i+1)
            c1 = x2*j
            c2 = x2*(j+1)
#            print(f"{r1} {r2} {c1} {c2}")
            g_avg = np.mean(image)
            x = image[r1:r2, c1:c2]# 4*4 sub matrix
            l_avg = np.mean(x)
            if l_avg > g_avg:
                binary[i][j] = 1


image_scaling(img, scaled_down)

spike_train = [0 for i in range(49)]

plt.imshow(scaled_down)
plt.show()

def spike_gen(binary_img, spike_train):
    k = 0
    for i in range(7):
        for j in range(7):
            if binary_img[i][j] == 1:
                spike_train[k] = 3 # 3 volts
            k = k + 1

spike_gen(scaled_down, spike_train)
print(spike_train)
spike_dict = []
"""spike_list = []
fields  =['0','voltage']"""

##### Writing data points in a text file
for i in range(49*3):
    x = i / 1000
    spike_dict.append( { '0': x, 'voltage' : spike_train[int(i/3)]} )
    voltage = spike_train[int(i/3)]
    print(f"{x:1.3f}  {voltage}",file = open("voltage_for_"+img_name+".txt","a"))
