import numpy as np
from matplotlib.pyplot import *
import lz4.frame
import lz4.block as lb
import cv2 as cv
import pandas as pd
import seaborn as sns


def convertTonum(data):
    tmp = bytearray(data)
    ret = int.from_bytes(tmp,byteorder="big", signed=False)
    return ret


def transformCoordinate(x_o, y_o,resolution, points):
    pt = {}
    pt['x'] = (points[0]+x_o)*resolution
    pt['y'] = (points[1]+y_o)*resolution
    return pt

    
#b = np.fromfile('/home/lixq/work/tmp/tymap.bin',dtype=np.uint8,offset=24)
b = np.fromfile('/home/lixq/work/tmp/tymap.bin',dtype=np.uint8)
map_width = convertTonum(b[4:6])
map_height = convertTonum(b[6:8])
#print(int.from_bytes(q,byteorder="big", signed=False))
before_compress = b[19:22]
q = bytearray(before_compress)
before_compress = int.from_bytes(q,byteorder="big", signed=False)
after_compress = b[22:24]
q = bytearray(after_compress)
after_compress = int.from_bytes(q,byteorder="big", signed=False)
map_ox = convertTonum(b[8:10])
map_oy = convertTonum(b[10:12])
map_resolution = convertTonum(b[12:14])

print("width:, height:,before_compress:,after_compress",map_width,map_height,before_compress,after_compress)
bq = b[24:]
print(len(bq))
print("width*height:",map_width*map_height)
decompressed = lb.decompress(bq,before_compress)
print(type(decompressed))
print(len(decompressed))
#print(decompressed[:3])
data = []
for i in decompressed:
    data.append(i)
#print(data)
np.savetxt("new3.csv",data,delimiter=',')
#print(b)
#np.savetxt("new1.csv",b&0b11,delimiter=',')
#dec = b[0]
#print("b:",dec)
#print("二进制b:",bin(dec))
#print("16进制b:",hex(dec))
#print(dec & 0b11)
c = bq&0b11
data1 = []
data0 = []
data3 = []
for i in c:
    if (i == 1):
        data1.append(i)
    if (i == 0):
        data0.append(i)
    if (i == 3):
        data3.append(i)

#data.reshape(map_width[0],map_height[0])
pix = np.array(data[:map_width*map_height])
#pixs = pix.reshape(map_height[0],map_width[0])
pixs = np.zeros((map_height,map_width))
#print(np.shape(pixs))
regionId = []
for r in range(map_height):
    index_pre = r*map_width
    for col in range(map_width):
        id = index_pre + col
        pixs[r][col] = pix[id]
        if (np.uint8(pixs[r][col]) >> np.uint8(2)) < (32):
            pixs[r][col] = 777


#pixs3 = pixs.transpose()
#print(np.shape(pixs))
np.savetxt("new1.csv",pixs,delimiter=',')

"""
d = np.fromfile('/home/lixq/work/tmp/tymapbk.bin',dtype=np.uint8)
ori = d[:7280]
origin_data = (ori.reshape(map_width[0],map_height[0])).transpose()
np.savetxt("new6.csv",origin_data,delimiter=',')
print((origin_data == pixs3).all())
#print(type(d))
"""



"""
index = []
pixs = [[]]
for i in range(map_width[0]):
    for j in range(map_height[0]):
        pixs[i][j] = data[i*map_width+j]
        index.append(i*map_width+j)
"""

#image = np.zeros((map_resolution[0],map_height[0],3),np.uint8)
#red_color = (0,0,255)
#cv.fillPoly(image,[pixs],red_color)
#cv.imshow('Result',image)
#cv.waitKey(0)

#figure(num=1, figsize=(map_width[0],map_height[0]))
#imshow(pixs)
#xlim((0,map_width[0]))
#ylim((0,map_height[0]))

#show()

src = pixs
cvImg = np.zeros((map_height,map_width,1), np.uint8)

#print(map_ox,map_oy)
#q = bytearray(x)
#print(int.from_bytes(q,byteorder="big", signed=False))