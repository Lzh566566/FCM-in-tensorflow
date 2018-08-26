# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:35:06 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from PIL import Image
from skimage import transform,data
from FCM_tf import FCM
import time
#散点图绘制函数
import tensorflow as tf
data=io.imread(r"D:\0_CT.tif")
data=transform.resize(data,(512,512))
plt.imshow(data)
plt.show()
data=data.flatten().reshape(262144,1)
data=tf.convert_to_tensor(data,dtype=tf.float32)
start = time.time()
final_location = FCM(data , 5 , 2)
print(time.time()-start) 
final_location=np.array(final_location)
a=np.argmax(final_location,axis=1).reshape(512,512)
plt.imshow(a)
plt.show()

#plt.scatter(300,300,marker='o',color='b',label='0',s=10)

