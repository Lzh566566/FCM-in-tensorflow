# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:44:22 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 20:48:24 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:44:22 2018

@author: Administrator
"""

import nibabel as nib
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import copy

def cluster_center(data,U):
    '''
    data: tensor,shape=(number_dataset,number_features)
    U: tensor,shape=(number_dataset,number_cluster)
    ''' 
    U=tf.transpose(U, perm=[1, 0])
    
    numerator=tf.matmul(U,data)
    denumerator=tf.matmul(U,tf.ones(data.get_shape(), "float32"))
    return(numerator/denumerator)
def initialise_U(data, cluster_number):
    '''
    data: tensor,shape=(number_dataset,number_features)
    cluster_number: int
    return U:tensor 
    '''
    dataset_number=data.get_shape()[0]
    U=np.random.randint(0,10000,size=(dataset_number,cluster_number))
    sum_U=np.sum(U,axis=1)[:,np.newaxis]
    U=U/sum_U
    return tf.cast(tf.convert_to_tensor(U),tf.float32)
def distance_matrix(data,C,cluster_number,dis):
    '''
    data:tensor,shape=(number_dataset,number_features)
    C:cluster_center,tensor,shape=(cluster_number,features_number)
    cluster_number:int
    '''
    
#    dis=tf.Variable(tf.zeros([data.get_shape()[0],cluster_number]))

    assign_op=[]
    for i in range(cluster_number):
        d=tf.sqrt(tf.reduce_sum(tf.square(data-C[i]),axis=1))
        xx=tf.assign(dis[:,i],d)
        assign_op.append(xx)
             
    return assign_op
def updata_U(distance_matrix,cluster_number,m):
    denominator=tf.Variable(0,dtype=tf.float32)
    for c in range(cluster_number):
        denominator=denominator+(distance_matrix/tf.expand_dims(distance_matrix[:,c],axis=1))** (2/(m-1))
    U= 1/denominator
    
    return U
    
def FCM(data,cluster_number, m)  :
    
#     data = tf.placeholder(tf.float32, shape=(3,4))
     U=tf.Variable(initialise_U(data,cluster_number))
     U_old=tf.Variable(tf.zeros_like(U))
     C=tf.Variable(tf.zeros([cluster_number,data.get_shape()[1]]))
     dis=tf.Variable(tf.zeros([data.get_shape()[0],cluster_number]))
     
     a=tf.assign(C,cluster_center(data,U))
     e=tf.assign(U_old,U)
     with tf.control_dependencies([a,e]):
         b=distance_matrix(data,C,cluster_number,dis)
         with tf.control_dependencies(b):
             
             c=tf.assign(U,updata_U(dis,cluster_number, m))
#                 
             
             
#     data,cluster_number, m,U,U_old,C,dis= tf.while_loop(cond, body, [data,cluster_number, m,U,U_old,C,dis])
     sess=tf.Session()
     sess.run(tf.global_variables_initializer())
     

         
     for i in range(100):
         
         sess.run(c)
     U=sess.run(U)
     sess.close()
     return U
         
def cond(data,cluster_number, m,U,U_old,C,dis):
    return tf.py_func(compare_u, [U, U_old], tf.bool)
def body(data,cluster_number, m,U,U_old,C,dis):
    
    a=tf.assign(C,cluster_center(data,U))
    
    e=tf.assign(U_old,U)
    with tf.control_dependencies([a,e]):
         b,dis=distance_matrix(data,C,cluster_number,dis)
         with tf.control_dependencies(b):
             
             c=tf.assign(U,updata_U(dis,cluster_number, m))
               
    return tf.tuple([data,cluster_number, m,U,U_old,C,dis],control_inputs=[c])
    
            
     
def compare_u(U,U_old):
    if((np.abs(U-U_old)<1e-8).all()):
        print(1)
        return True
    else:
        return False
         
     
     
    
    
if __name__ == '__main__':
    start = time.time()
    data, cluster_location = import_data_format_iris("iris.txt")
    
    data , order = randomise_data(data)
    data=np.array(data,dtype=np.float32)
    a=tf.convert_to_tensor(data)
     
    U=FCM(a,3,2)
    
    final_location = normalise_U(U)
    final_location = de_randomise_data(final_location, order)
    print (checker_iris(final_location))
    print(time.time()-start)
     
#    x,y=sess.run([a,U],feed_dict={data:[[1,2,3,4],
#                [5,6,7,8],
#                [12,13,14,15]]})

        
#    dis=tf.zeros([1,2])
    
    

    
    


