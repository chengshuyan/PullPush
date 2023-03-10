import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
# from matplotlib import rcParams
# from mpl_toolkits.axisartist.axislines import Subplot
# from matplotlib.pyplot import MultipleLocator
# import utils
# import tensorflow as tf

W_s = np.load('./Ws/Ws1.npy')

# fig,ax = plt.subplots(nrows=6,ncols=6,figsize=(64,64))

# for i in range(20):
#     #ax[i//6][i%6].imshow(np.sign(g_ince_0.transpose(0,3,1,2)[num][i]))
#     ax[i//4][i%5].imshow(W_s[i])
    
# plt.savefig('temp2.jpg')

# print(W_s[0])

# print(W_s[1])

# print(W_s[2])

# for i in range(20):
#     a = np.sign(W_s[i])
#     num1 = np.sum(a==1)
#     num_1 = np.sum(a==-1)
#     num0 = np.sum(a==0)
#     print("W_s",i,"------\n","1___",num1,"\n-1___",num_1,'\n0___',num0)

# print(np.sqrt(2))

# true_label_list,target_label_list = utils.label_dict("./dataset/dev_dataset.csv")

# # print(true_label_list)


# for images,names,labels,target_labels in utils.load_image('./dataset/images/',224,20,true_label_list,target_label_list):
#     print(names)

# tf.gfile.Copy('./temp2.jpg', './temp1.jpg', overwrite=False)

a = [-0.000115334646,0.000116157324, 8.226747e-07,-7.961155e-05,0.00011104836,3.143681e-05,-5.6896504e-05, 0.00010882518,5.1928677e-05,
-4.1525767e-05,0.00010827952,6.675375e-05,-3.0164694e-05,0.00010872755,7.856286e-05,-2.1364176e-05,0.0001097365,8.8372326e-05,-1.4319259e-05,
0.00011097999,9.666073e-05,-8.513496e-06,0.00011222819,0.000103714694,-3.6365745e-06,0.00011334139,0.000109704815,5.4683187e-07,0.000114268936,0.00011481577]

print(a[::3])
print(a[1::3])
print(a[2::3])

# a=[]
# a1 = np.zeros([299,299,3])
# a2 = np.zeros([299,299])
# a.append(a1)
# a.append(a2)

# print(np.array(a))

# image=Image.open('./dataset/images/0aebe24fc257286e.png')

# image.save('1.png')
# print(np.array(image).shape)
# print(len(image.split()))

loss___ -1.1942802e-09
loss1___ 8.3174643e-07
loss2___ 8.3055215e-07
loss___ 4.2164174e-09
loss1___ 8.290842e-07
loss2___ 8.333006e-07
loss___ 8.880477e-09
loss1___ 8.2727246e-07
loss2___ 8.3615294e-07
loss___ 1.2801934e-08
loss1___ 8.261252e-07
loss2___ 8.389271e-07
loss___ 1.6180081e-08
loss1___ 8.253318e-07
loss2___ 8.415119e-07
loss___ 1.9175047e-08
loss1___ 8.247822e-07
loss2___ 8.4395725e-07
loss___ 2.1869937e-08
loss1___ 8.2451584e-07
loss2___ 8.463858e-07
loss___ 2.4327903e-08
loss1___ 8.2439095e-07
loss2___ 8.4871886e-07
loss___ 2.6589305e-08
loss1___ 8.2448184e-07
loss2___ 8.5107115e-07
loss___ 2.8678414e-08
loss1___ 8.2479187e-07
loss2___ 8.534703e-07