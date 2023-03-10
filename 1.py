import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# file = r"D:\secure\攻防规避攻击\PSBA-master\raw_data\imagenet\ILSVRC2012_img_val\ILSVRC2012_val_00000034.JPEG"
# image=Image.open(file)
# #image=image.resize((224,224))
# print(np.array(image).shape)
# print(np.array(image))

#file1 = "./adv/TFA/0aebe24fc257286e.png"
file1 = "./adv/TFA/0af0a5dfee6b84ff.png"
img = Image.open(file1)
#img.show("1")
img = np.array(img)

#file2 = "./dataset/images/0aebe24fc257286e.png"

file2 = "./dataset/images/0af0a5dfee6b84ff.png"
img1 = Image.open(file2)
#img1.show('2')
img1 = np.array(img1)
print(img.shape)
print(img1.shape)
print(np.sum(img==img1))
print(299*299*3)
print(np.sum(img==img1)/(299*299*3))

a=plt.imread(file1)
b=plt.imread(file2)

# print('img')
# print(a)
# print('img1')
# print(b)
# print('img-img1')
# print(a-b)
# print('end')

print(len(np.unique(a-b)))
print(np.unique(a-b))

# np.save('a.txt',(a-b).reshape(()))
# plt.figure()
# plt.subplot(2,2,1)		# 将画板分为2行两列，本幅图位于第一个位置
# plt.imshow(a)
# plt.subplot(2,2,2)		# 将画板分为2行两列，本幅图位于第二个位置
# plt.imshow(b)
# plt.subplot(2,2,3)		# 将画板分为2行两列，本幅图位于第二个位置
# plt.imshow(a-b)
# plt.show()

# print((np.array([1,1])==np.array([1,0])).shape)
# print(np.array([True,False],dtype=int))
# c = np.array([1,1])==np.array([1,0])
# print(c.astype(int))

# print(np.sum(np.logical_and([1,0,0],[1,0,1]))/np.sum([1,0,1]))

# a = np.array([1, 3, 2, 4, 5])
# element = a[a.argsort()[-3:][0]]
# print(element)
# print(np.where(a>=element,1,0))
# print(a*a)

# print(25.30*10/300)

# print(0.6*10/300)