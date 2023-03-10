import pandas as pd
import numpy as np
import shutil
import tensorflow.compat.v1 as tf
# from tensorflow.keras.utils import to_categorical

path50 = './dataset/imagenet/'
batch50 = 50
# path20 = './dataset/imagenet_20/'
# batch20 = 20
# file20 = './dataset/label20'

path20 = './dataset/imagenet_8/'
batch20 = 8
file20 = './dataset/label8'

def get_label20():
    label50 = pd.read_csv('./dataset/label50.csv')
    matrix = label50.values
    matrix = matrix[matrix[:,1].argsort()]
    label20 = np.array(label50.loc[:batch20*1000])

    for batch in range(1000):
        tmp = np.array(matrix[batch*batch50:(batch+1)*batch50])
        tmp = tmp[np.where(tmp[:,1]==tmp[:,3])]
        if tmp.shape[0] < batch20:
            print(tmp.shape[0])
        tmp = tmp[tmp[:,2].argsort()]
        label20[batch*batch20:(batch+1)*batch20] = tmp[:batch20]
    np.save(file20,label20)
    for i in range(batch20*1000):
        shutil.copyfile(path50+label20[i][0]+'.JPEG',path20+label20[i][0]+'.JPEG')

if __name__=='__main__':
    get_label20()