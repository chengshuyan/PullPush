from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import time
import utils
import os
from scipy import misc
from scipy import ndimage
import PIL
import io
import pandas as pd

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')

tf.flags.DEFINE_string('layer_name','InceptionV3/InceptionV3/Mixed_5b/concat','The layer to be attacked.')

tf.flags.DEFINE_string('input_dir', '../PSBA-master/raw_data/imagenet/ILSVRC2012_img_val/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', './adv/FIA/', 'Output directory with images.')

tf.flags.DEFINE_string('label_csv', "./dataset/dev_dataset.csv", 'label information with csv file.')

tf.flags.DEFINE_string('valid_gt', "./dataset/valid_gt.csv", 'label information with csv file.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

"""parameter for DIM"""
tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

def main(_):
    FLAGS.image_size=utils.image_size[FLAGS.model_name]

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name=FLAGS.layer_name

    with tf.Graph().as_default():
        # Prepare graph
        ori_input  = tf.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size*2,num_classes])

        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x=tf.concat([ori_input,adv_input],axis=0)

        logits, end_points = network_fn(x)

        problity=tf.nn.softmax(logits,axis=1)
        pred = tf.argmax(logits, axis=1)
        one_hot = tf.one_hot(pred, num_classes)

        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,checkpoint_path)
            true_label_list,target_label_list = utils.label_dict(FLAGS.label_csv)
            count=0

            label_table = pd.read_csv(FLAGS.valid_gt)
            label_table.insert(label_table.shape[1],'problity',0)
            label_table.insert(label_table.shape[1],'pred',0)
            for images,names,labels,target_labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size,true_label_list,target_label_list):
                count+=FLAGS.batch_size
                if count%100==0:
                    print("Generating:",count)
 
                images_tmp=image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1

                # obtain true label
                labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                #labels = sess.run(one_hot, feed_dict={ori_input: images_tmp, adv_input: images_tmp})

                problity_output, pred_output = sess.run([problity, pred],feed_dict={ori_input: images_tmp, adv_input: images_tmp,label_ph: labels})

                for i in range(FLAGS.batch_size):
                    label_table.loc[label_table['name']+'.png'==names[i],'problity'] = problity_output[i]
                    label_table.loc[label_table['name']+'.png'==names[i],'pred'] = pred_output[i]
                print(label_table)
            label_table.to_csv('./dataset/label50.csv',index=None)

if __name__ == '__main__':
    tf.app.run()