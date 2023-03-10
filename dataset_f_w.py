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

slim = tf.contrib.slim

tf.flags.DEFINE_string('model_name', 'inception_v3', 'The Model used to generate adv.')

tf.flags.DEFINE_string('layer_name','InceptionV3/InceptionV3/Mixed_5b/concat','The layer to be attacked.')

tf.flags.DEFINE_string('input_dir', '../PSBA-master/raw_data/imagenet/ILSVRC2012_img_val_20/', 'Input directory with images.')

tf.flags.DEFINE_string('label_csv', "./dataset/dev_dataset.csv", 'label information with csv file.')

tf.flags.DEFINE_string('GPU_ID', '0', 'which GPU to use.')

tf.flags.DEFINE_integer('batch_size', 20, 'How many images process at one time.')

tf.flags.DEFINE_integer('image_size', 224, 'size of each input images.')

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID

"""obtain the feature map of the target layer"""
def get_opt_layers(layer_name):
    opt_operations = []
    #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
    operations = tf.get_default_graph().get_operations()
    for op in operations:
        if layer_name == op.name:
            opt_operations.append(op.outputs[0])
            shape=op.outputs[0][:FLAGS.batch_size].shape
            break
    return opt_operations,shape

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square)
    return nor_grad

def main():
    FLAGS.image_size=utils.image_size[FLAGS.model_name]

    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name=FLAGS.layer_name

    with tf.Graph().as_default():
        ori_input  = tf.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size,num_classes])
        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x = ori_input
        logits, end_points = network_fn(x)
        # problity=tf.nn.softmax(logits,axis=1)
        # pred = tf.argmax(logits, axis=1)
        # one_hot = tf.one_hot(pred, num_classes)

        opt_operations,shape = get_opt_layers(layer_name)
        weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
        ori_tensor = opt_operations[0]

        saver=tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,checkpoint_path)
            true_label_list,target_label_list = utils.label_dict(FLAGS.label_csv)
            count=0

            weight = []
            feature = []
            for images,names,labels,target_labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size,FLAGS.batch_size,true_label_list,target_label_list):
                count+=FLAGS.batch_size
                if count%100==0:
                    print("Generating:",count)
                images_tmp=image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels=labels-1
                labels= to_categorical(np.concatenate([labels,labels],axis=-1),num_classes)
                weight_tmp,feature_tmp = sess.run([weights_tensor,opt_operations[0]],feed_dict={ori_input: images_tmp, label_ph: labels})
                weight.append(normalize(weight_tmp, 2))
                feature.append(feature_tmp)
            np.save('./predataset/'+FLAGS.model_name+'/weight',weight)
            np.save('./predataset/'+FLAGS.model_name+'/feature',feature)

if __name__ == '__main__':
    tf.app.run()