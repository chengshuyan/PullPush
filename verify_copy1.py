import tensorflow as tf
import numpy as np
import argparse
import utils
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_names=['adv_inception_v3','adv_inception_resnet_v2','ens3_adv_inception_v3','ens4_adv_inception_v3','ens_adv_inception_resnet_v2']

# model_names=['inception_v3','inception_v4','inception_resnet_v2','resnet_v1_50','resnet_v1_152',
#             'vgg_16','vgg_19','adv_inception_v3','adv_inception_resnet_v2',
#             'ens3_adv_inception_v3','ens4_adv_inception_v3','ens_adv_inception_resnet_v2']

def verify(model_name,ori_image_path,adv_image_path):

    checkpoint_path=utils.checkpoint_paths[model_name]

    if model_name=='adv_inception_v3' or model_name=='ens3_adv_inception_v3' or model_name=='ens4_adv_inception_v3':
        model_name='inception_v3'
    elif model_name=='adv_inception_resnet_v2' or model_name=='ens_adv_inception_resnet_v2':
        model_name='inception_resnet_v2'

    num_classes=1000+utils.offset[model_name]

    network_fn = utils.nets_factory.get_network_fn(
        model_name,
        num_classes=(num_classes),
        is_training=False)

    image_preprocessing_fn = utils.normalization_fn_map[model_name]
    image_size = utils.image_size[model_name]

    batch_size=200
    image_ph=tf.placeholder(dtype=tf.float32,shape=[batch_size,image_size,image_size,3])

    logits, _ = network_fn(image_ph)
    max_logits = tf.reduce_max(tf.nn.softmax(logits,axis=1),axis=-1)#转成概率再求最大更合适
    predictions = tf.argmax(logits, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)

        ori_pre=[] # prediction for original images
        adv_pre=[] # prediction label for adversarial images
        ground_truth=[] # grund truth for original images
        adv_logits=[]
        target_labels_all=[]
        label_csv="./dataset/dev_dataset.csv"
        true_label_list,target_label_list = utils.label_dict(label_csv)
        for images,names,labels,target_labels in utils.load_image(ori_image_path, image_size, batch_size,true_label_list,target_label_list):
            images=image_preprocessing_fn(images)
            pres=sess.run(predictions,feed_dict={image_ph:images})
            ground_truth.extend(labels)
            ori_pre.extend(pres)

        for images,names,labels,target_labels in utils.load_image(adv_image_path, image_size, batch_size,true_label_list,target_label_list):
            images=image_preprocessing_fn(images)
            pres,pres_logits,logits_=sess.run([predictions,max_logits,logits],feed_dict={image_ph:images})
            #print(pres_logits)
            #print(pres)
            #print(logits_)
            adv_pre.extend(pres)
            adv_logits.extend(pres_logits)
            target_labels_all.extend(target_labels)
    tf.reset_default_graph()

    ori_pre=np.array(ori_pre)
    adv_pre=np.array(adv_pre)
    ground_truth=np.array(ground_truth)
    adv_logits=np.array(adv_logits)
    target_labels_all=np.array(target_labels_all)

    if num_classes==1000:
        ground_truth=ground_truth-1
        target_labels_all=target_labels_all-1


    return ori_pre,adv_pre,ground_truth,adv_logits,target_labels_all


def main(ori_path='./dataset/images/',adv_path='./adv/',output_file='./log.csv',num=10,index=5):
    ori_accuracys=[]
    adv_accuracys=[]
    adv_successrates=[]

    tSuc=[]
    uTR=[]
    tTR=[]
    uTR_n=[]
    tTR_n=[]

    arr = []
    arr_target = []
    target_labels = []

    with open(output_file,'a+',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([adv_path])
        writer.writerow(model_names)
        count = 0
        for model_name in model_names:
            print(model_name)
            ori_pre,adv_pre,ground_truth,adv_logits,target_labels=verify(model_name,ori_path,adv_path)
            ori_accuracy = np.sum(ori_pre == ground_truth)/1000
            adv_accuracy = np.sum(adv_pre == ground_truth)/1000
            #原始预测和攻击后预测不用才算————非定向攻击成功
            adv_successrate = np.sum(ori_pre != adv_pre)/1000
            adv_successrate2 = np.sum(ground_truth != adv_pre) / 1000
            print('ori_acc:{:.1%}/adv_acc:{:.1%}/adv_suc:{:.1%}/adv_suc2:{:.1%}'.format(ori_accuracy,adv_accuracy,adv_successrate,adv_successrate2))
            ori_accuracys.append('{:.1%}'.format(ori_accuracy))
            adv_accuracys.append('{:.1%}'.format(adv_accuracy))
            adv_successrates.append('{:.1%}'.format(adv_successrate))

            adv_successrate_target = np.sum(target_labels == adv_pre)/1000
            tSuc.append('{:.1%}'.format(adv_successrate_target))

            arr.append((ori_pre != adv_pre).astype(int))
            arr_target.append((target_labels == adv_pre).astype(int))

            if count == index:
                adv_logits_raw = adv_logits

            count += 1

        for i in range(len(model_names)):
            uTR.append('{:.1%}'.format(np.sum(np.logical_and(arr[i],arr[index]))/np.sum(arr[index])))
        
        for i in range(len(model_names)):
            tTR.append('{:.1%}'.format(np.sum(np.logical_and(arr_target[i],arr_target[index]))/np.sum(arr_target[index])))

        tmp = arr[index] * adv_logits_raw
        adv_logits_index=np.where(tmp >= tmp[tmp.argsort()[-num:][0]],1,0)

        tmp = arr_target[index] * adv_logits_raw
        adv_logits_index_target=np.where(tmp >= tmp[tmp.argsort()[-num:][0]],1,0)

        print('untarget success num \t',np.sum(arr[index]))
        print('target success num \t',np.sum(arr_target[index]))

        for i in range(len(model_names)):
            uTR_n.append('{:.1%}'.format(np.sum(np.logical_and(arr[i],adv_logits_index))/num))

        for i in range(len(model_names)):
            tTR_n.append('{:.1%}'.format(np.sum(np.logical_and(arr_target[i],adv_logits_index_target))/num))
        # print(adv_successrates)
        # writer.writerow(ori_accuracys)
        writer.writerow(['untargeted success rate'])
        writer.writerow(adv_successrates)
        writer.writerow(['targeted success rate'])
        writer.writerow(tSuc)
        writer.writerow(['untargeted transfer rate'])
        writer.writerow(uTR)
        writer.writerow(['targeted transfer rate'])
        writer.writerow(tTR)
        writer.writerow(['untargeted transfer rate_'+str(num)])
        writer.writerow(uTR_n)
        writer.writerow(['targeted transfer rate_'+str(num)])
        writer.writerow(tTR_n)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ori_path', default='./dataset/images/')
    parser.add_argument('--adv_path',default='./adv/FIA-ince_v3/')
    parser.add_argument('--output_file', default='./log.csv')
    parser.add_argument('--num', default=300)
    parser.add_argument('--index', default=0)
    args=parser.parse_args()
    main(args.ori_path,args.adv_path,args.output_file,args.num,args.index)
