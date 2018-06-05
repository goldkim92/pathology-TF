import os
import re
import numpy as np
import random
import scipy.misc as scm
import tensorflow as tf

from collections import namedtuple
from tqdm import tqdm
from glob import glob

import module
import util 

class Network(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.phase = args.phase
        self.data_dir = args.data_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.image_c = args.image_c
        self.label_n = args.label_n
        self.nf = args.nf
        self.lr = args.lr
        self.beta1 = args.beta1
        self.sample_step = args.sample_step
        self.log_step = args.log_step
        self.ckpt_step = args.ckpt_step
        
        # hyper parameter for building module
        OPTIONS = namedtuple('options', ['batch_size', 'input_size', 'image_c', 'nf', 'label_n'])
        self.options = OPTIONS(self.batch_size, self.input_size, self.image_c, self.nf, self.label_n)
        
        # build model & make checkpoint saver
        self.build_model()
        self.saver = tf.train.Saver()
        
        # labels
        self.labels_dic = util.get_labels(os.path.join('data','labels.xlsx'))
    
    def build_model(self):
        # placeholder
        self.place_images = tf.placeholder(tf.float32, 
                                          [None,self.input_size,self.input_size,self.image_c],
                                          name='place_images')
        self.place_labels = tf.placeholder(tf.float32, [None,self.label_n], name='labels')
        
        # loss funciton
        self.pred = module.classifier(self.place_images, self.options, reuse=False, name='net')
        self.loss = module.cls_loss(logits=self.pred, labels=self.place_labels)
        
        # accuracy
        corr = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.place_labels, 1))    
        self.accr_count = tf.reduce_sum(tf.cast(corr, "float"))

        # trainable variables
        t_vars = tf.trainable_variables()
#        self.module_vars = [var for var in t_vars if 'densenet' in var.name]
#        for var in t_vars: print(var.name)
        
        # optimizer
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=t_vars)
        
        # placeholder for summary
        self.total_loss = tf.placeholder(tf.float32)
        self.accr = tf.placeholder(tf.float32)
        
        # summary setting
        self.summary()
        
    def train(self):
        
        # load train-data & valid-data file list (label & image file)
        files = glob(os.path.join('data','224','*')) # len(files) = 6985
        usable_files = [file for file in files if re.split('[/_.]+', file)[2] in self.labels_dic.keys()] # len(usable_files) = 5000
        np.random.shuffle(usable_files)
        train_files = usable_files[:4000] # 4000
        valid_files = usable_files[4000:4500] # 500
        test_files = usable_files[4500:] # 500
        
        batch_idxs = len(train_files) // self.batch_size
        
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
            
        count_idx = 0
        # train
        for epoch in range(self.epoch):
            print('Epoch[{}/{}]'.format(epoch+1, self.epoch))
            
            np.random.shuffle(train_files)
            np.random.shuffle(valid_files)
            self.train_lst = train_files[:1000] # this is for accuracy
            self.valid_lst = valid_files[:] # this is for accuracy
            
            cost = 0
            for i in tqdm(range(batch_idxs)):
                # get batch images and labels
                lst = train_files[ i*self.batch_size : (i+1)*self.batch_size ]
                images, labels = self.preprocessing(lst, phase='train')
                          
                # update network
                feeds = {self.place_images: images, self.place_labels: labels}
                _, summary_loss = self.sess.run([self.optim, self.sum_loss], feed_dict=feeds)
              
                count_idx += 1
                
                # log step (summary)
                if count_idx % self.log_step == 0:
                    train_accr = self.accuracy('train')
                    valid_accr = self.accuracy('valid')
                    
#                     test_accr = self.accuracy('test')
#                     test_real_accr = self.accuracy('test_real')
#                     test_fake_accr = self.accuracy('test_fake')
                    
                    self.writer_cost.add_summary(summary_loss, count_idx)

                    summary = self.sess.run(self.sum_accr, feed_dict={self.accr:train_accr})
                    self.writer_train_accr.add_summary(summary, count_idx)

                    summary = self.sess.run(self.sum_accr, feed_dict={self.accr:valid_accr})
                    self.writer_valid_accr.add_summary(summary, count_idx)
                    
#                     summary = self.sess.run(self.sum_accr, feed_dict={self.accr:test_accr})
#                     self.writer_test_accr.add_summary(summary, count_idx)

#                     summary = self.sess.run(self.sum_accr, feed_dict={self.accr:test_real_accr})
#                     self.writer_test_real_accr.add_summary(summary, count_idx)

#                     summary = self.sess.run(self.sum_accr, feed_dict={self.accr:test_fake_accr})
#                     self.writer_test_fake_accr.add_summary(summary, count_idx)
                    
                    print('train: {:.04f}'.format(train_accr))
                    print('valid: {:.04f}'.format(valid_accr))
            
                # checkpoint step
                if count_idx % self.ckpt_step == 0:
                    self.checkpoint_save(count_idx)


    def test(self):
        # load test-data file list
        self.test_real_lst = glob(os.path.join('data','test','real','*'))
        self.test_fake_lst = glob(os.path.join('data','test','fake','*'))
        test_lst = self.test_real_lst + self.test_fake_lst
        np.random.shuffle(test_lst)
        self.test_lst = test_lst
        
        self.sess.run(tf.global_variables_initializer())
        
        # load checkpoint
        if self.checkpoint_load():
            print(" [*] checkpoint load SUCCESS ")
        else:
            print(" [!] checkpoint load failed ")
        
        # print test accuracy
        accr = self.accuracy('test')
        print(accr)
        
    
    def real_test(self):
        # load real_test-data file list
        lists = glob(os.path.join('data','real_test','*'))
        lists.sort(key=util.natural_keys)
        # np.random.shuffle(lists)
        
        self.sess.run(tf.global_variables_initializer())
        
        # load checkpoint
        if self.checkpoint_load():
            print(" [*] checkpoint load SUCCESS ")
        else:
            print(" [!] checkpoint load failed ")
        
        # remove file if exist
        result_file = os.path.join(self.test_dir, 'result.txt')
        try: os.remove(result_file)
        except: pass
        
        accr = 0.
        fake_pred = []
        i=0
        for i in range(len(lists) // self.batch_size):
            # get batch images and labels
            lst = lists[ i*self.batch_size : (i+1)*self.batch_size ]
            images, labels = self.preprocessing(lst, phase='test')
            feeds = {self.place_images: images, self.place_labels: labels}
            cur_accr,fake_pred = self.sess.run([self.accr_count, self.pred[:,0]], feed_dict=feeds)
            accr += cur_accr
            print(len(fake_pred))
            # write in file
            with open(result_file, 'a') as f:
                for i in range(len(lst)):
                    file_name = lst[i].split('/')[2]
                    file_name = file_name.split('.')[0]
                    f.write(file_name + ',' + '{:.4f}'.format(fake_pred[i]) + '\n')

    
    def summary(self):
        # summary writer
        self.writer_cost = tf.summary.FileWriter(os.path.join(self.log_dir,'cost'), self.sess.graph)
        self.writer_train_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'train_accr'),self.sess.graph)
        self.writer_valid_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'valid_accr'),self.sess.graph)
#         self.writer_test_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'test_accr'),self.sess.graph)
#         self.writer_test_real_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'test_real_accr'),self.sess.graph)
#         self.writer_test_fake_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'test_fake_accr'),self.sess.graph)
        
        # summary session
        self.sum_loss = tf.summary.scalar('loss value',self.loss)
        self.sum_accr = tf.summary.scalar('accr', self.accr)


    def accuracy(self, phase='valid'):
        # train or validate or test
        if phase == 'train':
            idxs = len(self.train_lst) // self.batch_size
            lists = self.train_lst
        elif phase == 'valid':
            idxs = len(self.valid_lst) // self.batch_size
            lists = self.valid_lst
#         elif phase == 'test':
#             idxs = len(self.test_lst) // self.batch_size
#             lists= self.test_lst
#         elif phase == 'test_real':
#             idxs = len(self.test_real_lst) // self.batch_size
#             lists = self.test_real_lst
#         elif phase == 'test_fake':
#             idxs = len(self.test_fake_lst) // self.batch_size
#             lists = self.test_fake_lst
            
        accr = 0.
        i=0
        for i in range(idxs):
            # get batch images and labels
            lst = lists[ i*self.batch_size : (i+1)*self.batch_size ]
            images, labels = self.preprocessing(lst, phase=phase)
            feeds = {self.place_images: images, self.place_labels: labels}
            accr += self.sess.run(self.accr_count, feed_dict=feeds)
        accr = accr / ((i+1)*self.batch_size)
            
        return accr 

    
    def checkpoint_save(self, count):
        model_name = "net.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=count)
    
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False
    
    def preprocessing(self, lst, phase):
        labels = []
        images = []
        
        for file in lst:
            person = re.split('[/_.]+',file)[2]

            labels.append(self.labels_dic[person])

            img = util.get_image(file, 112, phase=phase)
            images.append(img)

        labels = np.array(labels)
        images = np.array(images)
        
        return images, labels
    