# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.layers as layer


def leak_relu(x, leak=0.3):
    return tf.maximum(leak*x, x)

class WGAN(object):
    def __init__(self):
        self.device = '/gpu:0'
        self.channel = 3       
        self.z_dim   = 128
        
        self.batch_size = 64
        self.lr_gen = 5e-5          #learning rate of gennerator
        self.lr_cri = 5e-5          #learning rate of critics
        
        self.clip_lower = -0.01     
        self.clip_upper = 0.01
        
        self.log_dir = './log'
        self.ckpt    = './ckpt'
        
        with tf.device(self.device):
            self.net = self.build_graph()
            
    def build_graph(self):
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        
        self.fake_data = self.generator(self.z)
        
        self.true_logit = self.critic(self.real_data)
        self.false_logit = self.critic(self.fake_data)
        
        self.c_loss = tf.reduce_mean(self.false_logit - self.true_logit)
        self.g_loss = tf.reduce_mean(-self.false_logit)
        
        c_loss_sum = tf.summary.scalar('c_loss', self.c_loss)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        image_sum   = tf.summary.image('img', self.fake_data, max_outputs=10)
        
        self.theta_g = tf.get_collection(tf.GraphKeys.TRTRAINABLE_VARIABLES, scope='generator')
        self.theta_c = tf.get_collection(tf.GraphKeys.TRTRAINABLE_VARIABLES, scope='critic')
        
        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.float32)
        
        self.opt_g = layer.optimize_loss(loss=self.g_loss, 
                                    optimizer=tf.train.RMSPropOptimizer,
                                    learning_rate=self.lr_gen, variables=theta_g,
                                    global_step=counter_g, 
                                    summaries=' h')
        self.opt_c = layer.optimize_loss(loss=self.c_loss, 
                                    optimizer=tf.train.RMSPropOptimizer,
                                    learning_rate=self.lr_cri, variables=theta_c,
                                    global_step=counter_c)
        self.clip_theta_c = [tf.assign(theta, tf.clip_by_value(theta, self.clip_lower, self.clip_upper)) for theta in theta_c]
                        
        with tf.control_dependencies(self.clip_theta_c):
            self.opt_c = tf.tuple(self.clip_theta_c)
        
    def generator(self, z):
        with tf.variable_scope('generator') as scope:
            img = layer.fully_connected(self.z, num_outputs=4*4*512, 
                                        activation_fn=leak_relu, normalizer_fn=layer.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02))
            img = tf.reshape(img, [-1, 4, 4, 512])
            img = layer.conv2d_transpose(img, num_outputs=256, kernel_size=3, stride=2, padding='SAME',
                                         activation_fn=tf.nn.relu, normalizer_fn=layer.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02))
            img = layer.conv2d_transpose(img, num_outputs=128, kernel_size=3, stride=2, padding='SAME', 
                                         activation_fn=tf.nn.relu, normalizer_fn=layer.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02))
            img = layer.conv2d_transpose(img, num_outputs=64, kernel_size=3, stride=2, padding='SAME', 
                                         activation_fn=tf.nn.relu, normalizer_fn=layer.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02))
            img = layer.conv2d_transpose(img, num_outputs=self.channel, kernel_size=3, stride=1, padding='SAME', 
                                         activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.02))
        return img
            
    def critic(self, img, reuse = False):
        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            size =64
            logit = layer.conv2d(img, num_outputs=size, kernel_size=3, stride=2,
                                 activation_fn=leak_relu,  weights_initializer=tf.random_normal_initializer(0, 0.02))
            logit = layer.conv2d(logit, num_outputs=size * 2, kernel_size=3, stride=2,
                                 activation_fn=leak_relu, normalizer_fn=layer.batch_norm,  weights_initializer=tf.random_normal_initializer(0, 0.02))
            logit = layer.conv2d(logit, num_outputs=size * 4, kernel_size=3, stride=2,
                                 activation_fn=leak_relu, normalizer_fn=layer.batch_norm,  weights_initializer=tf.random_normal_initializer(0, 0.02))
            logit = layer.conv2d(logit, num_outputs=size * 8, kernel_size=3, stride=2,
                                 activation_fn=leak_relu, normalizer_fn=layer.batch_norm,  weights_initializer=tf.random_normal_initializer(0, 0.02))
            logit = layer.fully_connected(tf.reshape(logit, [self.batch_size, -1]), 
                                          num_outputs=1, activation_fn=None)  #activation_fn = None !!!  
        return logit
        
        
        
    
        
    