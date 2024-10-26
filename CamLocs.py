import tensorflow as tf

import numpy as np
# import tensorflow.keras.datasets.cifar10 as 
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

def load_data(raw_data,num_tr):
    # 一条子载波以平移复制的形式进行复用
    np.random.shuffle(raw_data)
    x1,y1 = np.split(raw_data, (50,), axis=1)
    y1 = y1.astype(np.uint8)
    temp1 = np.zeros((len(raw_data),10,10))
    for i in range(len(raw_data)):
        for j in range(10):
            for k in range(10):
                temp1[i,j,k] = raw_data[i,5*j+(k%5)]
    if num_tr==0:
        xtr = temp1
        ytr = y1       
    return xtr,ytr

class Baseline(tf.keras.Model):
    def __init__(self,input_shape=(10,10,1)):
        super(Baseline, self).__init__()        
        self.input_layer = tf.keras.layers.Input(input_shape)
        self.c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')
        self.p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.c2 = tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), padding='same')
        self.b2 = tf.keras.layers.BatchNormalization()
        self.a2 = tf.keras.layers.Activation('relu')
        self.p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = tf.keras.layers.Dropout(0.6)
        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.6)
        self.f2 = tf.keras.layers.Dense(2, activation='sigmoid')
        self.out = self.call(self.input_layer) 

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

# test = np.loadtxt('E:/matlab/data/location/B202/feature/raw_amp_an01_tr.txt', dtype=float, delimiter=',')
# test1 = np.loadtxt('E:/matlab/data/location/B202/20221116/f_amp_E1_an01.txt', dtype=float, delimiter=',')
# test3 = np.loadtxt('E:/matlab/data/location/B202/feature/raw_amp_MI_an01_te.txt', dtype=float, delimiter=',')
# test4 = np.loadtxt('E:/matlab/data/location/B202/20221116/f_amp_E2_an01.txt', dtype=float, delimiter=',')

'''
# device training and test
test1_0 = np.loadtxt('E:/matlab/data/location/device/an0_MI_amp.txt', dtype=float, delimiter=',')
test1_1 = np.loadtxt('E:/matlab/data/location/device/an1_MI_amp.txt', dtype=float, delimiter=',')
test2_0 = np.loadtxt('E:/matlab/data/location/device/an0_Y3_amp.txt', dtype=float, delimiter=',')
test2_1 = np.loadtxt('E:/matlab/data/location/device/an1_Y3_amp.txt', dtype=float, delimiter=',')
test3_0 = np.loadtxt('E:/matlab/data/location/device/an0_C2HC_amp.txt', dtype=float, delimiter=',')
test3_1 = np.loadtxt('E:/matlab/data/location/device/an1_C2HC_amp.txt', dtype=float, delimiter=',')
test4_0 = np.loadtxt('E:/matlab/data/location/device/an0_H6C_amp.txt', dtype=float, delimiter=',')
test4_1 = np.loadtxt('E:/matlab/data/location/device/an1_H6C_amp.txt', dtype=float, delimiter=',')
test5_0 = np.loadtxt('E:/matlab/data/location/device/an0_C43_amp.txt', dtype=float, delimiter=',')
test5_1 = np.loadtxt('E:/matlab/data/location/device/an1_C43_amp.txt', dtype=float, delimiter=',')
test6_0 = np.loadtxt('E:/matlab/data/location/device/an0_6C_amp.txt', dtype=float, delimiter=',')
test6_1 = np.loadtxt('E:/matlab/data/location/device/an1_6C_amp.txt', dtype=float, delimiter=',')
test7_0 = np.loadtxt('E:/matlab/data/location/device/an0_D806_amp.txt', dtype=float, delimiter=',')
test7_1 = np.loadtxt('E:/matlab/data/location/device/an1_D806_amp.txt', dtype=float, delimiter=',')
test8_0 = np.loadtxt('E:/matlab/data/location/device/an0_V38_amp.txt', dtype=float, delimiter=',')
test8_1 = np.loadtxt('E:/matlab/data/location/device/an1_V38_amp.txt', dtype=float, delimiter=',')
test9_0 = np.loadtxt('E:/matlab/data/location/device/an0_83do_amp.txt', dtype=float, delimiter=',')
test9_1 = np.loadtxt('E:/matlab/data/location/device/an1_83do_amp.txt', dtype=float, delimiter=',')

tr1 = np.vstack((test1_0,test1_1))
tr2 = np.vstack((test2_0,test2_1))
tr3 = np.vstack((test3_0,test3_1))
tr4 = np.vstack((test4_0,test4_1))
tr5 = np.vstack((test5_0,test5_1))
tr6 = np.vstack((test6_0,test6_1))
tr7 = np.vstack((test7_0,test7_1))
tr8 = np.vstack((test8_0,test8_1))
tr9 = np.vstack((test9_0,test9_1))

tr = np.vstack((tr5,tr9))

x_tr1, y_tr1 = load_data(tr1,0)
x_tr2, y_tr2 = load_data(tr2,0)
x_tr3, y_tr3 = load_data(tr3,0)
x_tr4, y_tr4 = load_data(tr4,0)
x_tr5, y_tr5 = load_data(tr5,0)
x_tr6, y_tr6 = load_data(tr6,0)
x_tr7, y_tr7 = load_data(tr7,0)
x_tr8, y_tr8 = load_data(tr8,0)
x_tr9, y_tr9 = load_data(tr9,0)

x_tr, y_tr = load_data(tr,0)

x_tr1 = x_tr1[...,tf.newaxis]
x_tr2 = x_tr2[...,tf.newaxis]
x_tr3 = x_tr3[...,tf.newaxis]
x_tr4 = x_tr4[...,tf.newaxis]
x_tr5 = x_tr5[...,tf.newaxis]
x_tr6 = x_tr6[...,tf.newaxis]
x_tr7 = x_tr7[...,tf.newaxis]
x_tr8 = x_tr8[...,tf.newaxis]
x_tr9 = x_tr9[...,tf.newaxis]

x_tr = x_tr[...,tf.newaxis]
'''
'''
test1 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U1_4m_amp.txt', dtype=float, delimiter=',')
test2 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U2_4m_amp.txt', dtype=float, delimiter=',')
test3 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U3_4m_amp.txt', dtype=float, delimiter=',')
test4 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U4_4m_amp.txt', dtype=float, delimiter=',')
test5 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U5_4m_amp.txt', dtype=float, delimiter=',')
test6 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U6_4m_amp.txt', dtype=float, delimiter=',')
test7 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U7_4m_amp.txt', dtype=float, delimiter=',')
test8 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U8_4m_amp.txt', dtype=float, delimiter=',')
test9 = np.loadtxt('E:/matlab/data/location/user/amp/an0_U9_4m_amp.txt', dtype=float, delimiter=',')
test10= np.loadtxt('E:/matlab/data/location/user/amp/an0_U10_4m_amp.txt', dtype=float, delimiter=',')

x_tr1, y_tr1 = load_data(test1,0)
x_tr2, y_tr2 = load_data(test2,0)
x_tr3, y_tr3 = load_data(test3,0)
x_tr4, y_tr4 = load_data(test4,0)
x_tr5, y_tr5 = load_data(test5,0)
x_tr6, y_tr6 = load_data(test6,0)
x_tr7, y_tr7 = load_data(test7,0)
x_tr8, y_tr8 = load_data(test8,0)
x_tr9, y_tr9 = load_data(test9,0)
x_tr10,y_tr10= load_data(test10,0)

x_tr1 = x_tr1[...,tf.newaxis]
x_tr2 = x_tr2[...,tf.newaxis]
x_tr3 = x_tr3[...,tf.newaxis]
x_tr4 = x_tr4[...,tf.newaxis]
x_tr5 = x_tr5[...,tf.newaxis]
x_tr6 = x_tr6[...,tf.newaxis]
x_tr7 = x_tr7[...,tf.newaxis]
x_tr8 = x_tr8[...,tf.newaxis]
x_tr9 = x_tr9[...,tf.newaxis]
x_tr10= x_tr10[...,tf.newaxis]
'''

# test1 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_B202.txt', dtype=float, delimiter=',')
# test2 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_B205.txt', dtype=float, delimiter=',')
# test3 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_B211.txt', dtype=float, delimiter=',')
# test4 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_B405.txt', dtype=float, delimiter=',')
# test5 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_C105.txt', dtype=float, delimiter=',')
# test6 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_C205.txt', dtype=float, delimiter=',')
# test7 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_room7.txt', dtype=float, delimiter=',')
test8 = np.loadtxt('E:/matlab/data/location/room/an01_Y3_amp_room8.txt', dtype=float, delimiter=',')

# x_tr1, y_tr1 = load_data(test1,0)
# x_tr2, y_tr2 = load_data(test2,0)
# x_tr3, y_tr3 = load_data(test3,0)
# x_tr4, y_tr4 = load_data(test4,0)
# x_tr5, y_tr5 = load_data(test5,0)
# x_tr6, y_tr6 = load_data(test6,0)
# x_tr7, y_tr7 = load_data(test7,0)
x_tr8, y_tr8 = load_data(test8,0)

# x_tr, y_tr = load_data(tr,0)

# x_tr1 = x_tr1[...,tf.newaxis]
# x_tr2 = x_tr2[...,tf.newaxis]
# x_tr3 = x_tr3[...,tf.newaxis]
# x_tr4 = x_tr4[...,tf.newaxis]
# x_tr5 = x_tr5[...,tf.newaxis]
# x_tr6 = x_tr6[...,tf.newaxis]
# x_tr7 = x_tr7[...,tf.newaxis]
x_tr8 = x_tr8[...,tf.newaxis]

# x_tr = x_tr[...,tf.newaxis]

# test = np.vstack((test5,test9))
# x_tr, y_tr = load_data(test,0)
# x_tr = x_tr[...,tf.newaxis]



model = Baseline()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
              # metrics=[tf.keras.metrics.sparse_categorical_accuracy])
# model.build(input_shape=(None, 10, 10, 1))
# model.call(input(shape=(10, 10, 1)))
history = model.fit(x_tr8, y_tr8, batch_size=200, epochs=1, validation_data=(x_tr8, y_tr8), validation_freq=1)

model.summary()

# eval_loss, eval_acc = model.evaluate(x_tr3,  y_tr3, verbose=1)

# def eva(model,xtr,ytr):
#     len1 = len(ytr)//100
#     len2 = len1//4
#     po = 0
#     p = np.zeros((len1,1))
#     for i in range(len1):
#         eval_loss, eval_acc = model.evaluate(xtr[100*i:100*i+100,:,:,:],  ytr[100*i:100*i+100,:], verbose=1)
#         if eval_acc>0.5:
#             p[i] = 1
#         else:
#             p[i] = 0
#     for i in range(len2):
#         temp = p[4*i]+p[4*i+1]+p[4*i+2]+p[4*i+3]
#         if temp==4:
#             po += 1
    
#     return po/len2

# re1 = eva(model, x_tr1,  y_tr1)
# re2 = eva(model, x_tr2,  y_tr2)
# re3 = eva(model, x_tr3,  y_tr3)
# re4 = eva(model, x_tr4,  y_tr4)
# re5 = eva(model, x_tr5,  y_tr5)
# re6 = eva(model, x_tr6,  y_tr6)
# re7 = eva(model, x_tr7,  y_tr7)
# re8 = eva(model, x_tr8,  y_tr8)

# print(re1,re2,re3,re4,re5,re6,re7,re8)

# show
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# print(acc)
# print(val_loss)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training loss')
# plt.plot(val_loss, label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()
# plt.show()

# # 存储模型
# keras_model_path = './room/mod_b202'
# model.save(keras_model_path)

# # 加载模型
# model1 = tf.keras.models.load_model(keras_model_path)

