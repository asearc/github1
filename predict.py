
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

start = time.perf_counter()

im_height = 224
im_width = 224
batch_size = 16
epochs = 10

image_path = "./datasets/"
# train_dir = "./datasets/train"
validation_dir = "./datasets/test"

# 定义训练集图像生成器，并进行图像增强
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,  # 旋转范围
                                           width_shift_range=0.2,  # 水平平移范围
                                           height_shift_range=0.2,  # 垂直平移范围
                                           shear_range=0.2,  # 剪切变换的程度
                                           zoom_range=0.2,  # 剪切变换的程度
                                           horizontal_flip=True,  # 水平翻转
                                           fill_mode='nearest')


train_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                           batch_size=batch_size,  # 一次训练所选取的样本数
                                                           shuffle=True,  # 打乱标签
                                                           target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                           class_mode='categorical')  # one-hot编码

total_train = train_data_gen.n

validation_image_generator = ImageDataGenerator(rescale=1. / 255)

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,  # 从验证集路径读取图片
                                                              batch_size=batch_size,  # 一次训练所选取的样本数
                                                              shuffle=False,  # 不打乱标签
                                                              target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                              class_mode='categorical')  # one-hot编码


total_val = val_data_gen.n

end = time.perf_counter()
print('读取图片时长: %s Seconds' % (end - start))

start = time.perf_counter()
# Restore the weights
model = tf.keras.Sequential()
covn_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
model.add(tf.keras.layers.Dense(512, activation='relu'))  # 添加全连接层
model.add(tf.keras.layers.Dropout(rate=0.5))  # 添加Dropout层，防止过拟合
model.add(tf.keras.layers.Dense(2, activation='softmax'))  # 添加输出层(2分类)

model.load_weights('./save_weights/myNASNetMobile.ckpt')

end = time.perf_counter()

#  获取数据集的类别编码
class_indices = train_data_gen.class_indices
# 将编码和对应的类别存入字典
inverse_dict = dict((val, key) for key, val in class_indices.items())

img = Image.open("./dog1.jpg")

print("加载图片完成")

img = img.resize((224, 224))

img1 = np.array(img) / 255.

img1 = (np.expand_dims(img1, 0))
# 将预测结果转化为概率值
result = np.squeeze(model.predict(img1))
predict_class = np.argmax(result)
print("预测结果为", inverse_dict[int(predict_class)])
print("预测概率为", result[predict_class])

plt.title([inverse_dict[int(predict_class)], result[predict_class]])
plt.imshow(img)
plt.show()
