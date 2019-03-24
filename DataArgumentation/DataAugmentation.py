from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt


data=ImageDataGenerator(
    rotation_range=40,           #表示随机旋转的角度范围
    width_shift_range=0.2,       #是图像在水平或垂直方向上平移的范围
    height_shift_range=0.2,
    shear_range=0.2,             #随机错且的角度
    zoom_range=0.2,             # 随机缩放
    horizontal_flip=True,       #随机将一半图像水平翻转
    fill_mode='nearest'         #填充像素
)


train_cats_dir='D:/pic/small_dogvscat/train/cats'
fnames=[os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)]
# 随机选取一张图片
img_path=fnames[3]
#读取图片并调整大小
img=image.load_img(img_path,target_size=(150,150))
#因为需要reshape,所以需要转化为数组
x=image.img_to_array(img)

x=x.reshape((1,)+x.shape)

i=0
#显示几张增强后的照片
for batch in data.flow(x,batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i=i+1
    if i%4==0:
        break

plt.show()
