"""
将图像复制到训练，验证和测试的目录
"""

import os,shutil


all_dataset_dir='D:/pic/dogvscat/train'

base_dir='D:/pic/small_dogvscat'
os.mkdir(base_dir)
#划分训练集，验证集，测试集
train_dir=os.path.join(base_dir,'train')
os.mkdir(train_dir)

validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)

test_dir=os.path.join(base_dir,'test')
os.mkdir(test_dir)

#分别划分猫狗的训练集，验证集，测试集
train_cats_dir=os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)

train_dogs_dir=os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

val_cats_dir=os.path.join(validation_dir,'cats')
os.mkdir(val_cats_dir)

val_dogs_dir=os.path.join(validation_dir,'dogs')
os.mkdir(val_dogs_dir)

test_cats_dir=os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)

test_dogs_dir=os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

#猫的训练集
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:
    src=os.path.join(all_dataset_dir,fname)
    dst=os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

# 猫的验证集
fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]

for fname in fnames:
    src=os.path.join(all_dataset_dir,fname)
    dst=os.path.join(val_cats_dir,fname)
    shutil.copyfile(src,dst)

#猫的测试集
fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:
    src=os.path.join(all_dataset_dir,fname)
    dst=os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)


#狗的训练集
fnames=['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:
    src=os.path.join(all_dataset_dir,fname)
    dst=os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]

for fname in fnames:
    src=os.path.join(all_dataset_dir,fname)
    dst=os.path.join(val_dogs_dir,fname)
    shutil.copyfile(src,dst)


fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]

for fname in fnames:
    src=os.path.join(all_dataset_dir,fname)
    dst=os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)






