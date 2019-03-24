from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from keras import models


#载入模型
model=load_model('small_dogvscat.h5')


#载入图片，并将图片转化为张量（1，size,size,nums_features）
img_path='1.jpg'
img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255.
print(img_tensor.shape)


#因为模型在Dense之前只有6层，所以值输出前六层的可视化
layer_output=[layer.output for layer in model.layers[:6]]
activate_model=models.Model(inputs=model.input,outputs=layer_output)

#activations返回8个numpy数组组成的列表，每个层激活对应一个numpy数组
activatios=activate_model.predict(img_tensor)


#保存每一层的名字，用以作图
layer_names=[]
for layer in model.layers[:6]:
    layer_names.append(layer.name)

#每一行展示多少个特征图
per_row_show_nums=10


for layer_name,layer_activation in zip(layer_names,activatios):
    n_features=layer_activation.shape[-1]

    size=layer_activation.shape[1]

    n_cols=n_features//per_row_show_nums

    display_grid=np.zeros((size*n_cols,per_row_show_nums*size))

    for col in range(n_cols):
        for row in range(per_row_show_nums):
            channel_image=layer_activation[0,:,:,col*per_row_show_nums+row]
#对特征进行处理，使其看起来更美观
            channel_image-=channel_image.mean()
            channel_image/=channel_image.std()
#           channel_image*=255
            channel_image*=64
            channel_image+=128
            channel_image=np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image

    scale=1./size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))

    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid,aspect='auto',cmap='viridis')