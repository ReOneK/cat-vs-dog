from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from keras import models


model=load_model('small_dogvscat.h5')
model.summary()

img_path='1.jpg'
img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255.
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()


layer_output=[layer.output for layer in model.layers[:6]]
activate_model=models.Model(inputs=model.input,outputs=layer_output)
activatios=activate_model.predict(img_tensor)



#可视化第一个卷积层的激活
first_layer_activation=activatios[0]
plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')