# Keras
keras_application


主要对于kaggle中经典数据集cat vs dog 数据集进行图像分类

    
包括：

    --数据增强（在小数据集上防止过拟合）
        
    --使用预训练模型
            
          ==特征提取预训练模式
            
          ==微调模型（fine-tuning）
            
    --可视化（visualsition）
        
          ==中间激活层的可视化
            
            ==卷积核的可视化
            
            ==热力图的可视化
            
 
选取了原数据集中的猫狗图片各500张，其中训练集1000，测试集以及验证集分别500张

first

使用原始数据集进行训练，结果
        

![loss](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/abc1.png)
        
        

![acc](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/abc2.png)
        
        
        
        可以明显的看出产生了过拟合，因此可以采用迁移学习或者数据增强进行改进
        
secend
        
使用数据增强，结果
        

![loss](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/data_Argu1.png)
    
    
![acc](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/data_Argy2.png)
        
        
        过拟合明显被降低了，说明数据增强在小数据集上有防止过拟合的效果
        
third
        
使用迁移学习
        
 ![loss](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/pretrained1.png)
        
 ![acc](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/pretrained2.png)
        
        
防止过拟合的效果也非常不错
        

Some Tips:    
  --  神经网络的核心组件是层（layer），它是一种数据处理模块，你可以将它看成数据过滤器。
    进去一些数据，出来的数据变得更加有用。具体来说，层从输入数据中提取表示——我们期望这种
    表示有助于解决手头的问题。大多数深度学习都是将简单的层链接起来，从而实现渐进式的数据
    蒸馏（data distillation）。深度学习模型就像是数据处理的筛子，包含一系列越来越精细的
    数据过滤器（即层）。


     
  --卷积神经网络学到的模式具有平移不变性（translation invariant）
   
  --卷积神经网络可以学到模式的空间层次结构（spatial hierarchies of patterns）
    
        ==视觉世界从根本上具有平移不变性
        
        ==视觉世界从根本上具有空间层次结构
    
   
  --深度神经网络可以有效地作为信息蒸馏管道（information distillation pipeline），
    输入原始数据（本例中是RGB 图像），反复对其进行变换，将无关信息过滤掉（比如图像的具体外观），
    并放大和细化有用的信息（比如图像的类别）。
     
     
  --卷积神经网络中每一层都学习一组过滤器，以便将其输入表示为过滤器的组合。这类似于傅里叶变换将信号分解为一
    组余弦函数的过程。随着层数的加深，卷积神经网络中的过滤器变得越来越复杂，越来越精细。
   
