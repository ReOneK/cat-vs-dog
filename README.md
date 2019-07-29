# Keras
keras_application


Image classification was mainly performed for the classic dataset cat vs. dog dataset in kaggle
===========
    
include：

-- data enhancement (preventing overfitting on small data sets)
        
-- use the pre-training model
            
    == pre-training mode of feature extraction
            
    == fine tuning model (fine-tuning)
            
-- visualsition
        
    == visualization of the middle activation layer
            
    == visualization of the convolution kernel
            
    == visualization of thermal diagram
            
 dataset:
500 pictures of cats and dogs in the original data set were selected, including 1000 training sets, 500 test sets and 500 verification sets
选取了原数据集中的猫狗图片各500张，其中训练集1000，测试集以及验证集分别500张

first


Use the original training set   
-------------
![](https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/abc1.png)     
        
<img width="200" height="200" src="https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/abc2.png"/>  

  
problem：It can be clearly seen that overfitting has been generated, so migration learning or data enhancement can be adopted for improvement      
        
secend
 ------      
data_argument
        
<img width="200" height="200" src="https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/data_Argu1.png"/>      
    
<img width="200" height="200" src="https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/data_Argy2.png"/> 

result:Overfitting is obviously reduced, indicating that data enhancement has the effect of preventing overfitting on small data sets
        
       
        
third
------        
Find-tuning
        
 <img width="200" height="200" src="https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/pretrained1.png"/>       

 <img width="200" height="200" src="https://github.com/ReOneK/Cat-vs-Dog/blob/master/pic/pretrained2.png"/>  
        

Some Tips:  

The core component of a neural network is the layer, a data processing module that you can think of as a data filter.
You put in some data, and the data that comes out becomes more useful.Specifically, layers extract representations from input data - we expect that
To help solve the problem at hand.Most deep learning links simple layers to implement incremental data
Data cut.The deep learning model is like a data processing sieve, containing a series of increasingly sophisticated
Data filters (layers).
     
-- the convolutional neural network learns that patterns have translation invariant
   
-- the convolution model of neural network can learn space hierarchy (spatial hierarchies of patterns)
    
The visual world is fundamentally translational invariant
        
Visual world has spatial hierarchy fundamentally
    
   
-- deep neural networks can effectively serve as a "information distillation pipeline".
Input the original data (in this case, the RGB image), transform it repeatedly and filter out irrelevant information (such as the specific appearance of the image).
And zoom in and refine useful information (such as the categories of images).
     
     
Each layer in the convolutional neural network learns a set of filters in order to represent its input as a combination of filters.It's like the Fourier transform decomposing the signal into one
Set of cosine functions.As the number of layers increases, the filters in the convolutional neural network become more and more complex and sophisticated.
  
   
