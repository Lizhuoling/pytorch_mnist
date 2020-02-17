Author: Zhuoling Li  
Date: 2020.2.17  

# Environment: 
Ubuntu17+Python2+Pytorch

# Attention: 
Before running the project, please install the used libraries, such as numpy, pytorch and opencv.

# Running step:
1.Run the file "download_data" for downloading the mnist dataset. You can use the following codes:
'''  
sudo chmod +x download_data  
./download_data
'''  
2.Run the file "prepare_mnist.py" for preprocessing the data. You can use the following code:  
  python prepare_mnist.py  
3.Run the file "train.py" for training and saving the deep learning model. You can use the following code:  
  python train.py  
4.Run the file "validation.py" for validating the performance of the trained model. You can use the following code:  
 python validation.py  
