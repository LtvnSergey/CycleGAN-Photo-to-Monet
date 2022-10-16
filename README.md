# CycleGAN-Photo-to-Monet


<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Claude-monet-le-bassin-aux-nympheas-reflets-verts.jpeg" height="200" width="1000">

# PROJECT OVERVIEW
Convert photo images into Monet-like paintings using CycleGAN network

Project goals:
1. Build, train and tune CycleGAN neural network for style transfer
2. Create web app using Gradio
3. Create demo
4. Create Docker image  

## Contents
- [Training](#training)
- [Results](#results)
- [Web app](#web-app)
- [Demo](#demo)
- [Docker](#docker)
- [Modules and tools](#modules-and-tools)


### Training
- Training dataset contents 300 Monet paintings and 7000 photos size of 256x256 px
<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Screenshot%20from%202022-10-16%2013-18-15.png">

- During preprocessing input images were normalized 

- As a model CycleGAN neural network was used for training

- CycleGAN network was constructed from several blocks:
  
    1\. Generator Photo that generates photo-like images from input
    
    2\. Generator Monet that generates Monet-like images from input 
    
    3\. Discriminator photo - it should predict if input photo image was generated - then output '0' or it is real - in that case output '1'
    
    4\. Discriminator Monet - it should predict if input Monet image was generated - then output '0' or it is real - in that case output '1'


- The main puprpose of generators is to produce monet style painting from photo input for monet-generator and vice verse for photo-generator

- The main purpose of discriminator is to correctly identify if input image was generated or is it real


- Scheme of CycleGAN training:

<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Screenshot%20from%202022-10-16%2013-05-10.png">

- After each training epoch losses were calculated:

    1\. Identity Loss for photo-generator and monet-generator
    
    2\. Adversarial Loss for photo-generator and monet-generator
    
    3\. Cycle Loss for photo-generator and monet-generator

    4\. Dicriminator Loss for photo-discriminator and monet-discriminator
    
    
- Neural Network was tuned with following hyper-parameters:
 
    1\. Learning rate (final choise - 0.01 with decay to 0.001)
    
    2\. Optimizer type (final choice - Adam)
    
    3\. Optimizer momentums (final choice - first momentum=0.5, second = 0.999)
    
    4\. Number of epoches (final choice - 100)

 
 
- More information about data analysis and training process can be found in notebook:
[cylcegan-training.ipynb](https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/notebook/cyclegan-with-comments-and-explanation-pytorch.ipynb)

- Trained model:
[model.pt]


    
### Results 

- On the following plot you can see values of generators and discriminators losses during training:

<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Screenshot%20from%202022-10-16%2013-11-08.png">

- Example of generator prediction on photo images after training:

<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Screenshot%20from%202022-10-16%2013-10-48.png">


### Web App

- To run Gradio web app type the following command:

  ``` python3 app.py ```

- Go to '''http://0.0.0.0:7000''' address in browser

<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Screenshot%20from%202022-10-16%2015-55-10.png">

- Upload image or choose from examples and press 'Submit' to get the prediction

<img src="https://github.com/LtvnSergey/CycleGAN-Photo-to-Monet/blob/master/assets/images/Screenshot%20from%202022-10-16%2015-55-40.png">


### Docker

- Install Docker: [docker](https://docs.docker.com/get-docker/)

- Download Docker image from repository


- (Optionaly) Build Docker image:
  ``` docker build -t painter -f Docker/Dockerfile ```


- Run following command to execute docker container:

  ```  docker run -it -p 7000:7000 plasticglass/painter ``` then go to http://0.0.0.0:7000


### Demos


### Modules and Tools

#### Web Development:

Docker | Gradio

#### Python-CNN:

Python | Pandas | Numpy | Pillow | Torch | Torchvision
