# FSBD
Few Shot Traffic Sign Detector for Bangladeshi Traffic signs. To implement our model and all others (CDFSOD,TFA,FRCN-ft) we needed FRCN. And we used [this git repo](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) to implement our FRCN. We are really thankful for all those extra comments and easy implementation by Yun Chen owner of that repository and all others who might have helped him.

## No dependencies or packages required
We used google colab as our development environment. And you don't need to install any extra dependencies or packages. That's why we could not provide you with better visualization methods. But we did it on purpose. Because when we were trying to implement different models lots of the time the error is in the extra dependencies or packages which got depracated. Another error was hard coded file path. We tried really hard to reduce hard coding of file path but there might be some. And if you find any we are sorry in advance.

## How to run our code
Since we used google colab, all the shell commands will be like google colab. We have 4 different models implemented in one repo. Thus we provide a controller file controller.py from where you can run any one of them with required specification.
As an example,
```
! python '/content/drive/MyDrive/FSBD/controller.py'
```
The above code is like that because we used google drive to store our files,weights and everything else. Or you can write like the below code if you don't want to use google drive (clearly a bad idea),
```
! python '/content/FSBD/controller.py'
```
At last you will find an interactive session where you are asked with different questions like,
  1. Want to train ? Write t if true or f if you want to test.
  2. Want to add cos ? Write t if you want to add instance level feature normalization. (Beware takes lot of time to train if cos added)
  3. Want to add proto ? Always write f because we implemented prototypical network but did not add it with any of our model because it heavily depends on its hyperparameter alpha in our case. For more clarification you can read paper on DeFRCN. 
  4. Give Number of shots ? We implemented for 1,2,3,5,10 shots.
  5. Want beta or alpha ? Implemented for our easy debugging. Sometimes we just need to run our model with only few samples then write b else write a.
  6. After that you are asked to give a model name and which weight you want to load to that model. Now from this more explanation is out of scopes of this readme file.

## More explanation
You will find more explanation if you read the controller.py file. There we have implemented with only basic if-else. Looks very bad but easy to understand. Again you might need to visit config.py file inside utils folder for more clarification about the file path.

The flow of our implementation is quite like the above given repo by Yun Chen. We have files inside dataset like Fsbd-1-tn.txt containing all the sample image file names in 5 way 1 shot setting. dataset.py inside data folder read this file names and collect them and provide them to our specific models we specified inside controller.py file. 

We really tried to keep our codebase easy to understand because all of us faced hard time understanding others. But after implementing and integrating everything we found that keeping your codebase easy to understand, is very hard. But we will urge you to go through all the files and you will definitely understand whatever we were doing there.

## Result
At last our result of running FSBD on Bangladeshi traffic Signs with all others is,
