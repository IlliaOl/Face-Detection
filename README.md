# Face-Detection
Tensorflow implementation of LeNet-5 archtecture trained for a face detection task.

Trained network is then used to detect in real time whether there is a human in front of a camera.

This model reaches 78.3% accuracy.


### LeNet-5

#### Authors
Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner

#### Abstract

Multilayer Neural Networks trained with the backpropagation algorithm constitute the best example of a successful
GradientBased Learning technique Given an appropriate network architecture, GradientBased Learning algorithms can be used to synthesize a complex decision surface that can classify highdimensional patterns such as handwritten characters, with minimal preprocessing This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task Convolutional Neural Networksm that are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques Reallife document recognition systems are composed of multiple modules including field extraction, segmentation, recognition, and language modeling A new learning paradigm, called Graph Transformer Networks (GTN), allows such multimodule systems to be trained globally using GradientBased methods so as to minimize an overall performance measure Two systems for online handwriting recognition are described Experiments demonstrate the advantage of global training, and the 	exibility of Graph Transformer Networks A Graph Transformer Network for reading bank check is also described It uses Convolutional Neural Network character recognizers combined with global training techniques to provides record accuracy on business and personal checks It is deployed commercially and reads several million checks per day.

[[paper]](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) [[code]](https://github.com/IlliaOl/Face-Detection/blob/main/FD/fd.py)

#### Run Example
``` 
$ cd FD
$ python3 api.py
```
