# PyTorch-AdaIN-StyleTransfer
This project is an unofficial PyTorch implementation of the paper: [**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**](https://arxiv.org/abs/1703.06868)

All credit goes to: [Xun Huang](http://www.cs.cornell.edu/~xhuang/) and
[Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)


## Description
The paper implements a style transfer algorithm, which uses a fixed pretrained vgg19 (up to ReLU 4.1) to encode a style image and a content image. Then the style of the style image is transferred to the content image. The novel approach this paper describes uses an AdaIN layer. This layer first normalises the the content image to unit mean and standard deviation. After that the content image is scaled such that it's mean and standard deviation are equal to the mean and standard deviation of the style image. Then the image is decoded using a decoder that mirrors the vgg19.
<p align='center'>
  <img src='examples/architecture.jpg' width="600px">
</p>

