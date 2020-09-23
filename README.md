# PyTorch-AdaIN-StyleTransfer
This project is an unofficial PyTorch implementation of the paper using Google Colab: [**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**](https://arxiv.org/abs/1703.06868)

All credit goes to: [Xun Huang](http://www.cs.cornell.edu/~xhuang/) and
[Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)


## Description
The paper implements a style transfer algorithm, which uses a fixed pretrained vgg19 (up to ReLU 4.1) to encode a style image and a content image. Then the style of the style image is transferred to the content image. The novel approach this paper describes uses an AdaIN layer. This layer first normalises the the content image to unit mean and standard deviation. After that the content image is scaled such that it's mean and standard deviation are equal to the mean and standard deviation of the style image. Then the image is decoded using a decoder that mirrors the vgg19.
<p align='center'>
  <img src='examples/architecture.jpg' width="600px">
</p>

## Requirements
- A google drive account to run the notebooks.
- A pretrained vgg19 pth file. I used the file provided by [Naoto Inoue](https://github.com/naoto0804/pytorch-AdaIN) in his implementation of the same paper. Link: [vgg_normalised.pth](https://drive.google.com/file/d/108uza-dsmwvbW2zv-G73jtVcMU_2Nb7Y/view).
To train:
- [2015 Coco Image Dataset, 13GB](http://images.cocodataset.org/zips/test2015.zip)
- [WikiArt Dataset, 25.4GB](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip)
