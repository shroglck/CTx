# CTx
Code for CTx-Net for vedio actiton recognition under occlusion

### Installation

The code uses **Python 3.9** and it is tested on PyTorch GPU version 1.11, with CUDA-11.6

### Setup CTx-Net Virtual Environment

```
virtualenv --no-site-packages <your_home_dir>/.virtualenvs/CTx
source <your_home_dir>/.virtualenvs/CTx/bin/activate
```

### Clone the project and install requirements

```
git clone https://github.com/shroglck/CTx.git
cd CTx
pip install -r requirements.txt
```

## Download models

* Download pretrained CompNet weights from [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/akortyl1_jh_edu/EYH4UDvQnQ9Ettu7cBQAfZoBFLU0gZeredTmfUssMJCrKg?e=HqxXAs) and copy them inside the **models** folder.




 

#### Evaluate the classification performance of a model

Run the following command in the terminal to evaluate a model on the full test dataset:
```
python Code/test.py 
```


## Initializing CompositionalNet Parameters

We initialize CompositionalNets (i.e. the vMF kernels and mixture models) by clustering the training data. 
In particular, we initialize the vMF kernels by clustering the feature vectors:

```
python Initialization_Code/vMF_clustering.py
``` 

Furthermore, we initialize the mixture models by EM-type learning.
The initial cluster assignment for the EM-type learning is computed based on the similarity of the vMF encodings of the training images.
To compute the similarity matrices use:
 
```
python Initialization_Code/comptSimMat.py
``` 

As this process takes some time we provide Afterwards you can compute the initialization of the mixture models by executing:

```
python Initialization_Code/Learn_mix_model_vMF_view.py
```


## Acknowledgement 

This code has been adapted from
```
@inproceedings{CompNet:CVPR:2020,
  title = {Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion},
  author = {Kortylewski, Adam and He, Ju and Liu, Qing and and Yuille, Alan},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2020},
  month_numeric = {6}
}

@article{kortylewski2021compositional,
  title={Compositional convolutional neural networks: A robust and interpretable model for object recognition under occlusion},
  author={Kortylewski, Adam and Liu, Qing and Wang, Angtian and Sun, Yihong and Yuille, Alan},
  journal={International Journal of Computer Vision},
  volume={129},
  number={3},
  pages={736--760},
  year={2021},
  publisher={Springer}
}

```

