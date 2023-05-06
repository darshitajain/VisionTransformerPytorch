### Implementation of the paper- "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)"

#### Notebook created by- [Darshita Jain](https://www.linkedin.com/in/jain-darshita/)


---



#### **Before starting with the implementation a few points to note-**
* #### This notebook contains implementation of Vision transformer for CIFAR10 dataset. Please note that the original ViT base model is having 82M params which is huge for training on a comparatively small dataset like CIFAR 10.
* #### When I trained the CIFAR10 dataset directly using the same parameters as ViT base model, the model was highly overfitting on train dataset and loss values were becoming Nan. To reduce overfitting the three mains things that can be modified are:   

    1.   Reducing the model size (number of layers and number of neurons in each layer)
    2.   Increase dropout
    3.   Data augmentation.

* #### I started with reducing the parameters of the model by using the below hyperparameter setting- 
      config= {"img_size":"32",
              "patch_size":"4", 
              "batch_size":"512", 
              "mlp_size":"512", 
              "emb_dim":"512", 
              "lr":"0.0001", 
              "num_heads":"8", 
              "num_trans_layer":"6"}
#### as opposed to what is mentioned in table 1 and 3 (in below cells) of ViT paper. 

* #### Even after doing the above changes, the train and test loss were still Nan. Then I reduced the learning rate from 3e-3(mentioned in the paper) to 1e-4 and it started learning something. After training for few epochs the model started to overfit again as shown in the plot below.

![overfitting](https://user-images.githubusercontent.com/19747895/236636498-b41d37b3-e1e5-44d0-8581-2e5da7842028.png)


* #### To reduce overfitting, I increased the embedding dropout from 0.0 to 0.1 results for the same are shown below.

![plots_150](https://user-images.githubusercontent.com/19747895/236636728-4272641c-3a95-40ac-ac5e-4226ed910d95.png)


* #### The performance can be improved even more by increasing dropout, adding data augmentations, etc. For now this implementation is in a working state and can be extended for different datasets by doing appropriate hyperparameter modifications.

## Installation

Start with making a conda or pip virtual environment. All the dependencies are in the requirements.txt file. They can be installed manually or with the following command-


```bash
  pip install requirements.txt
```

Usage example-

```bash
  cd src/
  python main.py --img_size 32 --patch_size 4 --wandb_flag True

  [or run with default settings]
  python main.py
```

Check all the available command line arguments in main.py file.

### Key learning from this implementation exercise- 
1. #### A working understanding of how to downscale large models for smaller dataset. 
2. #### Effect of different hyperparameters on reducing overfitting.

## References-
*  https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
*  https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model
*  https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular