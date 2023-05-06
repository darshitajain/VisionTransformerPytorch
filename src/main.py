
# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+
import matplotlib.pyplot as plt
import numpy as np
import utils, model, train_functions
import torch
import torchvision
from torchinfo import summary
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torchvision
from torchinfo import summary

# setting the hyper parameters
import argparse

parser = argparse.ArgumentParser(description="Hyperparameters for training ViT on CIFAR-10")
#parser.add_argument('-f')
parser.add_argument("--img_size", default="32", type=int)
parser.add_argument("--patch_size", default="4", type=int)
parser.add_argument("--batch_size", default="512", type=int)
parser.add_argument("--mlp_size", default="512", type=int)
parser.add_argument("--emb_dim", default="512", type=int)
parser.add_argument("--lr", default="1e-4", type=float)
parser.add_argument("--num_heads", default="8", type=int)
parser.add_argument("--num_trans_layer", default="6", type=int)
parser.add_argument("--n_epochs", default="10", type=int)
parser.add_argument("--wandb_flag", default=False, type=bool)

args = parser.parse_args()

#print(args.__str__())

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using CIFAR-10 dataset from torchvision
IMG_SIZE = args.img_size #224 #Mentioned in Table 3 of ViT paper

# Create transform pipeline manually 

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform_train,
    target_transform=None
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform_test
)

class_names = train_data.classes

BATCH_SIZE = args.batch_size

# Create Dataloader

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=4)

print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"Train features batch shape: {train_features_batch.shape}, train labels batch shape: {train_labels_batch.shape}")

# Create a model instance
vit = model.ViT(img_size=args.img_size, 
                 in_channels=3,
                 patch_size=args.patch_size,
                 num_transformer_layers=args.num_trans_layer,
                 embedding_dim=args.emb_dim,
                 mlp_size=args.mlp_size, 
                 num_heads=args.num_heads, 
                 attn_dropout=0.1, 
                 mlp_dropout=0.1, 
                 embedding_dropout=0.1, 
                 num_classes=len(class_names))


# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
optimizer = torch.optim.Adam(params=vit.parameters(), 
                             lr= args.lr) 
                             
# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

# Set the seeds
utils.set_seeds()

# Train the model and save the training results to a dictionary
results = train_functions.train(model=vit,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=args.n_epochs,
                       device=device,
                       args=args)

utils.plot_loss_curves(results)
plt.savefig("plots.png")
