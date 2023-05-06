
"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time
import wandb 
import os

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance to train the model.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    (0.1111, 0.8888)
    """


    # Put model in train mode
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
  
        optimizer.zero_grad()

        loss.backward()

        # Gradient clipping to prevent gradient explosion and Nan values
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)

        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Average loss and accuracy per batch 
    #print(f"Inside train step, train loss before dividing with len of dataloader: {train_loss}, len(dataloader): {len(dataloader)}")
    train_loss = train_loss / len(dataloader)
    #print(f"Inside train step function, train loss:{train_loss}")
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    (0.0111, 0.8888)
    """
    # Put model in eval mode
    model.eval() 

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Average loss and accuracy per batch 
    #print(f"Inside test step, test loss before dividing with len of dataloader: {test_loss}, len(dataloader): {len(dataloader)}")
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          args) -> Dict[str, List]:

    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Wandb initialization if flag set to True.
    if args.wandb_flag:
      project_name = "{}_{}_{}_{}".format("vit",args.img_size, args.patch_size, args.n_epochs)
      wandb.init(project="ViT_CIFAR10",
            name=project_name)
      wandb.config.update(args)

    if args.wandb_flag:
      wandb.watch(model)

    # Make sure model on target device
    model.to(device)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        start = time.time()
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
        
        #scheduler.step(epoch-1) # step cosine scheduling
        scheduler.step() # step cosine scheduling


        #print(f"testing for Nan value train loss: {train_loss}, test_loss: {test_loss}")
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if args.wandb_flag:
          wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss, "train_accuracy": train_acc, "test_accuracy": test_acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

        if epoch%10==0:
          print('Saving..')
          state = {"model": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "results":results}
          if not os.path.isdir('checkpoint'):
           os.mkdir('checkpoint')
           torch.save(state, './checkpoint/'+'vit_epoch_{}'+'-ckpt.t7'.format(epoch))
        
    # writeout wandb
    if args.wandb_flag:
      wandb.save("wandb_vit.h5")
    
    # save the trained model
    torch.save(model.state_dict(), 'model.pth')

    # Return the filled results at the end of the epochs
    return results
