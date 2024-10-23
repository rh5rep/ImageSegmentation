import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
from Dataloader import *
from torch.utils.data import ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from IPython.display import clear_output
from Loss import *
from torch.utils.tensorboard.writer import SummaryWriter
# from HyperParameterSearch import *
from EncDec import EncDec
import json

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def train_net(model, logger, hyper_parameters, modeltype, device, loss_function, dataloader_train, dataloader_validation, directory):

    optimizer, scheduler = set_optimizer_and_scheduler(hyper_parameters, model)


    epochs = hyper_parameters["epochs"]
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    validation_loss = 0

    images,labels = [],[]

    for epoch in range(epochs):  # loop over the dataset multiple times

        """    Train step for one batch of data    """
        training_loop = create_tqdm_bar(
            dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')

        training_loss = 0
        model.train()  # Set the model to training mode
        train_losses = []
        accuracies = []
        
        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the parameter gradients for the current minibatch iteration

            if len(batch) == 3:
                images, labels = batch[0], batch[1]
            else:
                images, labels = batch
            labels = labels.type(torch.LongTensor)

            labels = labels.to(device)
            images = images.to(device)

            # Forward pass, backward pass and optimizer step
            predicted_labels = model(images)
            loss_train = loss_function(labels, predicted_labels)
            loss_train.backward()
            optimizer.step()

            # Accumulate the loss and calculate the accuracy of predictions
            training_loss += loss_train.item()
            train_losses.append(loss_train.item())

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'Train loss', loss_train.item(
            ), epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'Train accuracy', train_accuracy, epoch*len(dataloader_train)+train_iteration)
        all_train_losses.append(sum(train_losses)/len(train_losses))
        all_accuracies.append(sum(accuracies)/len(accuracies)) 

        """    Validation step for one batch of data    """
        val_loop = create_tqdm_bar(
            dataloader_validation, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
        val_losses = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                
                if len(batch) == 3:
                    images, labels = batch[0], batch[1]
                else:
                    images, labels = batch
                    
                labels = labels.type(torch.LongTensor)

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                output = model(images)

                # Calculate the loss
                loss_val = loss_function(labels, output)

                validation_loss += loss_val.item()
                val_losses.append(loss_val.item())

                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            all_val_losses.append(sum(val_losses)/len(val_losses))

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        logger.add_scalars(f'Combined', {'Validation loss': validation_loss,
                                                 'Train loss': training_loss/len(dataloader_train)}, epoch)
        if scheduler is not None:
            scheduler.step()
            print(f"Current learning rate: {scheduler.get_last_lr()}")

    if scheduler is not None:
        logger.add_hparams(
            {f"Step_size": scheduler.step_size, f'Batch_size': hyper_parameters["batch size"], f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"]},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    else:
        logger.add_hparams(
            {f"Step_size": "None", f'Batch_size': hyper_parameters["batch size"], f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"]},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    
    
    # Check accuracy and save model
    accuracy = check_accuracy(model, dataloader_validation, device, hyper_parameters['batch size'])
    save_dir = os.path.join(directory, f'accuracy_{accuracy:.3f}.pth')
    torch.save(model.state_dict(), save_dir)  # type: ignore

    return accuracy

def set_optimizer_and_scheduler(new_hp, model):
    if new_hp["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=new_hp["learning rate"],
                                     betas=(new_hp["beta1"],
                                            new_hp["beta2"]),
                                     weight_decay=new_hp["weight decay"],
                                     eps=new_hp["epsilon"])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=new_hp["learning rate"],
                                    momentum=new_hp["momentum"],
                                    weight_decay=new_hp["weight decay"])
    if new_hp["scheduler"] == "Yes":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
    else:
        scheduler = None
    return optimizer, scheduler

def check_accuracy(model, dataloader, device, batch_size):
    model.eval()
    num_correct = 0
    num_pixels = 0
    y_true = []
    y_pred = []

    image,label = [],[]

    with torch.no_grad():
        for data in dataloader:
            
            if len(data) == 3:
                image, label = data[0], data[1]
            else:
                image, label = data
            label = label.type(torch.LongTensor)

            image = image.to(device)
            label = label.to(device)

            probs = F.sigmoid(model(image))

            predictions = (probs > 0.5).float()
            # Flatten tensors to compare pixel by pixel
            predictions = predictions.view(-1)
            label = label.view(-1)

            # Accumulate the number of correct pixels
            num_correct += (predictions == label).sum().item()

            # Accumulate the total number of pixels
            num_pixels += label.numel()

    # Calculate accuracy
    accuracy = num_correct / num_pixels
    
    print(
        f"Got {num_correct}/{num_pixels} with accuracy {accuracy * 100:.3f}%\n\n")
    model.train()
    return accuracy



if torch.cuda.is_available():
    print("This code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

drive_dataset = DRIVE(train=True, transform=transform)
drive_train_size = int(0.8 * len(drive_dataset))
drive_val_size = len(drive_dataset) - drive_train_size
drive_train, drive_val = random_split(drive_dataset, [drive_train_size, drive_val_size])
drive_test = DRIVE(train=False, transform=transform)

ph2_dataset = PH2(train=True, transform=transform)
ph2_train_size = int(0.8 * len(ph2_dataset))
ph2_val_size = len(ph2_dataset) - ph2_train_size
ph2_train, ph2_val = random_split(ph2_dataset, [ph2_train_size, ph2_val_size])
ph2_test = PH2(train=False, transform=transform)                    


model = EncDec().to(device)
summary(model, (3, 256, 256))

print("Current working directory:", os.getcwd())

run_dir = "Torch_model"
os.makedirs(run_dir, exist_ok=True)



# Define the loss function
loss_function = bce_loss
results = {}

hyperparameters = {
    'batch size': 1, 
    'step size': 5, 
    'learning rate': 0.001, 
    'epochs': 20, 
    'gamma': 0.9, 
    'momentum': 0.9, 
    'optimizer': 'Adam', 
    'number of classes': 2, 
    'device': 'cuda', 
    'image size': (256, 256), 
    'backbone': 'SimpleEncDec', 
    'torch home': 'TorchvisionModels', 
    'network name': 'Test-0', 
    'beta1': 0.9, 
    'beta2': 0.999, 
    'epsilon': 1e-08, 
    'number of workers': 3, 
    'weight decay': 0.0005, 
    'scheduler': 'Yes'
}

loss_function = bce_loss

trainset = ConcatDataset([drive_train, ph2_train])
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=hyperparameters['batch size'], shuffle=True)

valset = ConcatDataset([drive_val, ph2_val])
# val_loader = torch.utils.data.DataLoader(valset, batch_size=hyperparameters['batch size'], shuffle=False)

testset = ConcatDataset([drive_test, ph2_test])
# test_loader = torch.utils.data.DataLoader(testset, batch_size=hyperparameters['batch size'], shuffle=False)

print(f"Created a new Dataset for training of length: {len(trainset)}")
print(f"Created a new Dataset for validation of length: {len(valset)}")
print(f"Created a new Dataset for testing of length: {len(testset)}")


modeltype = hyperparameters['backbone']
modeltype_directory = os.path.join(run_dir, f'{modeltype}')

# Initialize model, optimizer, scheduler, logger, dataloader
dataloader_train = DataLoader(
    trainset, batch_size=hyperparameters["batch size"], shuffle=True, num_workers=hyperparameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for training with batch size: {hyperparameters['batch size']}")
dataloader_validation = DataLoader(
    valset, batch_size=hyperparameters["batch size"], shuffle=False, num_workers=hyperparameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for validation with batch size: {hyperparameters['batch size']}")
dataloader_test = DataLoader(
    testset, batch_size=hyperparameters["batch size"], shuffle=False, num_workers=hyperparameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for testing with batch size: {hyperparameters['batch size']}")

log_dir = os.path.join(modeltype_directory, f'{hyperparameters["network name"]}_{hyperparameters["optimizer"]}_Scheduler_{hyperparameters["scheduler"]}')
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(log_dir)


accuracy = train_net(model, logger, hyperparameters, hyperparameters['backbone'], device,
                             loss_function, dataloader_train, dataloader_validation, log_dir)
print(f"Accuracy on validation set {accuracy}")



