import os
from tqdm import tqdm
from Dataloader import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torch.utils.data import random_split
from torchsummary import summary
from Loss import *
from torch.utils.tensorboard.writer import SummaryWriter
from EncDec import EncDec

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def train_net(model, logger, hyper_parameters, modeltype, device, loss_function, dataloader_train, dataloader_validation, dataloader_test, directory):

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


            images, labels = batch

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
    accuracy = check_accuracy(model, dataloader_test, device, hyper_parameters['batch size'])
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

    with torch.no_grad():
        for data in dataloader:
        
            image, label = data

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
        transforms.ToImage(),                          # Replace deprecated ToTensor()    
        transforms.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    ]
)

drive_dataset = DRIVE(transform=transform)
print(f"The dataset has {len(drive_dataset)} images.")
drive_train_size = int(0.7 * len(drive_dataset))
drive_val_size = int(0.2 * len(drive_dataset))
drive_test_size = len(drive_dataset) - drive_train_size - drive_val_size

drive_train, drive_val, drive_test = random_split(
    drive_dataset, [drive_train_size, drive_val_size, drive_test_size])

print(f"Created a new Dataset for training of length: {len(drive_train)}")
print(f"Created a new Dataset for validation of length: {len(drive_val)}")
print(f"Created a new Dataset for testing of length: {len(drive_test)}")

model = EncDec().to(device)
summary(model, (3, 256, 256))

print("Current working directory:", os.getcwd())

run_dir = "Torch_model"
os.makedirs(run_dir, exist_ok=True)


hyperparameters = {
    'batch size': 4, 
    'step size': 20, 
    'learning rate': 0.001, 
    'epochs': 160, 
    'gamma': 0.5, 
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

print(f"Created a new Dataset for training of length: {len(drive_train)}")
print(f"Created a new Dataset for validation of length: {len(drive_val)}")
print(f"Created a new Dataset for testing of length: {len(drive_test)}")


modeltype = hyperparameters['backbone']
modeltype_directory = os.path.join(run_dir, f'{modeltype}')

# Initialize model, optimizer, scheduler, logger, dataloader
dataloader_train = DataLoader(
    drive_train, batch_size=hyperparameters["batch size"], shuffle=True, num_workers=hyperparameters["number of workers"], drop_last=True)
print(f"Created a new Dataloader for training with batch size: {hyperparameters['batch size']}")
dataloader_validation = DataLoader(
    drive_val, batch_size=hyperparameters["batch size"], shuffle=False, num_workers=hyperparameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for validation with batch size: {hyperparameters['batch size']}")
dataloader_test = DataLoader(
    drive_test, batch_size=1, shuffle=False, num_workers=hyperparameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for testing with batch size: {hyperparameters['batch size']}")

log_dir = os.path.join(modeltype_directory, f'{hyperparameters["network name"]}_{hyperparameters["optimizer"]}_Scheduler_{hyperparameters["scheduler"]}')
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(log_dir)


accuracy = train_net(model, logger, hyperparameters, hyperparameters['backbone'], device,
                             loss_function, dataloader_train, dataloader_validation, dataloader_test, log_dir)



# Just to save a sample image and its prediction
import matplotlib.pyplot as plt

model.eval()
i, l = next(iter(dataloader_test))
out = model(i.to(device))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(out[0].cpu().detach().numpy()[0], cmap='gray')
axes[0].axis('off')
axes[0].set_title("Output image", fontweight='bold')
axes[1].imshow(l.squeeze().cpu().detach().numpy(), cmap='gray')
axes[1].axis('off')
axes[1].set_title("Ground truth image", fontweight='bold')
plt.savefig("combined_sample_DRIVE_enc_dec.png")
