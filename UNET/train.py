import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from utils import *
from model import UNET
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

MODEL_PATH = '/home/user1/UNET/data/data700/output/saved_model'
LOAD_MODEL = False
ROOT_DIR = '../data/data700'
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
BATCH_SIZE = 1
LEARNING_RATE = 0.0005
EPOCHS = 130

def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data): 
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)
    
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
        

def main():
    global epoch
    epoch = 0 # epoch is initially assigned to 0. If LOAD_MODEL is true then
              # epoch is set to the last value + 1. 
    LOSS_VALS = [] # Defining a list to store loss values after every epoch
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST),
    ]) 

    train_set = get_chestxray_data(
        split='train',
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
    )

    print('Data Loaded Successfully!')

    # Defining the model, optimizer and loss function
    unet = UNET(in_channels=1, classes=4).to(DEVICE).train()        ### UTSAV: we need 3 classes
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    # loss_function = nn.CrossEntropyLoss(ignore_index=255)
    loss_function = DiceLoss()
    # loss_function = nn.MSELoss()

    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")    

    #Training the model for every epoch. 
    for e in range(epoch, EPOCHS):
        print('Epoch:', e)
        loss_val = train_function(train_set, unet, optimizer, loss_function, DEVICE)
        LOSS_VALS.append(loss_val) 
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': LOSS_VALS
        }, MODEL_PATH)
        plt.plot(LOSS_VALS)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('/home/user1/UNET/data/data700/output/loss_curve.png')
        print("Epoch completed and model successfully saved!", LOSS_VALS[-1])
    return LOSS_VALS

if __name__ == '__main__':
    loss = main()
    plt.plot(loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('/home/user1/UNET/data/data700/output/final_loss_curve.png')