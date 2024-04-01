import os
import torch
from model import UNET
from utils import *
from utils import save_as_images
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
# from labels import trainId2label as t2l

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = 'cpu'
    print('Running on the CPU')

ROOT_DIR_CITYSCAPES = '/home/user1/UNET/data'
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

MODEL_PATH = "/home/user1/UNET/data/data700/output/saved_model"

EVAL = True 
PLOT_LOSS = False

def save_predictions(data, model): 
    with open('/home/user1/UNET/data/data700/results/dice.txt', 'w') as f:   
        model.eval()
        loss_list=[]
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data)):
                X, y, s = batch # here 's' is the name of the file stored in the root directory
                X, y = X.to(device), y.to(device)
                predictions = model(X) 

                predictions = torch.nn.functional.softmax(predictions, dim=1)
                # print(f'The shape of the predictions is {predictions.shape}')
                Loss= DiceLoss()
                loss = 1 - Loss(predictions, y)
                print(f'loss is {loss}')
                loss_list.append(loss)
                f.write(f'{loss}\n')
                pred_labels = torch.argmax(predictions, dim=1) 
                pred_labels = pred_labels.float()
                pred_labels = pred_labels * 0.3

                # save_image(pred_labels, os.path.join('/home/user1/UNET/val_output', str(s[0])))
                # Remapping the labels
                pred_labels = pred_labels.to('cpu')
                # pred_labels.apply_(lambda x: t2l[x].id)
                pred_labels = pred_labels.to(device)   

                # Resizing predicted images too original size
                pred_labels = transforms.Resize((1024, 1024))(pred_labels)             
                
                # Configure filename & location to save predictions as images
                s = str(s[0])
                # pos = s.rfind('/', 0, len(s))
                # name = s[pos+1:-18]  
                # name = s.split('.')[0]
                # global location
                # location = '/home/user1/UNET/val_output'
                # save_as_images(pred_labels, location, name)  
        avg_dice = sum(loss_list)/len(loss_list)
        f.write(f'Avg Dice score is {avg_dice}')
    f.close()              

def evaluate(path):
    T = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.NEAREST)
    ])

    val_set = get_chestxray_data(
        root_dir=ROOT_DIR_CITYSCAPES,
        split='val',
        eval = True
    )
 
    print('Data has been loaded!')

    net = UNET(in_channels=1, classes=4).to(device)
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'{path} has been loaded and initialized')
    save_predictions(val_set, net)

def plot_losses(path):
    checkpoint = torch.load(path)
    losses = checkpoint['loss_values']
    epoch = checkpoint['epoch']
    epoch_list = list(range(epoch))

    plt.plot(epoch_list, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss over {epoch+1} epoch/s")
    plt.show()

if __name__ == '__main__':
    if EVAL:
        evaluate(MODEL_PATH)
    if PLOT_LOSS:
        plot_losses(MODEL_PATH)