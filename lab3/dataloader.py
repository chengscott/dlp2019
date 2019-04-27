import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms


def getData(mode):
  if mode == 'train':
    img = pd.read_csv('train_img.csv')
    label = pd.read_csv('train_label.csv')
    return np.squeeze(img.values), np.squeeze(label.values)

  img = pd.read_csv('test_img.csv')
  label = pd.read_csv('test_label.csv')
  return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(torch.utils.data.Dataset):
  def __init__(self, root, mode, device):
    """
    Args:
        root (string): Root path of the dataset.
        mode : Indicate procedure status(training or testing)

        self.img_name (string list): String list that store all image names.
        self.label (int or float list): Numerical list that store all ground truth label values.
    """
    self.root = root
    self.img_name, self.label = getData(mode)
    self.mode = mode
    self.device = device
    print('> Found', len(self.img_name), 'images')
    self.data_transform = {
        'train':
        transforms.Compose([
            #transforms.Resize(224),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test':
        transforms.Compose([
            #transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

  def __len__(self):
    """'return the size of dataset"""
    return len(self.img_name)

  def __getitem__(self, index):
    """
   step1. Get the image path from 'self.img_name' and load it.
          hint : path = root + self.img_name[index] + '.jpeg'
   
   step2. Get the ground truth label from self.label
             
   step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
          rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
               
          In the testing phase, if you have a normalization process during the training phase, you only need 
          to normalize the data. 
          
          hints : Convert the pixel value to [0, 1]
                  Transpose the image shape from [H, W, C] to [C, H, W]
                 
    step4. Return processed image and label
    """
    # load image from path
    path = os.path.join(self.root, self.img_name[index])
    image = Image.open(path + '.jpeg')
    # ground truth label
    label = torch.LongTensor([self.label[index]]).to(self.device)
    # transform
    #mean, std = np.mean(image, (0, 1)), np.std(image, (0, 1))
    image = self.data_transform[self.mode](image)
    #image = transforms.Normalize(mean, std)(image)
    image = transforms.Normalize([.0] * 3, [255.] * 3)(image)
    image = image.to(self.device)

    return image, label

def get_dataloaders(batch_size, device):
  return (utils.DataLoader(
      RetinopathyLoader('data', mode, args.device), batch_size, shuffle=True)
          for mode in ('train', 'test'))
