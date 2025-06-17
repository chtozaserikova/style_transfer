from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torch.autograd import Variable
import numpy as np
import torch

COCO2014_DATA_PATH = './data/coco2014/'
WIKIART_DATA_PATH = './data/wikiart/'
STYLE_PATH = './data/style'

WIKIART_STYLE_MAP = {
    'Baroque': 1,
    'Cubism': 2,
    'Early_Renaissance': 3,
    'Pointillism': 4,
    'Ukiyo_e': 5,
}

class COCO2014(Dataset):
    def __init__(self, split, max_files, transform=None):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split}")
        
        split = split + '2014'
        self.image_path = os.path.join(COCO2014_DATA_PATH, 'images', split)
        images = os.listdir(self.image_path)
        
        if len(images) > max_files:
            images = images[:max_files]
        
        self.images = images
        self.length = len(images)
        self.transform = transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.image_path, img_name)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img

class StyleDataset(Dataset):
    def __init__(self, ttv, transform=None):
        if ttv not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {ttv}")
        self.image_path = os.path.join(STYLE_PATH, ttv)
        self.images = os.listdir(self.image_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.image_path, img_name)).convert('RGB')

        img_style = [s for s in img_name.split('_') if s[0].isupper() or (s[0] == 'e' and len(s) == 1)]
        label = WIKIART_STYLE_MAP["_".join(img_style)]

        if self.transform:
            img = self.transform(img)
            
        return img, label
    
class StyleLoader():
    def __init__(self, style_size=64, style_folder='./data/style/train', cuda=True):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        self.cuda = cuda
    
    def get(self, i):
        idx = i % len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = self.tensor_load_rgbimage(filepath, self.style_size)    
        style = style.unsqueeze(0)
        style = self.preprocess_batch(style)
        if self.cuda:
            style = style.cuda()
        style_v = Variable(style, requires_grad=False)
        return style_v

    @staticmethod
    def preprocess_batch(batch):
        batch = batch.transpose(0, 1)
        (r, g, b) = torch.chunk(batch, 3)
        batch = torch.cat((b, g, r))
        batch = batch.transpose(0, 1)
        return batch

    @staticmethod
    def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            if keep_asp:
                size2 = int(size * 1.0 / img.size[0] * img.size[1])
                img = img.resize((size, size2), Image.ANTIALIAS)
            else:
                img = img.resize((size, size), Image.ANTIALIAS)

        elif scale is not None:
            img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
        img = np.array(img).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

def get_datasets(train_tf=None, val_tf=None):
    """Returns the train and val datasets with optional transform to apply to the datasets"""

    style_trainset = StyleDataset('train', transform=train_tf)
    style_validset = StyleDataset('val', transform=val_tf)
    style_testset = StyleDataset('test', transform=val_tf)
    # Set the size of the content sets to the length of the style sets
    content_trainset = COCO2014('train', len(style_trainset), train_tf)
    content_validset = COCO2014('val', len(style_validset), val_tf)
    content_testset = COCO2014('val', len(style_testset), val_tf)
    return content_trainset, content_validset, content_testset, style_trainset, style_validset, style_testset

def get_dataloaders(bs=64, train_tf=None, valid_tf=None):
    """Returns the train and val dataloaders for content and style datasets with optional transforms to apply to datasets and batch size 'bs'"""
    content_trainset, content_validset, content_testset, style_trainset, style_validset, style_testset = get_datasets(train_tf, valid_tf)
    
    content_trainloader = DataLoader(content_trainset, bs, shuffle=True)
    content_validloader = DataLoader(content_validset, bs, shuffle=False)
    content_testloader = DataLoader(content_testset, bs, shuffle=False)

    style_trainloader = DataLoader(style_trainset, bs, shuffle=True)
    style_validloader = DataLoader(style_validset, bs, shuffle=False)
    style_testloader = DataLoader(style_testset, bs, shuffle=False)

    return content_trainloader, content_validloader, content_testloader, style_trainloader, style_validloader, style_testloader