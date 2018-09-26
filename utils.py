# -*- coding: utf-8 -*-
import os
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
import glob
import re
import numpy as np

def impath_to_image(impath,grey=True):
    """
        Read an image path and return image as a numpy array.
    """
    return io.imread(impath,as_grey=grey)


def get_all_samples(font_dataset,thresh=False,char_vecs=False):
    """
        Get all samples in font_dataset. 
        if thresh is not False, images will be thresholded using thresh as threshold value.
    """
    DATA_SIZE = len(font_dataset)
    all_images = None
    all_vecs = None
    for i in range(DATA_SIZE):
        sample = font_dataset[i]
        img = sample['image'].reshape(1,1,font_dataset.im_height,font_dataset.im_width)
        if thresh:
            img = (img > thresh).astype(float)
        all_images = img if all_images is None else np.vstack((all_images,img))        

        if char_vecs:
            letter = sample['name'].split('_')[-1]
            vec = np.zeros((1,26))
            vec[ord(letter)-65]=1
            all_vecs = vec if all_vecs is None else np.vstack((all_vecs,vec))        
    if char_vecs:
        return all_images,all_vecs
    else:
        return all_images

class FontAlphabetsDataset(Dataset):
    """
        Create a Dataset class from from folder path.
    """
    def __init__(self, folder_path,info_path =  None,transform=None,custom_path=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = folder_path
        self.transform = transform
        self.im_width = 30
        self.im_height = 30
        path_pattern = '/*/single_alphabet/*.png' if custom_path is None else custom_path
        self.im_paths = glob.glob(folder_path+path_pattern)
        print ("Number of files:",len(self.im_paths))   
        if custom_path is None:
            self.names = [re.findall('/(.*)/single_alphabet',fname  )[0] for fname in self.im_paths]
        else:
            self.names = [os.path.splitext(os.path.basename(fname))[0] for fname in self.im_paths]
        self.set_size()

    def set_size(self):
        image = io.imread(self.im_paths[0],as_grey=True)
        self.im_height,self.im_width = image.shape

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        image = io.imread(self.im_paths[idx],as_grey=True)

        sample = {'image': image,
                  'name':self.names[idx],
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()