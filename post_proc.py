from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
from utils import impath_to_image
from scipy import ndimage
import matplotlib.pyplot as plt
from keras_utils import get_dec_image
import numpy as np

def get_cluster_centres(codes,k=8):
    kmeans_model = KMeans(n_clusters=k, random_state=1)
    kmeans_model.fit(codes)
    return kmeans_model.cluster_centers_

def get_closest(sample,codes,names=None,limit=1):
    sample = sample.reshape(1,-1)
    arr = []                                                           
    for c in codes:
        c = c.reshape(1,-1)
        arr.append([cosine_distances(sample,c),euclidean_distances(sample,c)])
    cos_inds = np.array(arr)[:,0].reshape(-1).argsort()[:limit]
    euc_inds = np.array(arr)[:,1].reshape(-1).argsort()[:limit]
    if names:
        cos_names = [names[i] for i in cos_inds]
        euc_names = [names[i] for i in euc_inds]
        return cos_inds,euc_inds,cos_names,euc_names
    else:
        return cos_inds,euc_inds,None,None


def plot_images(img_names,font_path):
    """Function to get cluster centres and plot them"""
    n= len(img_names)
    m = len(img_names[0])
    plt.figure(figsize=(n*2, m*2))
    print (n,m)
    for i in range(n):
        for j in range(m):
            ax = plt.subplot(m, n, i + (n)*j+1)
            image = impath_to_image(font_path % img_names[i][j])
            plt.imshow(1 - image.reshape(28*2, 28*2),plt.cm.binary)
            # plt.title(img_names[i][j],fontsize=6)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.set_cmap('gray_r')
    plt.show()  

def plot_centres(codes,names,fpath='font_ims/%s.png',lim=3,k=8):
    """Function to get cluster centres and plot them"""
    centres = get_cluster_centres(codes,k)
    print ("Cluster Centers generated")
    res = []
    for cen in centres:
        res.append(get_closest(cen,codes,names,lim))
    img_names = [x[3] for x in res]
    plot_images(img_names,fpath)
    return res
