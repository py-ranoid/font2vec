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

def find_best(nums,codes,names,n=3):
    sample = codes[nums]
    sample_names = codes[nums]
    all_choices = []
    for samp,samp_name in sample:
        counter = 0
        choices = [samp_name]
        best = get_closest(samp,codes,limit=n+5)
        for x in best[2]:
            if x.startswith(samp_name[:5]):
                continue
            else:
                counter+=1
                choices.append(x)
            if counter==n:
                all_choices.append(choices)
                break
    plt.figure(figsize=(len(nums)*2, (n+1)*2))
    font_path='font_ims_56/%s.png'                
    for i in range(len(nums)):
        for j in range(n+1):
            print (i + (nums)*j+1)
            ax = plt.subplot(len(nums), n+1, j + len(nums)*i+1)
            im = impath_to_image(font_path % img_name)
            #plt.title('-'.join(["%0.2f" %x for x in weight]),fontsize=6)
            plt.imshow(im.reshape(28*2, 28*2),plt.cm.binary)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def compare_dims(sample,codes,decoder,all_names=None,approx=False):
    dims = 32
    nums = 6
    plt.figure(figsize=(nums*4, dims*4))
    counter = 0
    font_path='font_ims_56/%s.png'                
    for i in range(dims):
        mx,mn = codes[:,i].max(),codes[:,i].min()
        diff = mx-mn
        temp_sample = sample.copy()
        for j in range(nums):
            print (i + (nums)*j+1)
            ax = plt.subplot(nums, dims, i + (dims)*j+1)
            temp_sample[i] = diff * (j/nums) + mn
            if approx:
                im = get_dec_image(decoder,temp_sample)
            else:
                _,_,_,euc_names = get_closest(temp_sample,codes,all_names)
                img_name = euc_names[0]
                im = impath_to_image(font_path % img_name)
            #plt.title('-'.join(["%0.2f" %x for x in weight]),fontsize=6)
            # plt.imshow(im.reshape(28*2, 28*2),plt.cm.binary)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('dimchart.png')
    # plt.show()

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
