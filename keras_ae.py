from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras_models import basic_ae,basic_vae
from utils import FontAlphabetsDataset,get_all_samples
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import random
import os

hig,wid = 28,28

def get_font_dataset():
    """ Function to load font images (as numpy array), list of font names and their path."""
    global hig,wid
    font_dataset = FontAlphabetsDataset(folder_path='./font_ims_all_56',custom_path='/*png')
    wid,hig = 28*2,28*2
    print ("Loading data")
    images = get_all_samples(font_dataset)
    print ("Loaded data")
    return images,font_dataset.names,font_dataset.im_paths

def get_data(data_set = "alpha"):
    """Function to load fetch train and test dataset."""
    global hig,wid  
    if data_set ==  "mnist":
        wid,hig = 28,28
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

    elif data_set == "alpha":
        images,_,_ = get_font_dataset()
        print ("IMAGES SHAPE:",images.shape)
        
        DATA_SIZE = images.shape[0]
        all_indices = list(range(DATA_SIZE))
        random.shuffle(all_indices)
        test_split = int(0.05*DATA_SIZE)
        x_test = images[all_indices[:test_split]]
        print ("Untrained :",all_indices[:test_split])
        x_train = images[all_indices[test_split:]]
        
    x_train = np.reshape(x_train, (-1, hig, wid, 1))
    x_test = np.reshape(x_test, (-1, hig, wid, 1))
    return x_train,x_test

def get_model():
    """Function to create/import model from keras_models"""
    im_shape = (hig, wid, 1)
    autoencoder,encoder,decoder,loss = basic_vae()
    return autoencoder,encoder,decoder ,loss

def save_model(model,model_fname='ae_model.json',weights_fname = 'ae_weights.h5'):
    """Function to save model and weights at given paths."""
    model_json = model.to_json()
    with open(model_fname, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_fname)
    print("Saved model to disk")

def load_model(model_fname='ae_model.json',weights_fname = 'ae_weights.h5'):
    """Function to load model and weights from given paths."""
    # load json and create mode
    with open(model_fname, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_fname)
    print("Loaded model from disk")
    return loaded_model

def plot_results(autoencoder,x_test, n=10):
    """Function to plot 10 font images from test set and their reconstructed output from embeddings"""
    decoded_imgs = autoencoder.predict(x_test)
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(hig, wid))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        print (decoded_imgs[i].shape,i)
        plt.imshow(decoded_imgs[i].reshape(hig, wid))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def main(model_fname,weights_fname,name):
    """ main function. Called when keras_ae.py is run"""
    if os.path.exists(model_fname) and os.path.exists(weights_fname):
        # Loads model if it exists
        autoencoder = load_model(model_fname,weights_fname)
        x_train,x_test = get_data()
        plot_results(autoencoder,x_test)
    else:
        # Loads dataset and trains models
        print ("MODEL NOT FOUND")
        autoencoder,encoder,decoder,loss = get_model()
        autoencoder.compile(optimizer='rmsprop', loss=loss)
        plot_model(autoencoder, to_file='model_%s.png'%name,show_shapes=True)
        x_train,x_test = get_data()
        autoencoder.fit(x_train, x_train,
                        epochs=2500,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder_0')])
        save_model(autoencoder,model_fname, )
        save_model(encoder,model_fname.replace('ae_model','e_model'),weights_fname.replace('ae_weights','e_weights'))
        save_model(decoder,model_fname.replace('ae_model','d_model'),weights_fname.replace('ae_weights','d_weights'))
        print ("ENCODER SAVED TO")
        print ("model_fname='%s'" % model_fname.replace('ae_model','e_model'))
        print ("weights_fname = '%s'" % weights_fname.replace('ae_weights','e_weights'))
        print ("from keras_ae import load_model\nencoder = load_model(model_fname,weights_fname)")
        plot_results(autoencoder,x_test)


if __name__ == "__main__":
    name = "var_3445_56_dim32_allfont"
    model_fname='models/keras_models/ae_model_%s.json' % name
    weights_fname = 'models/keras_models/ae_weights_%s.h5' % name
    main(model_fname,weights_fname,name)    