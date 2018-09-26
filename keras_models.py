from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Concatenate,Dense,Flatten,Reshape,Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def iter_over_layers(klayers,layer_input):
    x = layer_input
    for l in klayers:
        x = l(x)
    return x


def basic_ae(im_shape = (28*2, 28*2, 1)):
    """Autoencoder model"""
    input_img = Input(shape=im_shape)  # adapt this if using `channels_first` image data format
    num_fil = [16,32,16,8]
    conv_size = [(3,3),(4,4),(4,4),(5,5)]
    pool_size = [(2,2),(2,2),(2,2),(2,2)]
    EMBED_DIM = 32
    print ("Input image",input_img.shape)
    x = Conv2D(num_fil[0], conv_size[0], activation='relu', padding='same')(input_img)
    print (x.shape)
    x = MaxPooling2D(pool_size[0], padding='same')(x)
    print (x.shape)
    
    x = Conv2D(num_fil[1], conv_size[1], activation='relu', padding='same')(x)
    print (x.shape)
    x = MaxPooling2D(pool_size[1], padding='same')(x)
    print (x.shape)
    
    x = Conv2D(num_fil[2], conv_size[2], activation='relu', padding='same')(x)
    print (x.shape)
    x = MaxPooling2D(pool_size[2], padding='same')(x)
    print (x.shape)
    
    x = Conv2D(num_fil[3], conv_size[3], activation='relu', padding='same')(x)
    print (x.shape)
    enc = MaxPooling2D(pool_size[3], padding='same')(x)
    print (enc.shape)
    enc_flat = Flatten()(enc)
    x = Dense(64)(enc_flat)
    encoded = Dense(EMBED_DIM)(x)
    print ("Encoding shape :",encoded.shape)
    x = Dense(64)(encoded)
    x = Dense(128)(x)
    dec_input = Reshape((4,4,8))(x)

    x = UpSampling2D(pool_size[3])(dec_input)
    print (x.shape)
    x = Conv2D(num_fil[2], conv_size[3], activation='relu', padding='same')(x)
    print (x.shape)
    x = UpSampling2D(pool_size[1])(x)
    print (x.shape)

    x = Conv2D(num_fil[1], conv_size[0], activation='relu')(x)
    print (x.shape)
    x = UpSampling2D(pool_size[1])(x)
    print (x.shape)


    x = Conv2D(num_fil[0], conv_size[1], activation='relu', padding='same')(x)
    print (x.shape)
    x = UpSampling2D(pool_size[0])(x)
    print (x.shape)
        
    decoded = Conv2D(1, conv_size[0], activation='sigmoid', padding='same')(x)    
    print ("Decoder output :",decoded.shape)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(EMBED_DIM,))
    decoder = Model(encoded_input,iter_over_layers(autoencoder.layers[-11:],encoded_input))
    
    return autoencoder,encoder,decoder,'binary_crossentropy'