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

def basic_vae(im_shape = (56, 56, 1)):    
    """Variational Autoencoder model"""
    input_img = Input(shape=im_shape)  # adapt this if using `channels_first` image data format
    num_fil = [16,32,16,8]
    conv_size = [(3,3),(4,4),(4,4),(5,5)]
    pool_size = [(2,2),(2,2),(2,2),(2,2)]
    EMBED_DIM = 32
    print ("Input image",input_img.shape)
    
    x = Conv2D(num_fil[0], conv_size[0], activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size[0], padding='same')(x)
    x = Conv2D(num_fil[1], conv_size[1], activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size[1], padding='same')(x)
    x = Conv2D(num_fil[2], conv_size[2], activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size[2], padding='same')(x)
    x = Conv2D(num_fil[3], conv_size[3], activation='relu', padding='same')(x)
    enc = MaxPooling2D(pool_size[3], padding='same')(x)
    
    enc_flat = Flatten()(enc)
    x = Dense(64)(enc_flat)
    print (x.shape)
    
    # encoded = Dense(EMBED_DIM)(x)
    z_mean = Dense(EMBED_DIM)(x)
    print ("z_mean",z_mean.shape)
    z_log_sigma = Dense(EMBED_DIM)(x)
    print ("z_sig",z_log_sigma.shape)
    z = Lambda(sampling, output_shape=(EMBED_DIM,))([z_mean, z_log_sigma])
    print ("z",z.shape)
    encoded = z
    print ("Encoding shape :",encoded.shape)
    
    x = Dense(64)(encoded)
    print (x.shape)
    x = Dense(128)(x)
    print (x.shape)
    dec_input = Reshape((4,4,8))(x)

    x = UpSampling2D(pool_size[3])(dec_input)
    x = Conv2D(num_fil[2], conv_size[3], activation='relu', padding='same')(x)
    x = UpSampling2D(pool_size[1])(x)
    x = Conv2D(num_fil[1], conv_size[0], activation='relu')(x)
    x = UpSampling2D(pool_size[1])(x)
    x = Conv2D(num_fil[0], conv_size[1], activation='relu', padding='same')(x)
    x = UpSampling2D(pool_size[0])(x)
        
    decoded = Conv2D(1, conv_size[0], activation='sigmoid', padding='same')(x)    
    print ("Decoder output :",decoded.shape)

    var_autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, z_mean)

    encoded_input = Input(shape=(EMBED_DIM,))
    decoder = Model(encoded_input,iter_over_layers(var_autoencoder.layers[-11:],encoded_input))
    
    def vae_loss(x, x_decoded_mean):
        print ("X:",x.shape)
        print ("DEC:",x_decoded_mean.shape)
        xent_loss = K.sum(objectives.binary_crossentropy(x, x_decoded_mean), axis=[1, 2])
        print ("XENT",xent_loss.shape)
        # xent_loss = 10 * K.mean(K.square(x_ - x_decoded_mean_), axis=[1, 2])


        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        print ("KL",kl_loss.shape)
        return xent_loss + kl_loss

    return var_autoencoder,encoder,decoder,vae_loss

def char_ae(im_shape = (28, 28, 1,),char_shape=(26,)):
    input_img = Input(shape=im_shape)  # adapt this if using `channels_first` image data format
    char_input = Input(shape=char_shape)
    char_input = Input(shape=char_shape)
    num_fil = [32,16,8]
    conv_size = [(3,3),(3,3),(3,3)]
    pool_size = [(2,2),(2,2),(2,2)]
    x = Conv2D(num_fil[0], (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size[0], padding='same')(x)
    x = Conv2D(num_fil[1], (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size[1], padding='same')(x)
    x = Conv2D(num_fil[2], (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size[2], padding='same')(x)

    print ("Encoding shape :",encoded.shape)
    print (encoded.shape)
    
    enc_flat = Flatten()(encoded)    
    dec_input = Concatenate(axis=1)([char_input,enc_flat])
    dec_input = Dense(128,activation='relu')(dec_input)
    dec_input = Reshape((4,4,num_fil[2]))(dec_input)
    
    x = Conv2D(num_fil[2], (3, 3), activation='relu', padding='same')(dec_input)
    x = UpSampling2D(pool_size[2])(x)
    x = Conv2D(num_fil[1], (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D(pool_size[1])(x)
    x = Conv2D(num_fil[0], (3, 3), activation='relu')(x)
    x = UpSampling2D(pool_size[0])(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    print ("Decoder output :",decoded.shape)

    autoencoder = Model([input_img,char_input], decoded)
    encoder = Model([input_img,char_input], encoded)
    return autoencoder,encoder
