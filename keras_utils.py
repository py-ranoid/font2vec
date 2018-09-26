from keras_ae import load_model,get_font_dataset
from proj_utils import project
import numpy as np

def gen_embedding(model,images):
    """Use encoder to generate embeddings from fonts"""
    EMBED_DIM = 32
    codes = model.predict(images.reshape(-1,28*2, 28*2, 1)).reshape(-1,EMBED_DIM)
    return codes

def get_dec_image(decoder,encoding):
    """Use decoder to generate font from embeddings"""
    image = decoder.predict(encoding.reshape(-1,32))
    return image

def store_embeddings(codes,vec_file_path):
    """Store embeddings in tab-spaced value file"""
    all_str = '\n'.join(['\t'.join(map(str,row.tolist())) for row in codes])
    with open(vec_file_path,'w') as vec_file:
        vec_file.write(all_str)

def load_embeddings(vec_file_path):
    """Load embedding from file"""
    with open(vec_file_path,'r') as vec_file:
        content = vec_file.read()
    return np.array([line.split('\t') for line in content.splitlines()])

if __name__ == "__main__":
    model_fname='models/keras_models/e_model_trial345.json'
    weights_fname = 'models/keras_models/e_weights_trial345.h5'
    encoder = load_model(model_fname,weights_fname)
    images,names,paths = get_font_dataset()
    codes = gen_embedding(encoder,images).reshape(-1,128)
    print ("IMAGES:",images.shape)
    print ("ENCODING:",codes[0].shape)
    store_embeddings(codes,'vecs_trial345.h5')