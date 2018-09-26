from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

LOG_DIR = 'minimal_sample'
NAME_TO_VISUALISE_VARIABLE = "fonts_embedding"
TO_EMBED_COUNT = 500

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix(images,hig=56,wid=56):
    return np.reshape(images,(-1,hig,wid))

def invert_grayscale(images):
    """ Makes black white, and white black """
    return 1-images

def gen_sprite(images,sprite_path,invert=False):
    ims = vector_to_matrix(images)
    if invert:
        ims = invert_grayscale(ims)
    sprite_image = create_sprite_image(ims)
    plt.imsave(sprite_path,sprite_image,cmap='gray')
    print ("SAVED SPRITE AT",sprite_path)    

def gen_metadata(fnames,metadata_path):
    # with open(metadata_path,'w') as f:
    #     f.write("Index\tLabel\n")
    #     for index,label in enumerate(fnames):
    #         f.write("%d\t%s \n" % (index,label))
    all_names = '\n'.join(fnames)
    with open(metadata_path,'w') as vecfile:
        vecfile.write(all_names)
    print ("SAVED METADATA AT",metadata_path)

def load_names(metadata_path):
    with open(metadata_path,'r') as vecfile:
        all_names = vecfile.read()
    return all_names.splitlines()

def project(embeddings,font_names,images,logdir=None):
    logdir = LOG_DIR if logdir is None else logdir
    logdir = os.getcwd()+'/tf_logs/'+logdir    
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    sprite_path =  os.path.join(logdir,'font_sprite.png')
    metadata_path =  os.path.join(logdir,'metadata.tsv')
    gen_sprite(images,sprite_path)
    gen_metadata(font_names,metadata_path)
    
    embedding_var = tf.Variable(embeddings, name=NAME_TO_VISUALISE_VARIABLE)
    summary_writer = tf.summary.FileWriter(logdir)
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = metadata_path
    embedding.sprite.image_path = sprite_path
    embedding.sprite.single_image_dim.extend([images.shape[1],images.shape[2]])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logdir, "model.ckpt"), 1)
    print ("RUN :\n"+"tensorboard --logdir "+logdir)


#   
# sprite_image = create_sprite_image(vector_to_matrix_mnist(images))


# plt.imsave(LOG_DIR+'/sprite.png',sprite_image,cmap='gray')
# plt.imsave(LOG_DIR+'/sprite.png',sprite_image,cmap='gray')
# plt.imsave(LOG_DIR+'/sprite.png',sprite_image,cmap='gray')
# from tensorflow.contrib.tensorboard.plugins import projector
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
# all_names = '\n'.join(names)
# with open(LOG_DIR+'/metadata.tsv','w') as vecfile:
#     vecfile.write(all_names)

# config = projector.ProjectorConfig()    
# summary_writer = tf.summary.FileWriter(LOG_DIR)
# embedding_var = tf.Variable(codes, name="font_embedding")
# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name
# embedding.metadata_path = path_for_mnist_metadata
# embedding.sprite.image_path = path_for_mnist_sprites
# embedding.sprite.single_image_dim.extend([56,56])
# projector.visualize_embeddings(summary_writer, config)
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)


