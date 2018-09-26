from PIL import Image, ImageDraw, ImageFont
from glob import glob
import time
import os

WHITE = "#FFFFFF"
BLACK = "#000000"

def gen_image(font_path,letter,size,dim,save_path,save):    
    """
        Generates image of font with given letter (and saves it save_path)
    """
    min_dim,max_dim = 0.6*dim,0.8*dim
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    im_name = font_name+ "_"+letter+ ".png"
    im_path = os.path.join(save_path, im_name)
    if os.path.exists(im_path):
        return None
    font = ImageFont.truetype(font_path, size)
    canvas = Image.new("RGB", (dim,dim), WHITE)
    draw = ImageDraw.Draw(canvas)
    w, h = draw.textsize(letter, font=font)
    if w == 0 and h == 0:
        return None
    if w>max_dim or h>max_dim:
        print ("Too big",font_name,w,h)
        return gen_image(font_path,letter,int(size*0.9),dim,save_path,save)
    elif w<dim*min_dim and h<min_dim:
        print ("Too Small",font_name,w,h)
        time.sleep(0.1)
        return gen_image(font_path,letter,int(size*1.1),dim,save_path,save)
    else:
        position = ((dim-w)/2, (dim-h)/2)
    draw.text(position,letter, fill=BLACK, font=font)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if save:
        canvas.save(os.path.join(save_path, im_name))
        return 
    else:
        return canvas

def gen_dataset(font_dir='font_files',save=True,save_path="gfont_ims"):
    """
        Read all fonts files in font_dir and save their images to save_path.
    """
    font_files = glob(font_dir+"/*tf")+glob(font_dir+"/*TF")
    so_list = []
    im_outs = []
    for fname in font_files:
        print(fname)
        try:im_outs.append(gen_image(fname,'R',55,dim=56,save_path=save_path,save=save))
        except OSError:so_list.append(fname)
    print ("FAILED:",so_list)
    if save:
        return im_outs

if __name__ == "__main__":
    print (len(gen_dataset("gfonts")))
