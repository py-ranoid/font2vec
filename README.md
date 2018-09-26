# Font2vec : Learning low-dimensional representation of fonts
## About
Our work successfully creates font embeddings that can be used for achieving the following goals:
1. Identify the fonts effectively from a given font image, and also suggest fonts similar to it.
2. Provide an interface to alter features of a font (such as thickness, width, alignment, etc). In addition to generating new fonts from the altered embedding, we also suggest similar fonts.
3. Combine multiple fonts, and visualize the output of the same by interpolating their embeddings.
4. Unsupervised labelling of fonts by clustering them effectively. This could augment performance of OCR tasks.

## Dataset
- We generated the dataset from scratch using [Google Fonts](github.com/google/fonts) and extra fonts were scraped from the web. 
- .ttf and .otf files are used to create 56 * 56 grayscale images were generated
- There are two datasets :
    - [2409 font images](https://drive.google.com/file/d/1sLe0Oyf4tP-XXG1OYpbTtC5YPbD3aa1Y/view?usp=sharing), generated from Google Fonts.
    ```
    gdrive_download 1sLe0Oyf4tP-XXG1OYpbTtC5YPbD3aa1Y gfont_ims.zip
    unzip gfont_ims.zip
    ```
    - [13,674 font images](https://drive.google.com/file/d/13-MaBAjnX2WY5xFwayYt73nm01QHywGJ/view?usp=sharing), generated from (Google Fonts + Fonts scraped from the web).
    ```
    gdrive_download 13-MaBAjnX2WY5xFwayYt73nm01QHywGJ font_ims_all.zip
    unzip font_ims_all.zip
    ```



```
function gdrive_download () {
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
rm -rf /tmp/cookies.txt
}
```