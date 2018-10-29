import numpy as np
import sys
import os
sys.path.append("..")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import glob

#input_path = './test_images'
input_path = "/scratch2/NAACL2018/shooting_images/result_images/aug2018/related_images/related_aug_theme1"
img2vec = Img2Vec()
vector_fh = open('resnet50_feature_vectors.txt', 'a+')

filenames = glob.glob(input_path + "/*.*")

for filename in filenames:
    print(filename)
    img = Image.open(filename)
    nd_arr = img2vec.get_vec(img)
    #print(nd_arr.shape)
    #print(type(nd_arr))
    #print(nd_arr.T.shape)
    #print(np.transpose([nd_arr]).shape)
    #print(nd_arr.tolist())
    #print(list(nd_arr))
    curr_line = np.reshape(nd_arr, (1, 2048))
    np.savetxt(vector_fh, curr_line)
