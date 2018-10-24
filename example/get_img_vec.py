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


# For each test image, we store the filename and vector as key, value in a dictionary
pics = {}
'''for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    pics[filename] = vec'''



#pic_name = "test_images/cat.jpg"

#pic = Image.open("test_images/cat.jpg")
pic = Image.open(input_path + "/969.jpeg")
#vec = img2vec.get_vec(img)

#print("vec is: ", vec)
#print("vec shape is: ", len(vec))
#print("vec shape is: ", vec.shape)

filenames = glob.glob(input_path + "/*.*")

for filename in filenames:
    print(filename)
    img = Image.open(filename)
    nd_arr = img2vec.get_vec(img)
    print(type(nd_arr))
    print(nd_arr.T)
    np.savetxt(vector_fh, nd_arr.T, delimiter='\t')

'''while pic_name != "exit":
    pic_name = str(input("Which filename would you like similarities for?\n"))

    try:
        sims = {}
        for key in list(pics.keys()):
            if key == pic_name:
                continue

            sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

        d_view = [(v, k) for k, v in sims.items()]
        d_view.sort(reverse=True)
        for v, k in d_view:
            print(v, k)

    except KeyError as e:
        print('Could not find filename %s' % e)

    except Exception as e:
        print(e)'''
