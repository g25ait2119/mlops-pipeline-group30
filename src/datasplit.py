import os
import shutil
import random
from tqdm import tqdm
print()


"""sampling data"""

s = 'Fake'
traindir = './Dataaset/Train/'+s
valdir = './Dataaset/Validation/'+s
testdir = './Dataaset/Test/'+s


traind_o = './train/'+s.lower()
valdir_o = './val/'+s.lower()
testdir_o = './test/'+s.lower()


# create directories
def create_dirs(path):

    if not os.path.exists(path):
        os.mkdir(path)
        print(path, 'created!')
    else:
        print(path, 'exists!')


print()


# split data

def split_data(in_img, out_img, limit):
    """
    in_img: input image directory
    in_lb,: input label directory
    out_img: output image directory
    out_lb: output label directory

    """

    images1 = os.listdir(in_img)[:limit]
    random.shuffle(images1)

    # labels_temp1 = os.listdir(in_lb)

    count = 1
    for im in tqdm(images1, colour='blue'):
        # print(im)
        # base = os.path.basename(im)
        # fname = os.path.splitext(base)[0]

        # print(count, fname)

        shutil.copy(in_img+'/'+im, out_img)
        count += 1

    print('Done!')


lmt = 1000
split_data(traindir, traind_o, lmt)
print('Training data-set created')

split_data(valdir, valdir_o, int(lmt*0.2))
print('Valdation data-set created')

split_data(testdir, testdir_o, int(lmt*0.1))
print('Test data-set created')
