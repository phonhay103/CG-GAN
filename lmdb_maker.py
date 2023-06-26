# -*- coding: utf-8 -*-  

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False

    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        # if img == None:
        #     return False
        imgH, imgW, imgC = img.shape[0], img.shape[1], img.shape[2]
        if imgH * imgW * imgC == 0:
            return False
    except:
        return False
    # img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    # imgH, imgW = img.shape[0], img.shape[1]
    # if imgH * imgW == 0:
    #     return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            try:
                txn.put(k.encode(), v)
            except:
                print(k, v)
                exit()


def createDataset(outputPath, imagePathList, labelList, writerIDList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        writerID = writerIDList[i]

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        writerIDKey = 'writerID-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[writerIDKey] = writerID.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = lexiconList[i]
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

PATH = '/mnt/disk1/naver/nl/vietocr/datasets/images/full_img'
LABEL_PATH = '/mnt/disk1/naver/nl/vietocr/datasets/labels/train.txt'
# LABEL_PATH = '/mnt/disk1/naver/nl/vietocr/datasets/labels/valid.txt'
# LABEL_PATH = '/mnt/disk1/naver/nl/vietocr/datasets/labels/test.txt'
OUTPUT_PATH = 'data/datasets/train_VIE'
# OUTPUT_PATH = 'data/datasets/valid_VIE'
# OUTPUT_PATH = 'data/datasets/test_VIE'

if __name__ == '__main__':
    img_path_list = []
    label_list = []
    ID_list = []

    with open(LABEL_PATH) as f:
        img_file_list = f.read().splitlines()

    for img_path in img_file_list:
        # error when using split(): "none_cap_109_(1)_cưới .jpg	cưới"
        img_path, label = img_path.split('\t')
        img_path_list.append(f'{PATH}/{img_path}')
        label_list.append(label)
        ID_list.append('0')  # TODO: no writer_id => clustering

    assert (len(img_path_list) == len(label_list))
    assert (len(img_path_list) == len(ID_list))

    createDataset(
        outputPath=OUTPUT_PATH,
        imagePathList=img_path_list,
        labelList=label_list,
        writerIDList=ID_list,
        lexiconList=None
    )
