# -*- coding: utf-8 -*-  

import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
import json

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        imgH, imgW, imgC = img.shape[0], img.shape[1], img.shape[2]
        if imgH * imgW * imgC == 0:
            return False
    except:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            try:
                txn.put(k.encode(), v)
            except:
                print("writeCache error:", k, v)
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
    for i in tqdm(range(nSamples)):
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
            cache[lexiconKey] = lexiconList[i].encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}

        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

PATH = 'NomScript/all_images'
LABEL_PATH = 'NomScript/all_images.txt'
OUTPUT_PATH = 'data/datasets/NOM_train'

if __name__ == '__main__':
    dictionary_dir = 'data/dictionaries/NOM_IDS_dictionary.txt'
    with open(dictionary_dir) as f:
        radical_data = f.read().splitlines()

    with open('NomScript/unicode_to_nom_dict.json') as f:
        unicode_to_json_dict = json.load(f)
    
    nom_to_radical_dict = dict()
    for line in radical_data:
        label, radical = line.split(':')[0], line.split(':')[1]
        nom_to_radical_dict[label] = radical

    img_path_list = []
    label_list = []
    ID_list = []
    lexicon_list = []

    with open(LABEL_PATH) as f:
        img_file_list = f.read().splitlines()

    for img_path in img_file_list:
        img_path, label = img_path.split('\t')
        if label == 'Nom-Script':
            continue

        lexicon = nom_to_radical_dict[unicode_to_json_dict[label]]

        lexicon_list.append(lexicon)
        img_path_list.append(f'{PATH}/{img_path}')
        label_list.append(label)
        ID_list.append('0')  # TODO: no writer_id => set default value is 0

    assert (len(img_path_list) == len(label_list))
    assert (len(img_path_list) == len(ID_list))
    assert (len(img_path_list) == len(lexicon_list))

    createDataset(
        outputPath=OUTPUT_PATH,
        imagePathList=img_path_list,
        labelList=label_list,
        writerIDList=ID_list,
        lexiconList=lexicon_list
    )
