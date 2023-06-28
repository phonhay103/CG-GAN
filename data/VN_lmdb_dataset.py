import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class ConcatLmdbDataset(Dataset):
    def __init__(self, dataset_list, batchsize_list, font_path=None, corpusRoot=None,
                 transform_img=None, transform_target_img=None):
        assert len(dataset_list) == len(batchsize_list)

        self.corpus = open(corpusRoot, "r").read().splitlines()
        self.datasets = []
        self.prob = [batchsize / sum(batchsize_list)
                     for batchsize in batchsize_list]
        for i in range(len(dataset_list)):
            self.datasets.append(lmdbDataset(
                dataset_list[i], font_path, self.corpus, transform_img, transform_target_img))
        self.datasets_range = range(len(self.datasets))

    def __len__(self):
        return max([dataset.__len__() for dataset in self.datasets])

    def __getitem__(self, index):
        idx_dataset = np.random.choice(
            self.datasets_range, 1, p=self.prob).item()
        idx_sample = index % self.datasets[idx_dataset].__len__()
        return self.datasets[idx_dataset][idx_sample]


class lmdbDataset(Dataset):
    def __init__(self, root=None, font_path=None, corpus=None,
                 transform_img=None, transform_target_img=None, radical_dict=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.root = root
        self.transform_img = transform_img
        self.transform_target_img = transform_target_img
        self.font_path = font_path
        self.corpus = corpus
        self.radical_dict = radical_dict

    def __len__(self) -> int:
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            writerID_key = 'writerID-%09d' % index
            writerID = int(txn.get(writerID_key.encode()))

            font = ImageFont.truetype(self.font_path, 80)
            label_target = self.corpus[random.randint(0, len(self.corpus)-1)]

            try:
                label_w, label_h = font.getsize(label_target)
                img_target = Image.new('RGB', (label_w, label_h), (255, 255, 255))
                drawBrush = ImageDraw.Draw(img_target)
                drawBrush.text((0, 0), label_target, fill=(0, 0, 0), font=font)

                # ZeroDivisionError: division by zero
                img_target = self.transform_target_img(img_target)
                img = self.transform_img(img)
            except:
                return self[index + 1]

            return {
                'A': img,
                'B': img_target,
                'A_paths': index,
                'writerID': writerID,
                'A_label': label,
                'B_label': label_target,
                'root': self.root
            }


class resizeKeepRatio(object):
    def __init__(self, size, interpolation=Image.BILINEAR, train=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.train = train

    def __call__(self, img):

        if img.mode == 'L':
            img_result = Image.new("L", self.size, (255))
        elif img.mode == 'RGB':
            img_result = Image.new("RGB", self.size, (255, 255, 255))
        else:
            print("Unknow image mode!")

        img_w, img_h = img.size

        target_h = self.size[1]
        target_w = max(1, int(img_w * target_h / img_h))

        if target_w > self.size[0]:
            target_w = self.size[0]

        img = img.resize((target_w, target_h), self.interpolation)
        begin = random.randint(
            0, self.size[0]-target_w) if self.train else int((self.size[0]-target_w)/2)
        box = (begin, 0, begin+target_w, target_h)
        img_result.paste(img, box)

        img = self.toTensor(img_result)
        img.sub_(0.5).div_(0.5)
        return img
