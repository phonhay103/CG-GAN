import os
from options.test_options import TestOptions
from models import create_model
from PIL import Image, ImageDraw, ImageFont
from util import util
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random
import string


class ResizeKeepRatio(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

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
        begin = int((self.size[0]-target_w)/2)
        box = (begin, 0, begin+target_w, target_h)
        img_result.paste(img, box)

        img = self.toTensor(img_result)
        img.sub_(0.5).div_(0.5)
        return img


def draw(font_path, label) -> Image:
    font = ImageFont.truetype(font_path, 80)
    label_w, label_h = font.getsize(label)
    img_content = Image.new('RGB', (label_w, label_h), (255, 255, 255))
    drawBrush = ImageDraw.Draw(img_content)
    drawBrush.text((0, 0), label, fill=(0, 0, 0), font=font)
    return img_content


opt = TestOptions().parse()
transform_img = ResizeKeepRatio((opt.imgW, opt.imgH))

save_dir = os.path.join(opt.results_dir, opt.name, datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
Path(save_dir).mkdir(parents=True, exist_ok=True)

with open(opt.corpusRoot) as f:
    corpus = f.read().splitlines()

model = create_model(opt)

threshold = 250
for i in [11, 12, 13, 14, 15]:
    try:
        opt.epoch = i
        model.setup(opt)
        model.eval()
    except:
        continue

    for word in tqdm(corpus):
        img_content = draw(opt.ttfRoot, word)
        # img_content.save(os.path.join(save_dir, f"{word}_print.png"))
        img_content = transform_img(img_content)
        img_content = img_content.unsqueeze(0)
        data = {'A': img_content, 'B': img_content}
        model.set_single_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img = list(visuals.items())[0][1]
        img = util.tensor2im(img)

        try:
            first_idx = np.where(np.all(img[:, :, 0] < threshold, axis=0))[0][0] + 1
            last_idx = np.where(np.all(img[:, :, 0] < threshold, axis=0))[0][-1] + 1
            img = img[:, first_idx:last_idx, :]
            img = Image.fromarray(img)
        except:
            continue

        random_chars = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=3))
        img.save(os.path.join(save_dir, f"{word}_{random_chars}_epoch_{i}.png"))
