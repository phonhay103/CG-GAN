import os
from options.test_options import TestOptions
from models import create_model
from PIL import Image, ImageDraw, ImageFont
from util import util
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random
import string
import json


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

def main():
    opt = TestOptions().parse()
    transform_img = ResizeKeepRatio((opt.imgW, opt.imgH))

    timeNow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(opt.results_dir, opt.name, timeNow)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open(opt.corpusRoot) as f:
        corpus = f.read().splitlines()
    
    with open('NomOCR/nom_to_unicode_dict.json') as f:
        nom_to_unicode_dict = json.load(f)

    model = create_model(opt)

    try:
        opt.epoch = 45
        model.setup(opt)
        model.eval()
    except:
        print(f'model {opt.epoch} is invalid')
        return

    for word in tqdm(corpus):
        img_content = draw(opt.ttfRoot, word)
        unicode_word = nom_to_unicode_dict[word]
        # img_content.save(os.path.join(save_dir, f"{word}_print.png"))
        img_content = transform_img(img_content)
        img_content = img_content.unsqueeze(0)
        data = {'A': img_content, 'B': img_content}
        model.set_single_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img = list(visuals.items())[0][1]
        img = util.tensor2im(img)
        img = Image.fromarray(img)

        random_chars = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=3))
        filename = f"epoch_{opt.epoch}_{word}_{unicode_word}_{random_chars}.png"
        img.save(os.path.join(save_dir, filename))

if __name__ == '__main__':
    main()