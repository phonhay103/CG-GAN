from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from tqdm import tqdm

# Define the transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

def get_images_tensor(txt_file_path, transform=transform):
    # Load image file paths from the text file
    with open(txt_file_path, 'r') as f:
        image_filenames = [line.split('\t')[0] for line in f.read().splitlines()]

    # Load and preprocess the images
    images = []
    for filename in tqdm(image_filenames):
        file_path = os.path.join('/mnt/disk3/CGGANv2', filename)
        image = Image.open(file_path)
        image = transform(image)
        images.append(image)

    # Stack the images into a single tensor
    return torch.stack(images)

fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)

gan_test_path = "/mnt/disk3/vietocr/datasets/labels/test.txt"
gan_sys_path = "/mnt/disk3/vietocr/datasets/labels/CGGANv2.2_84_filter.txt"
# images_real = get_images_tensor(gan_test_path)
# images_fake = get_images_tensor(gan_sys_path)
images_fake = torch.load('filter_47.pt')
# fid.update(images_real, real=True)
fid.update(images_fake, real=False)
# fid.compute()