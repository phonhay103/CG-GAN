import torch
from pytorch_msssim import ssim, ms_ssim
from PIL import Image
import torchvision.transforms as TF
import pathlib
from tqdm import tqdm
import numpy as np

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return path.name, img

batch_size = 1000
num_workers = 10
device = torch.device('cuda:1')
transform = TF.Compose([
    TF.Resize((100, 100)),
    TF.ToTensor(),
])

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

real_path = pathlib.Path('/mnt/disk3/VALID')
real_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in real_path.glob('*.{}'.format(ext))])
real_dataset = ImagePathDataset(real_files, transforms=transform)
real_loader = torch.utils.data.DataLoader(real_dataset,
    batch_size=batch_size,
    shuffle=False,  
    drop_last=False,
    num_workers=num_workers,
)

gan_path = pathlib.Path('/mnt/disk3/GAN/GAN_47_F')
gan_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in gan_path.glob('*.{}'.format(ext))])
gan_dataset = ImagePathDataset(gan_files, transforms=transform)
gan_loader = torch.utils.data.DataLoader(gan_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False
)

ssim_values = []
ssim_metric = []

for filename, gan_image in tqdm(gan_loader):
    filename = filename[0]
    gan_images = gan_image.repeat(batch_size, 1, 1, 1).to(device)
    ssim_value = torch.empty(0).to(device)

    for _, real_images in tqdm(real_loader):
        real_images = real_images.to(device)
        if len(real_images) < batch_size:
            gan_images = gan_image.repeat(len(real_images), 1, 1, 1).to(device)
        _ssim = ssim(gan_images, real_images, size_average=False, data_range=1)
        ssim_value = torch.cat((ssim_value, _ssim))

    ssim_value = ssim_value.mean().cpu().numpy()
    ssim_values.append(ssim_value)
    ssim_metric.append(f'{filename}\t{ssim_value}')

ssim_metric.append(f'SSIM\t{np.mean(ssim_values)}')

with open('ssim.log', 'w') as f:
    f.write('\n'.join(ssim_metric))
