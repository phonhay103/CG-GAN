import torch
from PIL import Image
import torchvision.transforms as TF
from torchmetrics.image.fid import FrechetInceptionDistance as  FID
import pathlib
from tqdm import tqdm

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

IMG_SIZE = (32, 96)
device = torch.device('cuda:1')
transform = TF.Compose([
    TF.Resize(IMG_SIZE),
    TF.ToTensor(),
])

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

real_path = pathlib.Path('/mnt/disk3/VALID')
real_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in real_path.glob('*.{}'.format(ext))])
real_dataset = ImagePathDataset(real_files, transforms=transform)
real_loader = torch.utils.data.DataLoader(real_dataset,
    batch_size=1000,
    shuffle=False,  
    drop_last=False,
    num_workers=10,
)
real_images = torch.cat([batch[1] for _, batch in enumerate(real_loader)], dim=0).to(device)
print('real_images', real_images.shape)

gan_path = pathlib.Path('/mnt/disk3/GAN/GAN_183_NF')
gan_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in gan_path.glob('*.{}'.format(ext))])
gan_dataset = ImagePathDataset(gan_files, transforms=transform)
gan_loader = torch.utils.data.DataLoader(gan_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False
)

fid = FID(2048, True, True)
psnr_metric = []
for filename, gan_image in tqdm(gan_loader):
    gan_image = gan_image.to(device)
    filename = filename[0]
    psnr_value = fid(gan_image, real_images, reduction='elementwise_mean')
    psnr_value = psnr_value.cpu().numpy()
    psnr_metric.append(f'{filename}\t{psnr_value}')

print('Logging')
with open(f'psnr_{IMG_SIZE[0]}_{IMG_SIZE[1]}.txt', 'w') as f:
    f.write('\n'.join(psnr_metric))
print('Done')
