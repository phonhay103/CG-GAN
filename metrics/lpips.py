import torch
from PIL import Image
import torchvision.transforms as TF
import pathlib
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

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
    TF.Resize((32, 96)),
    TF.ToTensor(),
])

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

real_path = pathlib.Path('/mnt/disk3/VALID')
real_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in real_path.glob('*.{}'.format(ext))])
real_dataset = ImagePathDataset(real_files, transforms=transform)
real_loader = torch.utils.data.DataLoader(real_dataset,
    batch_size=batch_size,
    shuffle=False,  
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

gan_path = pathlib.Path('/mnt/disk3/GAN/GAN_183_NF')
gan_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in gan_path.glob('*.{}'.format(ext))])
gan_dataset = ImagePathDataset(gan_files, transforms=transform)
gan_loader = torch.utils.data.DataLoader(gan_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False
)

try:
    with open('lpips.txt') as f:
        lpips_metric = f.read().splitlines()
        filelist = set(line.split('\t')[0] for line in lpips_metric)
except:
    lpips_metric = []
    filelist = set()

lpips = LPIPS(net_type='alex', reduction='mean', normalize=True).to(device)
for filename, gan_image in tqdm(gan_loader):
    filename = filename[0]
    if filename in filelist:
        continue
    else:
        filelist.add(filename)
    
    lpips_value = torch.empty(0).to(device)
    for _, real_images in real_loader:
        real_images = real_images.to(device)
        if len(real_images) < batch_size:
            gan_image = gan_image.repeat(len(real_images), 1, 1, 1).to(device)
        else:
            gan_image = gan_image.repeat(batch_size, 1, 1, 1).to(device)

        _lpips = lpips(gan_image, real_images).detach()
        _lpips = _lpips.unsqueeze(0)
        lpips_value = torch.cat((lpips_value, _lpips))

    lpips_value = lpips_value.mean().cpu().numpy()
    lpips_metric.append(f'{filename}\t{lpips_value}')

    if len(lpips_metric) % 30 == 0:
        with open('lpips.txt', 'w') as f:
            f.write('\n'.join(lpips_metric))

with open('lpips.txt', 'w') as f:
    f.write('\n'.join(lpips_metric))
