# CG-GAN(CVPR2022)

This is the official PyTorch implementation of the CVPR 2022 paper: "Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator".

![](img/pipeline.png)

## Requirements

(Welcome to develop CG-GAN together.)

We recommend you to use [Anaconda](https://www.anaconda.com/) to manage your libraries.

- [Python](https://www.python.org/) 3.6* 
- [PyTorch](https://pytorch.org/) 1.0.* 
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/#)
- [LMDB](https://pypi.org/project/lmdb/)
- [matplotlib](https://pypi.org/project/matplotlib/)

## Data Preparation
Please convert your own dataset to **LMDB** format by using the tool ``lmdb_maker.py`` 

Both the char(text) label, the radical list and the corresponding writer ID are required for every text image. 

Please prepare the **TTF font** and **corpus** for the rendering of printed style images.

For handwritten word synthesis task, please download the datasets prepared by us. 

- [Google Drive (datasets in **LMDB** format)]()
```
data
-- datasets
   -- train_img_104K
      -- data.mdb
      -- lock.mdb
```

## Training

### Handwritten word synthesis 

Modify the **dataRoot** , **ttfRoot** and **corpusRoot** in `scripts/train_handwritten.sh`as your settings.

```bash
  --dataroot data/<train_folder> \
  --ttfRoot data/fonts/<font_folder> \
  --corpusRoot data/texts/<seen_char>.txt \
```

Train your model, run

```bash
 sh scripts/train_handwritten.sh
```

## Testing

### Handwritten word synthesis 

test your model, run

```bash
 sh scripts/test_handwritten.sh
```

## Citation
If our paper helps your research, please cite it in your publication(s):
```
@article{cluo2019moran,
  author    = {Yuxin Kong, Canjie Luo, Weihong Ma, Qiyuan Zhu, Shenggao Zhu, Nicholas Yuan, Lianwen Jin},
  title     = {Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator},
  year      = {2022},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  publisher = {IEEE}
}
```
