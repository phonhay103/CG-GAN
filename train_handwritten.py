import time
from options.train_options import TrainOptions
import data.lmdb_dataset_iam as lmdb_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
from util import util
from PIL import Image
import numpy as np
import shutil
import os
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths

opt = TrainOptions().parse()
transform_img = lmdb_dataset.resizeKeepRatio((opt.imgW, opt.imgH))
train_ds = lmdb_dataset.ConcatLmdbDataset(
    dataset_list=opt.dataroot,
    batchsize_list=opt.batch_size,
    font_path=opt.ttfRoot,
    corpusRoot=opt.corpusRoot,
    transform_img=transform_img,
    transform_target_img=transform_img,
)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=sum(opt.batch_size),
    shuffle=True, sampler=None, drop_last=True,
    num_workers=int(opt.num_threads))
dataset_size = len(train_ds)    # get the number of images in the dataset.
threshold = 250
print('The number of training images = %d' % dataset_size)

# with open('valid_image.txt') as f:
#     test_corpus = [word.split('\t')[1] for word in f.read().splitlines()]

# create a model given opt.model and other options
model = create_model(opt)
# regular setup: load and print networks; create schedulers
model.setup(opt)
# create a visualizer that display/save images and plots
visualizer = Visualizer(opt)
total_iters = 0                # the total number of training iterations

# outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0
    model.train()

    ### Training ###
    for i, data in enumerate(train_loader):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_iters += sum(opt.batch_size)
        epoch_iter += sum(opt.batch_size)
        # unpack data from dataset and apply preprocessing
        model.set_input(data)
        # calculate loss functions, get gradients, update network weights
        model.optimize_parameters()

        if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(
                model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / sum(opt.batch_size)
            visualizer.print_current_losses(
                epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(
                    epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' %
                  (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()

    if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()

    # ### Testing ###
    # model.eval()
    # shutil.rmtree('valid_result')
    # os.mkdir('valid_result')
    # for word in tqdm(test_corpus):
    #     img = util.draw(opt.ttfRoot, word)
    #     img = transform_img(img)
    #     img = img.unsqueeze(0)
    #     data = {'A': img, 'B': img}
    #     model.set_single_input(data)
    #     visuals = model.get_current_visuals()
    #     img = list(visuals.items())[0][1]
    #     img = util.tensor2im(img)
    #     try:
    #         first_idx = np.where(np.all(img[:, :, 0] < threshold, axis=0))[
    #             0][0] + 1
    #         last_idx = np.where(np.all(img[:, :, 0] < threshold, axis=0))[
    #             0][-1] + 1
    #         img = img[:, first_idx:last_idx, :]
    #         img = Image.fromarray(img)
    #     except:
    #         continue
    #     img.save(os.path.join('valid_result', f"{word}_.png"))
    # fid_value = calculate_fid_given_paths(
    #     paths=['valid_image', 'valid_result'],
    #     batch_size=opt.batch_size[0],
    #     device=f'cuda:{opt.gpu_ids[0]}',
    #     dims=2048,
    #     num_workers=0,
    # )
    # print('FID: ', fid_value)
    # break
