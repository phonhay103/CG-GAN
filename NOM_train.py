import time
from options.train_options import TrainOptions
import data.NOM_lmdb_dataset as lmdb_dataset
from models import create_model
from util.visualizer import Visualizer
import torch

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
print('The number of training images = %d' % dataset_size)

# create a model given opt.model and other options
model = create_model(opt)
# regular setup: load and print networks; create schedulers
model.setup(opt)
# create a visualizer that display/save images and plots
visualizer = Visualizer(opt)
total_iters = 0                # the total number of training iterations
if opt.continue_train:
    opt.epoch_count = int(opt.epoch) + 1
# outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0 
    model.train() 
    
    for i, data in enumerate(train_loader):  # inner loop within one epoch         
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_iters += sum(opt.batch_size)
        epoch_iter += sum(opt.batch_size)
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / sum(opt.batch_size)
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()
