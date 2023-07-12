# Create datasets
* python NOM_lmdb_maker.py
* python VN_lmdb_maker.py
    - PATH: path to all images
    - LABEL_PATH: path to all labels file .txt
    - OUTPUT_PATH: path to store lmdb dataset

# Metrics

python -m pytorch_fid ../VALID valid_stats --device cuda:2 --batch-size 100 --num-workers 20 --save-stats
python -m pytorch_fid ../TEST test_stats --device cuda:2 --batch-size 100 --num-workers 20 --save-stats

python -m pytorch_fid full_img_stats.npz ../GAN/GAN_47_F --device cuda:2 --batch-size 100 --num-workers 20


* NOM_IDS_dictionary.txt
    https://hvdic.thivien.net/whv/%E7%B9%A3
    https://ctext.org/dictionary.pl?if=en&char=%E7%B3%B8
    http://www.nomfoundation.org/nom-tools/Nom-Lookup-Tool/Nom-Lookup-Tool?uiLang=vn
    https://www.cns11643.gov.tw/wordView.jsp?ID=1254469
    https://hanzii.net/search/word/%E5%AD%98?hl=en

# Dataset
dataroot => The path to train dataset (lmdb dataset)
ttfRoot => The path to the font file (depending on the language), used to create the input image for the generator
corpusRoot => The path to the file containing the words, used to generate the input image for the generator

# Model
* num_writer
- num_writer is a hyperparameter that controls the diversity of the training data by specifying the number of different writing styles.
- A higher num_writer means the model is trained on more diverse writing styles, allowing it to generate more varied and realistic samples.
- An embedding vector is learned for each writer/style, and concatenated to the input sequence embedding in the discriminator model. The discriminator can learn to identify different writers/styles.

* batch_size = 6
- batch_size is a hyperparameter that controls the number of samples processed in one iteration of training or testing. It impacts the speed and memory usage of the model.
- A higher batch_size means more samples are processed at once, leading to faster training but higher memory usage.
- The optimal value depends on the hardware and model.

* lr_scheduler: lr = 1.0 - max(0, epoch - niter) / float(niter_decay + 1)
- This is a common learning rate schedule for training GANs. It starts with a high learning rate to quickly converge, then decays it to fine-tune the model. The decay to 0 at the end of training helps stabilize the model. This schedule helps the model initially converge fast, then gradually fine-tune to reach its optimal performance.
- Keeps the learning rate at 1.0 for the first niter epochs.
- Then decays the learning rate linearly to 0 over the next niter_decay epochs.

* optimizers: adam
- This is a common optimizer

* gan_mode = lsgan
- The gan_mode parameter specifies the type of GAN objective used in the model. It can take the following values:
+ vanilla: The vanilla GAN loss is the cross-entropy loss used in the original GAN paper.
+ lsgan: The least-squares GAN loss. This loss minimizes the Pearson χ2 divergence between the generator and data distributions.
+ wgangp: The WGAN-GP loss. This loss uses the Wasserstein distance and gradient penalty to improve WGAN training stability.
- The gan_mode parameter impacts the model in the following ways:
+ vanilla GAN loss can be unstable in some cases.
+ lsgan loss is more stable but may converge slower.
+ wgangp loss is the most stable but also converges the slowest.

* max_length
- max_length is a hyperparameter that controls the maximum length of the input text sequences. This is used to pad shorter input text sequences to the same length, so they can be processed in batches by the AttentionRNN layer.
- If max_length is too high, it will pad shorter sequences with more padding tokens. The model may learn to ignore the excess padding, reducing efficiency.
- If max_length is too low, it cannot accommodate longer sequences and more sequences will need padding. The model will see less actual input.
- An ideal max_length is one that can accommodate most sequences with minimal padding.
- Need to analyze the length distribution of input text sequences and choose an appropriate value

* D_ch
- D_ch is a hyperparameter that controls the width and complexity of the discriminator network by defining the number of output channels in its first convolutional layer.
- A higher D_ch leads to a wider discriminator network with more parameters, allowing it to represent more complex functions to better distinguish real and fake samples.
- For the handwritten word synthesis task, D_ch controls the width of the discriminator used to distinguish real handwritten words from synthesized ones.

* G_ch
- G_ch is a hyperparameter that controls the width and complexity of the generator network by defining the number of output channels in its first convolutional layer.
- A higher G_ch leads to a wider generator network with more parameters, allowing it to represent more complex functions to generate higher quality and more diverse samples.
- For the handwritten word synthesis task, G_ch controls the width of the generator used to synthesize new handwritten words in different styles.

* imgW, imgH
- imgW and imgH refer to the width and height that all input images are resized to before being fed into the GAN model. Resizing the input images to a fixed size is a common preprocessing step when training GANs on image data.
- imgW and imgH would need to be tuned based on dataset and hardware.
