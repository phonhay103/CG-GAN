python VN_train.py \
--dataroot data/datasets/VN_train \
--ttfRoot data/fonts/VN/arial.ttf \
--corpusRoot data/texts/VN_all_chars.txt \
--dictionaryRoot data/dictionaries/VN_dictionary.txt \
--name VN_handwriting_arial \
--model handwritten \
--batch_size 6 \
--num_threads 12 \
--num_writer 1 \
--gpu_ids 3 \
--lr 0.0001 \
--niter 15 \
--niter_decay 30 \
--imgH 64 \
--G_ch 64 \
--imgW 384 \
--max_length 96 \
--D_ch 64
# --epoch 45 \
# --continue_train