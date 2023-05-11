python test_handwriting.py \
--dataroot data/datasets/train_VIE \
--ttfRoot data/fonts/VIE/arial.ttf \
--corpusRoot data/texts/VIE_all_chars.txt \
--name VIE_handwriting_arial \
--model handwritten \
--imgH 64 \
--imgW 384 \
--G_ch 64 \
--gpu_ids 3