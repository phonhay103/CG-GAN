import os

folder_A = 'results/VN_handwriting_arial/2023-07-01_15-43-38'

lines = os.listdir(folder_A)
lines = [line + '\t' + line.split('_')[0] for line in lines]

with open('CGGAN_label.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines).rstrip())