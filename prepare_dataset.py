import os
import shutil

lines = os.listdir('CGGANv2.2')
label_data = []

for line in lines:
    filename = line
    label = line.split('_')[0]
    label_data.append(filename + '\t' + label)
    shutil.copy('CGGANv2.2/'+filename, '/mnt/disk3/CGGANv2/' + filename)

with open('CGGAN_label.txt', 'w') as f:
    f.write('\n'.join(label_data).rstrip())