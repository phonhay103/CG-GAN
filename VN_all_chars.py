with open('/mnt/disk1/naver/nl/vietocr/datasets/labels/train.txt') as f:
    train_list = f.read().splitlines()

with open('/mnt/disk1/naver/nl/vietocr/datasets/labels/valid.txt') as f:
    valid_list = f.read().splitlines()

with open('/mnt/disk1/naver/nl/vietocr/datasets/labels/test.txt') as f:
    test_list = f.read().splitlines()

char_list = set(map(lambda img_path: img_path.split('\t')[1], train_list+valid_list+test_list))
new_set = set()
for char in char_list:
    new_set.add(char.lower())
    new_set.add(char.upper())
    new_set.add(char.capitalize())

with open('data/texts/VN_all_chars.txt', 'w') as f:
    f.write('\n'.join(new_set).rstrip('\n'))