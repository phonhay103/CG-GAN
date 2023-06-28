# def nom_to_unicode(nom_char):
#     return hex(ord(nom_char)).lstrip('0x').upper()

# with open('data/dictionaries/NOM_dictionary.txt') as f:
#     radical_data = f.read().splitlines()

# with open('all_images.txt') as f:
#     unicode_data = f.read().splitlines()

# word = radical_data[0].split(':')[0]
# print(nom_to_unicode(word))

# import json
# with open('unicode_to_nom_dict.json') as f:
#     unicode_to_json_dict = json.load(f)

# print(unicode_to_json_dict['48EB'])

# dictionary_dir = 'data/dictionaries/NOM_dictionary.txt'
# with open(dictionary_dir) as f:
#     radical_data = f.read().splitlines()

# nom_to_radical_dict = dict()
# for line in radical_data:
#     label, radical = line.split(':')[0], line.split(':')[1]
#     nom_to_radical_dict[label] = radical