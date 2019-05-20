import os
import glob
import shutil

fileList = glob.glob('./data/dataset5/**/**/depth*', recursive=True)
for filePath in fileList:
    if os.path.exists(filePath):
        os.remove(filePath)

original_path = os.path.join('.', 'data', 'dataset5')
labels = ['r', 'u', 'i', 'n', 'g', 't', 's', 'a', 'f', 'o', 'h', 'm', 'c', 'd', 'v', 'q', 'x', 'e', 'b', 'k',
          'l', 'y', 'p', 'w']

if os.path.exists('data/dataset5'):
    for person in ['A', 'B', 'C', 'D']:
        for label in labels:
            fileList = os.listdir(os.path.join(original_path, person, label))
            for file in fileList:
                if not file.startswith('.'):
                    parts = file.split('.')
                    os.rename(os.path.join(original_path, person, label,file),
                              os.path.join(original_path, 'A', label, parts[0] + "_" + person + '.png'))

    os.rename(os.path.join(original_path, 'A'), os.path.join('.', 'data/train'))
    os.rename(os.path.join(original_path, 'E'), os.path.join('.', 'data/test'))
    shutil.rmtree(os.path.join('.', 'data/dataset5'))

if os.path.exists('data/asl_alphabet_train'):
    shutil.rmtree(os.path.join('.', 'data/asl_alphabet_train/del'))
    shutil.rmtree(os.path.join('.', 'data/asl_alphabet_train/J'))
    shutil.rmtree(os.path.join('.', 'data/asl_alphabet_train/nothing'))
    shutil.rmtree(os.path.join('.', 'data/asl_alphabet_train/space'))
    shutil.rmtree(os.path.join('.', 'data/asl_alphabet_train/Z'))
    dirs = os.listdir('data/asl_alphabet_train')
    for directory in dirs:
        os.rename(os.path.join('.', 'data/asl_alphabet_train', directory), os.path.join('.', 'data/asl_alphabet_train', directory).lower())
    os.rename(os.path.join('.', 'data/asl_alphabet_train'), os.path.join('.', 'data/test2'))
