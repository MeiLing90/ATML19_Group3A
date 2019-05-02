# Just here as a backup, in case we want to switch to the ImageFolder Dataloader

import os
import glob
import os
import glob
import random

fileList = glob.glob('./data/dataset5/**/**/depth*', recursive=True)
for filePath in fileList:
    os.remove(filePath)

original_path = os.path.join('.', 'data', 'dataset5')
labels = ['r', 'u', 'i', 'n', 'g', 't', 's', 'a', 'f', 'o', 'h', 'm', 'c', 'd', 'v', 'q', 'x', 'e', 'b', 'k',
                  'l', 'y', 'p', 'w']

if os.path.exists('dataset5'):
    os.rename(os.path.join(original_path, 'E'), os.path.join(original_path, 'test'))
    for person in ['A','B', 'C', 'D']:
        for label in labels:
            fileList = os.listdir(os.path.join(original_path, person, label))
            for file in fileList:
                if not file.startswith('.'):
                    parts = file.split('.')
                    os.rename(os.path.join(original_path, person, label,file),
                              os.path.join(original_path, 'A', label, parts[0] + "_" + person + '.png'))
    os.rename(os.path.join(original_path, 'A'), os.path.join(original_path, 'train'))
    os.rename(os.path.join(original_path, 'B'), os.path.join(original_path, 'validate'))
    for label in labels:
        fileList = os.listdir(os.path.join(original_path, 'train', label))
        samples_per_label = int((len(fileList) / len(labels))*0.1)
        sample_files = random.sample(fileList, samples_per_label)
        for file in sample_files:
            os.rename(os.path.join(original_path, 'train', label, file),
                      os.path.join(original_path, 'validate', label, file))
    os.rename(original_path, os.path.join('.', 'data', 'full'))

# Create a sampled version
os.mkdir(os.path.join('.', 'data', 'sample'))
# TODO: Create sampled version