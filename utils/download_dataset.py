import subprocess
import os
import shutil

subprocess.run(["wget", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
subprocess.run(["unzip", "tiny-imagenet-200.zip", "-d", "dataset/tiny-imagenet"])


with open('dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t') # filename, class, other fields
        os.makedirs(f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)
        # it creates a directory tiny-imagenet/tiny-imagenet-200/val/{cls} if it does not exist

        shutil.copyfile(f'dataset/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')
        # it copies the file tiny-imagenet/tiny-imagenet-200/val/images/{fn} in tiny-imagenet/tiny-imagenet-200/val/{cls}/

shutil.rmtree('dataset/tiny-imagenet/tiny-imagenet-200/val/images')