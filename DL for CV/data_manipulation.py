'''
in the dogs-vs-cats dataset, split the data so that the training set contains 1k images
of each class, the validation set contains 500 images of each class and the testing set
contains 500 images of each class. Note: run this only once
'''

import os, shutil

original_dataset_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats\\train'

train_size = 1000
val_size = 500
test_size = 500

# create directories
base_dir = 'C:\\Users\\ishik\\le code\\DL in python\\dogs-vs-cats-small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

val_dir = os.path.join(base_dir, 'validation')
os.mkdir(val_dir)
val_cats_dir = os.path.join(val_dir, 'cats')
os.mkdir(val_cats_dir)
val_dogs_dir = os.path.join(val_dir, 'dogs')
os.mkdir(val_dogs_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# copy images from the original directory to the new directory

# copy 1k training images
fnames=['cat.{}.jpg'.format(i) for i in range(train_size)]
for f in fnames:
    src = os.path.join(original_dataset_dir, f)
    dst = os.path.join(train_cats_dir, f)
    shutil.copyfile(src, dst)

fnames=['dog.{}.jpg'.format(i) for i in range(train_size)]
for f in fnames:
    src = os.path.join(original_dataset_dir, f)
    dst = os.path.join(train_dogs_dir, f)
    shutil.copyfile(src, dst)

# copy 500 validation images
end_ind = train_size + val_size

fnames=['cat.{}.jpg'.format(i) for i in range(train_size, end_ind)]
for f in fnames:
    src = os.path.join(original_dataset_dir, f)
    dst = os.path.join(val_cats_dir, f)
    shutil.copyfile(src, dst)

fnames=['dog.{}.jpg'.format(i) for i in range(train_size, end_ind)]
for f in fnames:
    src = os.path.join(original_dataset_dir, f)
    dst = os.path.join(val_dogs_dir, f)
    shutil.copyfile(src, dst)

# copy 500 test images
start_ind = end_ind
end_ind = train_size + val_size + test_size

fnames=['cat.{}.jpg'.format(i) for i in range(start_ind, end_ind)]
for f in fnames:
    src = os.path.join(original_dataset_dir, f)
    dst = os.path.join(test_cats_dir, f)
    shutil.copyfile(src, dst)

fnames=['dog.{}.jpg'.format(i) for i in range(start_ind, end_ind)]
for f in fnames:
    src = os.path.join(original_dataset_dir, f)
    dst = os.path.join(test_dogs_dir, f)
    shutil.copyfile(src, dst)
