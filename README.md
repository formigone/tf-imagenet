# tf-imagenet

### The challenge

[https://www.kaggle.com/c/imagenet-object-localization-challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)

### To do

 - [x] Upload all images to Google cloud storage
 - [ ] Create some plumbing for training/eval and testing
 - [ ] Create a model
 - [ ] Write a csv-tfrecord converter
 - [ ] Write tfrecord generator from list of files (for test images)
 - [ ] Write a submission formatter

### Datasets

| Type | Size |
|---|---|
| Train | 544,547 |
| Validation | 50,000 |
| Test | 100,000 |

 * All images are contained in the `ILSVRC` bucket under `~/ILSVRC/Data/CLS-LOC/[train|val|test]/<wnid>/<wnid>_<count>.JPEG`.
 * Description for training/val each image can be found in `LOC_[train|val]_solution.csv`.

##### Sample LOC_*_solution record

````
ILSVRC2012_val_00008726,n02119789 255 142 454 329 n02119789 44 21 322 295
````

The above is interpreted as:

 * Describes image `~/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00049992.JPEG`
 * Two objects are labeled in that image:
    * Object n02119789, with bounding box at (255, 142) and (454, 329)
    * Object n02119789, with bounding box at (44, 21) and (322, 295)

A mapping of all 1,000 classes to be detected is found in `LOC_synset_mapping.txt`.

