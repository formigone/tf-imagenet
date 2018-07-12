import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('file', help='Input file to parse')
parser.add_argument('directory', help='Directory where images from input file will be found')

args = parser.parse_args()

BASE_PATH = args.directory

with open(args.file, 'r') as fp:
  for index, line in enumerate(fp.readlines()):
    if index == 0:
      continue

    image, label = line.strip().split(',')

    filename_parts = re.split('^(n\d+)_', image)

    if len(filename_parts) == 3:
      image_path = '{}/train/{}/{}.JPEG'.format(BASE_PATH, filename_parts[1], image)
      if not os.path.exists(image_path):
        cmd = 'gsutil cp gs://ddm-imagenet/ILSVRC/Data/CLS-LOC/train/{}/{}.JPEG data/train/{}/{}.JPEG'.format(filename_parts[1], image, filename_parts[1], image)
        os.system(cmd)
      else:
        print(' >> Skipping {}'.format(image))
    else:
      image_path = '{}/val/{}.JPEG'.format(BASE_PATH, image)
      if not os.path.exists(image_path):
        cmd = 'gsutil cp gs://ddm-imagenet/ILSVRC/Data/CLS-LOC/val/{}.JPEG data/val/{}.JPEG'.format(image, image)
        os.system(cmd)
      else:
        print(' >> Skipping {}'.format(image))
