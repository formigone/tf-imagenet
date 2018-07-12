import argparse
import os
import re
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('file', help='Input file to parse')
parser.add_argument('directory', help='Directory where images from input file will be found')

args = parser.parse_args()

BASE_PATH = args.directory

with open(args.file, 'r') as fp:
  for index, line in enumerate(fp.readlines()):
    if index == 0:
      print('ImageId,Width,Height,PredictionString')
      continue

    image, label = line.strip().split(',')

    filename_parts = re.split('^(n\d+)_', image)

    if len(filename_parts) == 3:
      image_path = '{}/train/{}/{}.JPEG'.format(BASE_PATH, filename_parts[1], image)
    else:
      image_path = '{}/val/{}.JPEG'.format(BASE_PATH, image)

    if os.path.exists(image_path):
      im = Image.open(image_path)
      print('{},{},{},{}'.format(image, im.size[0], im.size[1], label))
