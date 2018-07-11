import random

with open('sample_goog.csv', 'r') as file:
  lines = file.readlines()
  for i, line in enumerate(lines):
    parts = line.strip().split(',')
    if i > 0 and random.randint(1, 10) < 8:
      print('{},'.format(parts[0]) if parts[0] else line.strip())
    else:
      print(line.strip())
