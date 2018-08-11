import subprocess


def get_cmd(buffer, batch, map_first, num_parallel_calls, prefetch):
  return ['python',
          'models/inception_v2_bn.py',
          '--model_dir', 'model_dir/bmark_buf{0}_bch{1}_map{2}_thds{3}_pfet{4}_dp3_lr01'.format(buffer, batch, map_first, num_parallel_calls, prefetch),
          '--train_set', 'data/csv/train/cool_pets_224x224.csv',
          '--val_set', 'data/csv/val/cool_pets_224x224.csv',
          '--img_width', '224',
          '--img_height', '224',
          '--num_classes', '42',
          '--epochs', '1',
          '--max_steps', '500',
          '--buffer_size', str(buffer),
          '--batch_size', str(batch),
          '--map_first', str(map_first),
          '--num_parallel_calls', str(num_parallel_calls),
          '--prefetch', str(prefetch),
          '--norm_2', '1',
          '--dropout_keep_prob', '0.3',
          '--learning_rate', '0.01',
          ]


buffer = [64, 128, 256]
batch = [4, 8, 16, 32, 64, 128]
num_parallel_calls = [0, 2, 4, 8]
prefetch = [0, 1, 2, 4]
map_first = [0, 1]

for m in map_first:
        for buf in buffer:
                for bch in batch:
                        for thds in num_parallel_calls:
                                for pfet in prefetch:
                                        print('* * * * * * *')
                                        print('map {} buf {} bch {} thds {} pfet {}'.format(m, buf, bch, thds, pfet))
                                        cmd = get_cmd(buf, bch, m, thds, pfet)
                                        subprocess.call(cmd)
