const fs = require('fs');

const SAMPLES_PATH = `${__dirname}/../../../data/train`;
const LOG_PATH = `${__dirname}/../../../LOC_train_solution.csv`;
const SYNSET_PATH = `${__dirname}/../../../LOC_synset_mapping.txt`;

const mapping = {};
const synset = {};

function parseSynsets(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) {
        return reject(err);
      }

      data.split('\n').forEach((line, index) => {
        if (!line) {
          return;
        }

        const parts = line.match(/^(n\d+)\s(.*?)$/);
        if (!parts || parts.length !== 3) {
          console.log(' bad parts: ' + line);
        }
        synset[parts[1]] = parts[2].split(',').map((word) => word.trim());
      });

      resolve(synset);
    });
  });
}

function parseObjects(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) {
        return reject(err);
      }

      data.split('\n').forEach((line, index) => {
        if (index === 0) {
          return;
        }

        //   'n02017213_1193,n02017213 80 150 381 455 n02017213 176 33 426 215 ',
        const parts = line.split(',');
        const group = { boxes: []};
        if (parts.length === 2) {
          const groups = parts[1].split(' ');
          for (let i = 0; i < groups.length; i += 5) {
            if (!groups[i]) {
              continue;
            }
            group.boxes.push({
              label: groups[i],
              // y_min, x_min, y_max, x_max
              box: [Number(groups[i + 2]), Number(groups[i + 1]), Number(groups[i + 4]), Number(groups[i + 3])],
            });
          }
        }

        mapping[parts[0]] = group;
      });

      resolve(mapping);
    });
  });
}

parseSynsets(SYNSET_PATH)
  .then(() => {
    console.log('Synsets ready');
    return parseObjects(LOG_PATH);
  })
  .then(() => {
    console.log('Boxes map ready');
  })
  .catch((err) => {
    console.error('Error parsing synsets', err);
  });

const samples = {
  fetch(numDirs = 5, maxPerDir = 3) {
    const images = [];
    return new Promise((resolve, reject) => {
      fs.readdir(SAMPLES_PATH, (err, items) => {
        if (err) {
          return reject(err);
        }

        const dirs = [];
        for (let i = 0; i < numDirs; i++) {
          if (items[i].match(/^n\d+$/)) {
            dirs.push(items[i]);
          }
        }

        dirs.forEach((dir) => {
          fs.readdir(`${SAMPLES_PATH}/${dir}`, (err, files) => {
            for (let j = 0; j < maxPerDir; j++) {
              images.push(`${dir}/${files[j]}`);
            }

            if (images.length >= dirs.length * maxPerDir) {
              resolve(images);
            }
          });
        });
      })
    });
  },
  getBox(path, width, height) {
    return new Promise((resolve, reject) => {
      if (path in mapping) {
        const boxes = { ...mapping[path] };

        boxes.path = path;
        boxes.boxes = boxes.boxes.map((box) => {
          box.synset = synset[box.label] || ['N/A'];
          box.pct = [box.box[0] / height, box.box[1] / width, box.box[2] / height, box.box[3] / width];
          box.rot90 = [box.pct[1], 1 - box.pct[2], box.pct[3], 1 - box.pct[0]];
          box.rot180 = [box.rot90[1], 1 - box.rot90[2], box.rot90[3], 1 - box.rot90[0]];
          box.rot270 = [box.rot180[1], 1 - box.rot180[2], box.rot180[3], 1 - box.rot180[0]];
          return box;
        });

        resolve(boxes);
      }

      reject(new Error('Mapping not found'));
    });
  },
  getSynset(id) {
    return new Promise((resolve) => {
      resolve(synset[id] || ['N/A']);
    });
  },
  synset() {
    return Promise.resolve(synset);
  },
};

module.exports = samples;
