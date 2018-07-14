const fs = require('fs');

const SAMPLES_PATH = `${__dirname}/../../../data/train`;
const LOG_PATH = `${__dirname}/../../../LOC_train_solution.csv`;
const SYNSET_PATH = `${__dirname}/../../../LOC_synset_mapping.txt`;

const mapping = {};
const synset = {};

fs.readFile(SYNSET_PATH, 'utf8', (err, data) => {
  if (err) {
    throw err;
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
});

fs.readFile(LOG_PATH, 'utf8', (err, data) => {
  if (err) {
    throw err;
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
          box: [Number(groups[i + 2]), Number(groups[i + 1]), Number(groups[i + 4]), Number(groups[i + 3])],
        });
      }
    }

    if (index === 1) {
      console.log('sample group: ' + JSON.stringify(group));
    }

    mapping[parts[0]] = group;
  })
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
              console.log(` ${dir}/${files[j]}`);
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
  getBox(path) {
    return new Promise((resolve, reject) => {
      if (path in mapping) {
        return resolve(mapping[path]);
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
