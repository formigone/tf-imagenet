const express = require('express');
const fs = require('fs');

const samples = require('../services/samples');

const router = express.Router();

const NUM_DIRS = 25;
const PER_DIR = 4;

router.get('/samples/:numDirs?/:perDir?', (req, res) => {
  samples.fetch(Number(req.params.numDirs || NUM_DIRS), Number(req.params.perDir || PER_DIR))
    .then((images) => {
      res.json({ images });
    })
    .catch((err) => {
      res.status(500).json({ error: err });
    });
});

router.get('/samples/:img/:width/:height', async(req, res) => {
  let path = req.params.img;
  path = path.replace(/^\w+\//, '');
  path = path.replace(/\.\w+$/, '');
  const { width, height } = req.params;

  try {
    const box = await samples.getBox(path, Number(width), Number(height));
    res.json(box);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/synset', async(req, res) => {
  const synset = await samples.synset();
  res.json({ synset });
});

router.get('/synset/:id', async(req, res) => {
  const synset = await samples.getSynset(req.params.id);
  res.json({ synset });
});

router.get('/ls/:path', (req, res) => {
  fs.readdir(`${__dirname}/../../../${decodeURIComponent(req.params.path)}`, (err, out) => {
    if (err) {
      return res.json({ error: err.message });
    }

    res.json({ out });
  });
});

module.exports = router;
