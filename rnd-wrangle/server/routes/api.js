const express = require('express');

const samples = require('../services/samples');

const router = express.Router();

router.get('/samples', (req, res) => {
  samples.fetch(250, 2)
    .then((images) => {
      res.json({ images });
    })
    .catch((err) => {
      res.status(500).json({ error: err });
    });
});

router.get('/samples/:img', async(req, res) => {
  let path = req.params.img;
  path = path.replace(/^\w+\//, '');
  path = path.replace(/\.\w+$/, '');

  try {
    const box = await samples.getBox(path);
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

module.exports = router;
