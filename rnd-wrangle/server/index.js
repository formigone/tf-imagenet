const express = require('express');
const bodyParser = require('body-parser');

const app = express();

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use('/public', express.static(`${__dirname}/../public`));
app.use('/img', express.static(`${__dirname}/../../data`));

const staticDirs = ['/weights-vis', '/feature-maps', '/weights-vis-bn', '/feature-maps-bn', '/data'];

staticDirs.forEach((path) => {
  app.use(path, express.static(`${__dirname}/../../${path}`, {
    setHeaders: (res, path) => {
      console.log('req ' + path);
      res.set('Cache-Control', 'public max-age=31557600');
    },
  }));
});

app.use('/api', require('./routes/api'));

app.get('/conv', (req, res) => {
  res.sendFile(`${__dirname}/views/conv.html`);
});

app.get('/val', (req, res) => {
  res.sendFile(`${__dirname}/views/val.html`);
});

app.get('/train', (req, res) => {
  res.sendFile(`${__dirname}/views/train.html`);
});

app.get('/', (req, res) => {
  res.sendFile(`${__dirname}/views/index.html`);
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
