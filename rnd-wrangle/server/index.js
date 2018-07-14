const express = require('express');
const bodyParser = require('body-parser');

const app = express();

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use('/public', express.static(`${__dirname}/../public`));
app.use('/img', express.static(`${__dirname}/../../data`));

app.get('/', (req, res) => {
  res.sendFile(`${__dirname}/views/index.html`);
});

app.use('/api', require('./routes/api'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});