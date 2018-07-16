function drawBoundingBox(event) {
  const image = event.target;
  const path = image.src.replace(/.*?\/img\/train\//, '');
  const width = image.naturalWidth;
  const height = image.naturalHeight;

  fetch(`/api/samples/${encodeURIComponent(path)}/${width}/${height}`)
    .then((res) => res.json())
    .then(({ boxes }) => {
      if (!Array.isArray(boxes)) {
        throw new Error('Could not find list of boxes');
      }

      boxes.forEach(({ box, synset, pct }) => {
        if (!Array.isArray(box) || box.length !== 4) {
          throw new Error('Invalid box');
        }

        image.parentNode.setAttribute('data-loc', `${path} ${width} ${height} ${pct}`);

        // Top
        const boundingBox = dfrag('div', { className: 'img-bb' });
        // const topProps = { top: `${box[0]}px`, left: `${box[1]}px`, height: '3px', width: `${box[3] - box[1]}px` };
        const topProps = { top: `${pct[0] * 100}%`, left: `${pct[1] * 100}%`, height: `${pct[2] * 100 - pct[0] * 100}%`, width: `${pct[3] * 100 - pct[1] * 100}%` };
        for (let prop in topProps) {
          boundingBox.style[prop] = topProps[prop];
        }

        image.parentNode.appendChild(boundingBox);
        boundingBox.appendChild(dfrag('span', { className: 'label' }, `${synset[0]}`));
      });
    })
    .catch((err) => {
      console.error('Error drawing bounding box', err);
    });
}

function boundingBox() {
  const images = document.querySelectorAll('img');
  images.forEach((image) => {
    ['rot270', 'rot180', 'rot90'].forEach((className) => {
      const clone = image.cloneNode();
      clone.classList.add(className);
      clone.addEventListener('load', drawBoundingBox);
      image.parentNode.parentNode.insertBefore(dfrag('div', { className: 'img-wrapper' }, clone), image.parentNode.nextSibling);
    });

    image.addEventListener('load', drawBoundingBox);
  });
}

fetch('/api/samples/3/2')
  .then((res) => res.json())
  .then((samples) => {
    const images = dfrag('div', { className: 'container' }, (
      samples.images.map((sample) => dfrag('div', { className: 'img-wrapper' }, dfrag('img', {
        src: `/img/train/${sample}`,
      }), {}))));
    document.body.appendChild(images);

    boundingBox();
  });