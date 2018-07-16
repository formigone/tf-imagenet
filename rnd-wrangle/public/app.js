function boundingBox() {
  const images = document.querySelectorAll('img');
  images.forEach((image) => {
    image.addEventListener('load', () => {
      const path = image.src.replace(/.*?\/img\/train\//, '');
      const synsetId = path.replace(/\/.*?$/, '');
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
            const topLine = dfrag('div', { className: 'img-bb' });
            // const topProps = { top: `${box[0]}px`, left: `${box[1]}px`, height: '3px', width: `${box[3] - box[1]}px` };
            const topProps = { top: `${pct[0] * 100}%`, left: `${pct[1] * 100}%`, height: '3px', width: `${pct[3] * 100 - pct[1] * 100}%` };
            for (let prop in topProps) {
              topLine.style[prop] = topProps[prop];
            }

            image.parentNode.appendChild(topLine);

            // Bottom
            const bottomLine = dfrag('div', { className: 'img-bb' });
            // const bottomProps = { top: `${box[2]}px`, left: `${box[1]}px`, height: '3px', width: `${box[3] - box[1] + 3}px` };
            const bottomProps = { top: `${pct[2] * 100}%`, left: `${pct[1] * 100}%`, height: '3px', width: `${pct[3] * 100 - pct[1] * 100}%` };
            for (let prop in bottomProps) {
              bottomLine.style[prop] = bottomProps[prop];
            }

            image.parentNode.appendChild(bottomLine);

            // Left
            const leftLine = dfrag('div', { className: 'img-bb' });
            // const leftProps = { top: `${box[0]}px`, left: `${box[1]}px`, height: `${box[2] - box[0]}px`, width: '3px' };
            const leftProps = { top: `${pct[0] * 100}%`, left: `${pct[1] * 100}%`, height: `${pct[2] * 100 - pct[0] * 100}%`, width: '3px' };
            for (let prop in leftProps) {
              leftLine.style[prop] = leftProps[prop];
            }

            image.parentNode.appendChild(leftLine);

            // Right
            const rightLine = dfrag('div', { className: 'img-bb' });
            // const rightProps = { top: `${box[0]}px`, left: `${box[3]}px`, height: `${box[2] - box[0]}px`, width: '3px' };
            const rightProps = { top: `${pct[0] * 100}%`, left: `${pct[3] * 100}%`, height: `${pct[2] * 100 - pct[0] * 100}%`, width: '3px' };
            for (let prop in rightProps) {
              rightLine.style[prop] = rightProps[prop];
            }

            image.parentNode.appendChild(rightLine);
            image.parentNode.appendChild(dfrag('span', { className: 'label', style: `left: ${pct[1] * 100}%; top: ${pct[0] * 100}%;` }, `${synset[0]}`));
          });
        })
        .catch((err) => {
          console.error('Error drawing bounding box', err);
        });
    });
  });
}

fetch('/api/samples')
  .then((res) => res.json())
  .then((samples) => {
    const images = dfrag('div', { className: 'container' }, (
      samples.images.map((sample) => dfrag('div', { className: 'img-wrapper' }, dfrag('img', {
        src: `/img/train/${sample}`,
      }), {}))));
    document.body.appendChild(images);

    boundingBox();
  });