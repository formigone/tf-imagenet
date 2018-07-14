function boundingBox() {
  const images = document.querySelectorAll('img');
  images.forEach((image) => {
    const path = image.src.replace(/.*?\/img\/train\//, '');
    const synsetId = path.replace(/\/.*?$/, '');
    const imgId = path.replace(/.*?\//, '').replace(/\..*?$/, '');

    Promise.all([fetch(`/api/samples/${encodeURIComponent(path)}`), fetch(`/api/synset/${synsetId}`)])
      .then((res) => Promise.all([res[0].json(), res[1].json()]))
      .then(([{ boxes }, { synset }]) => {
        if (!Array.isArray(boxes)) {
          throw new Error('Could not find list of boxes');
        }

        boxes.forEach(({ box, label }) => {
          if (!Array.isArray(box) || box.length !== 4) {
            throw new Error('Invalid box');
          }

          // Top
          const topLine = dfrag('div', { className: 'img-bb' });
          const topProps = { top: `${box[0]}px`, left: `${box[1]}px`, height: '3px', width: `${box[3] - box[1]}px` };
          for (let prop in topProps) {
            topLine.style[prop] = topProps[prop];
          }

          image.parentNode.appendChild(topLine);

          // Bottom
          const bottomLine = dfrag('div', { className: 'img-bb' });
          const bottomProps = { top: `${box[2]}px`, left: `${box[1]}px`, height: '3px', width: `${box[3] - box[1] + 3}px` };
          for (let prop in bottomProps) {
            bottomLine.style[prop] = bottomProps[prop];
          }

          image.parentNode.appendChild(bottomLine);

          // Left
          const leftLine = dfrag('div', { className: 'img-bb' });
          const leftProps = { top: `${box[0]}px`, left: `${box[1]}px`, height: `${box[2] - box[0]}px`, width: '3px' };
          for (let prop in leftProps) {
            leftLine.style[prop] = leftProps[prop];
          }

          image.parentNode.appendChild(leftLine);

          // Right
          const rightLine = dfrag('div', { className: 'img-bb' });
          const rightProps = { top: `${box[0]}px`, left: `${box[3]}px`, height: `${box[2] - box[0]}px`, width: '3px' };
          for (let prop in rightProps) {
            rightLine.style[prop] = rightProps[prop];
          }

          image.parentNode.appendChild(rightLine);
          image.parentNode.appendChild(dfrag('span', { className: 'label', style: `left: ${box[1]}px; top: ${box[0] - 23}px;` }, `${synset[0]}`));
        });
      })
      .catch((err) => {
        console.error('Error drawing bounding box', err);
      })
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