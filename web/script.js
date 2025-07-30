const grid = document.querySelector(".container");
const items = grid.querySelector(".items");
const judgement = document.querySelector(".judgment");
let mousedown = false;

const image = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
];

for (let i = 0; i < 80; i++) {
  let item = items.cloneNode(true);
  let y = (i % 9) + 1;
  let x = Math.floor(i / 9);
  item.id = `${x}${y}`;

  if (i % 9 === 7) {
    item.classList.add("right-border");
  }
  if (i > 70) {
    item.classList.add("bottom-border");
  }
  grid.appendChild(item);
}

grid.addEventListener("mousedown", (e) => {
  mousedown = true;
});

grid.addEventListener("mouseup", (e) => {
  mousedown = false;
});

const paint = (target) => {
  target.classList.add("painted");
  const id = target.id;

  const x = parseInt(id[0]);
  const y = parseInt(id[1]);
  if (image[x][y] === 0) {
    image[x][y] = 1;
  }

  judge();
};

const clearCanvas = () => {
  const items = grid.querySelectorAll(".items");
  items.forEach((item) => {
    item.classList.remove("painted");
  });
  for (let i = 0; i < image.length; i++) {
    for (let j = 0; j < image[i].length; j++) {
      image[i][j] = 0;
    }
  }
  judge();
};

const judge = () => {
  predict(image, "model.json").then((prob) => {
    console.log("O일 확률:", 1 - prob);
    judgement.innerHTML = `This is ${1 - prob > 0.5 ? "'O'" : "'X'"}`;
  });
};
