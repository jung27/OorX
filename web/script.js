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

// Hidden Layer 1
const hidden1 = [
  [
    0.2798897952708537, -0.7101009693416892, -0.24667992174612607,
    -0.34171991968753856,
  ],
  [
    0.3529813902171546, -0.07440900075902626, -0.32121251967516595,
    -0.46984346442514646,
  ],
  [
    0.6340358606997528, 1.3156903216844509, -0.7133558632936637,
    -1.3001935382816074,
  ],
  [
    0.008271876229769878, 0.6087978368916779, 0.514995007935623,
    0.2009104435480401,
  ],
];

// Hidden Layer 2
const hidden2 = [
  [
    0.08871848189395415, -0.5689582907503489, -0.955142963024508,
    -0.1979133498743954,
  ],
  [
    0.8221233835845243, 1.6018506890729123, -0.36069828002747967,
    -1.375198523721566,
  ],
  [
    -0.4476171993527761, 1.357254195084656, 0.7762002923751211,
    -1.5316649811402978,
  ],
  [
    -1.9242974681782672, 0.4447868268144433, 1.079721867351468,
    0.7430650401781987,
  ],
];

// Hidden Layer 3
const hidden3 = [
  [
    -1.0391410987715843, -0.229636245760308, -0.3601922154544508,
    0.8605639014302057,
  ],
  [
    0.917968416117192, 0.38158659497851255, 0.28700043579971923,
    -0.6100885790587203,
  ],
  [
    1.3856314537358367, -1.2438257571828302, -0.027536309638469054,
    0.5559264194115553,
  ],
  [
    1.190764076580569, -0.022634081437188532, -0.6913370120751189,
    0.10584267277391646,
  ],
];

// Thresholds (bias terms)
const threshold1 = 0.34159222060259664;
const threshold2 = 1.4132103727579428;
const threshold3 = 0.8334485115025386;

// Output Weights for Class 1
const outputWeights1 = [
  [
    [0.37375726526653835, -0.5363678311008445, 0.4283077589168812],
    [1.0343278297918517, -0.8444804508017821, -0.042997566173274576],
    [-0.5163489028071949, 1.4642570729280915, -0.4786682496512901],
  ],
  [
    [-1.1258746199035732, -0.471379542935155, 1.966466965114003],
    [1.2291819310461667, -1.2197012730819565, 0.1403932105170624],
    [0.7600301534256402, -1.0213378280751944, -0.43046174454710523],
  ],
  [
    [0.42408640043807405, 0.6274884308929285, -0.06839425147399347],
    [0.02623812914519316, 1.3037107937686674, -1.5289927310474918],
    [-0.7815358542536273, -0.46056089361956537, 0.01379645038395115],
  ],
  0.058376294326894715,
];

// Output Weights for Class 2
const outputWeights2 = [
  [
    [-0.029845325351763333, 0.25468754356902984, -0.03445228544490429],
    [-0.7690033753081142, 1.3925882887368475, 0.1380496755194754],
    [-1.4134115492176798, 0.5350332552880667, 0.7658659011972301],
  ],
  [
    [0.31151786929248393, 0.3195854039321444, -2.696328900176776],
    [-1.5195645005998601, 1.3591379029631776, -0.6483781864336293],
    [-0.5469305375378941, -0.6048691986526162, 1.3668063491305489],
  ],
  [
    [-1.753119367090852, -0.4697566780868695, 0.6142822972281212],
    [0.02110121483438354, -0.0718660014016003, 1.1915780093777526],
    [0.4680216335162428, -0.20695489787277377, -0.20427272831277626],
  ],
  -0.05643249028693938,
];

const judge = () => {
  const first = convolution(image, hidden1, threshold1);
  const second = convolution(image, hidden2, threshold2);
  const third = convolution(image, hidden3, threshold3);

  const maxPool1 = maxPooling(first);
  const maxPool2 = maxPooling(second);
  const maxPool3 = maxPooling(third);

  console.log(maxPool1);

  const result1 =
    sumProduct(maxPool1, outputWeights1[0], [0, 0]) +
    sumProduct(maxPool2, outputWeights1[1], [0, 0]) +
    sumProduct(maxPool3, outputWeights1[2], [0, 0]) +
    -1 * outputWeights1[3];
  const output1 = 1 / (1 + Math.exp(-result1));

  const result2 =
    sumProduct(maxPool1, outputWeights2[0], [0, 0]) +
    sumProduct(maxPool2, outputWeights2[1], [0, 0]) +
    sumProduct(maxPool3, outputWeights2[2], [0, 0]) +
    -1 * outputWeights2[3];
  const output2 = 1 / (1 + Math.exp(-result2));

  console.log(`Image Output:`);
  console.log(`  O Probability: ${output1 * 100}%`);
  console.log(`  X Probability: ${output2 * 100}%`);

  judgement.innerHTML = `This is ${output1 > output2 ? "'O'" : "'X'"}`;
};

const maxPooling = (image) => {
  const result = [];
  for (let i = 0; i < image.length; i += 2) {
    const row = [];
    for (let j = 0; j < image[0].length; j += 2) {
      const maxVal = Math.max(
        image[i][j],
        image[i][j + 1],
        image[i + 1][j],
        image[i + 1][j + 1]
      );
      row.push(maxVal);
    }
    result.push(row);
  }
  return result;
};

function sumProduct(image, kernel, pos) {
  const size = kernel.length;
  let output = 0;

  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const value = image[pos[1] + i][pos[0] + j] * kernel[i][j];
      output += value;
    }
  }

  return output;
}

function convolution(image, kernel, bias = 0) {
  const output = [];
  const imageHeight = image.length;
  const imageWidth = image[0].length;
  const kernelSize = kernel.length;

  for (let y = 0; y <= imageHeight - kernelSize; y++) {
    const row = [];
    for (let x = 0; x <= imageWidth - kernelSize; x++) {
      const value = sumProduct(image, kernel, [x, y]);
      const sigmoidValue = 1 / (1 + Math.exp(-1 * value + bias));
      row.push(sigmoidValue);
    }
    output.push(row);
  }

  return output;
}
