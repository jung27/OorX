// model_utils.js

async function loadModel(path = "model.json") {
  const resp = await fetch(path);
  const mdl = await resp.json();
  console.log("🔍 loaded model keys:", Object.keys(mdl));
  console.log(
    "🔍 conv_kernels shape:",
    mdl.conv_kernels.length,
    mdl.conv_kernels[0].length,
    mdl.conv_kernels[0][0].length
  );
  return {
    convKernels: mdl.conv_kernels, // [C][3][3]
    convBiases: mdl.conv_biases, // [C]
    fcW: mdl.fc_w, // [featDim]
    fcB: mdl.fc_b, // scalar
  };
}

function relu(x) {
  return x > 0 ? x : 0;
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function conv2d(image, kernel, bias) {
  const H = image.length,
    W = image[0].length,
    K = kernel.length;
  const out = [];
  for (let y = 0; y <= H - K; y++) {
    const row = [];
    for (let x = 0; x <= W - K; x++) {
      let sum = 0;
      for (let i = 0; i < K; i++)
        for (let j = 0; j < K; j++) sum += image[y + i][x + j] * kernel[i][j];
      row.push(sum - bias);
    }
    out.push(row);
  }
  return out;
}

function maxPool2d(matrix, size = 2, stride = 2) {
  const H = matrix.length,
    W = matrix[0].length;
  const oh = Math.floor((H - size) / stride) + 1;
  const ow = Math.floor((W - size) / stride) + 1;
  const pooled = Array.from({ length: oh }, () => Array(ow).fill(0));
  for (let i = 0; i < oh; i++) {
    for (let j = 0; j < ow; j++) {
      let m = -Infinity;
      for (let di = 0; di < size; di++)
        for (let dj = 0; dj < size; dj++)
          m = Math.max(m, matrix[i * stride + di][j * stride + dj]);
      pooled[i][j] = m;
    }
  }
  return pooled;
}

async function predict(image, modelPath = "model.json") {
  // image: 9×9 array of 0/1
  const { convKernels, convBiases, fcW, fcB } = await loadModel(modelPath);

  console.log("🔍 First kernel:", convKernels[0]); // kernel[0] 실제 값 체크

  const C = convKernels.length;
  let feats = [];

  for (let c = 0; c < C; c++) {
    // 1) conv
    let fm = conv2d(image, convKernels[c], convBiases[c]);
    console.log(`🖼︎ feature map ${c}[0][0]=`, fm[0][0]);
    // 2) relu
    fm = fm.map((row) => row.map(relu));
    // 3) pool
    const pm = maxPool2d(fm, 2, 2);
    console.log(`🔢 pooled ${c}[0][0]=`, pm[0][0]);
    // 4) flatten
    for (const row of pm) feats.push(...row);
  }

  // 5) FC → logit
  let logit = feats.reduce((acc, v, i) => acc + v * fcW[i], 0) + fcB;
  console.log("📊 logit:", logit);
  const prob = sigmoid(logit);
  console.log("🎯 sigmoid(logit):", prob);

  return prob;
}
