module.exports = function nn(sizes, options = {}) {
  let {
    learningRate = 0.4,
    momentum = 0.1,
    decay = 0.9999,
    weights = [],
    biases = [],
    activation = x => 1 / (1 + Math.exp(-x))
  } = options;
  const outputLayer = sizes.length - 1;

  const importingData = biases.length && weights.length;

  const outputs = [];
  const deltas = [];
  const changes = [];
  const errors = [];

  let timeout = null;

  const zeros = n => Array(n).fill(0);
  const randoms = n => zeros(n).map(e => Math.random());

  for (let layer = 0; layer <= outputLayer; layer++) {
    let size = sizes[layer];
    deltas[layer] = zeros(size);
    errors[layer] = zeros(size);
    outputs[layer] = zeros(size);
    if (layer > 0) {
      if (!importingData) {
        biases[layer] = randoms(size);
        weights[layer] = new Array(size);
      }
      changes[layer] = new Array(size);
      for (let node = 0; node < size; node++) {
        let prevSize = sizes[layer - 1];
        if (!importingData) {
          weights[layer][node] = randoms(prevSize);
        }
        changes[layer][node] = zeros(prevSize);
      }
    }
  }

  function run(input) {
    outputs[0] = input;
    let output = null;
    for (let layer = 1; layer <= outputLayer; layer++) {
      for (let node = 0; node < sizes[layer]; node++) {
        let weight = weights[layer][node];
        let sum = biases[layer][node];
        for (let k = 0; k < weight.length; k++) {
          sum += weight[k] * input[k];
        }
        outputs[layer][node] = activation(sum);
      }
      output = input = outputs[layer];
    }
    return output;
  }

  function calculateDeltas(target) {
    for (let layer = outputLayer; layer >= 0; layer--) {
      for (let node = 0; node < sizes[layer]; node++) {
        let output = outputs[layer][node];
        let error = 0;
        if (layer === outputLayer) {
          error = target[node] - output;
        } else {
          let delta = deltas[layer + 1];
          for (let k = 0; k < delta.length; k++) {
            error += delta[k] * weights[layer + 1][k][node];
          }
        }
        errors[layer][node] = error;
        deltas[layer][node] = error * output * (1 - output);
      }
    }
  }

  function adjustWeights() {
    for (let layer = 1; layer <= outputLayer; layer++) {
      let incoming = outputs[layer - 1];

      for (let node = 0; node < sizes[layer]; node++) {
        let delta = deltas[layer][node];

        for (let k = 0; k < incoming.length; k++) {
          let change = changes[layer][node][k];

          change = learningRate * delta * incoming[k] + momentum * change;

          changes[layer][node][k] = change;
          weights[layer][node][k] += change;
        }
        biases[layer][node] += learningRate * delta;
      }
    }
  }

  function trainPattern(value) {
    run(value.input);
    calculateDeltas(value.output);
    adjustWeights();
  }

  function trainingTick(trainingSet, iterations, stopError, log, done) {
    learningRate = learningRate * decay;
    let errorTotal = 0;
    for (let i = 0; i < trainingSet.length; i++) {
      trainPattern(trainingSet[i]);
      errorTotal += Math.abs(errors[outputLayer][0]);
    }
    const averageError = errorTotal / trainingSet.length;
    if (log && iterations % 100 === 0) log(averageError);
    if (--iterations === 0 || averageError < stopError) {
      return done(averageError);
    }
    timeout = setTimeout(
      () => trainingTick(trainingSet, iterations, stopError, log, done),
      0
    );
  }

  function train(trainingSet, options = {}) {
    const {
      iterations = 20000,
      stopError = 0.05,
      log,
      done = () => 0
    } = options;
    trainingTick(trainingSet, iterations, stopError, log, done);
  }
  return {
    train,
    run,
    stop: () => clearTimeout(timeout),
    export: () => ({ weights, biases })
  };
};
