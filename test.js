require('should');
const nn = require('./index');

const delta = 0.15;

describe('nn1k', () => {
  it('should learn AND with enough iterations', done => {
    const net = nn([2, 4, 1]);
    net.train(
      [
        { input: [1, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] }
      ],
      {
        done: () => {
          net.run([1, 1])[0].should.be.approximately(1, delta);
          net.run([1, 0])[0].should.be.approximately(0, delta);
          net.run([0, 1])[0].should.be.approximately(0, delta);
          net.run([0, 0])[0].should.be.approximately(0, delta);
          done();
        }
      }
    );
  });
  it('should learn AND with stop error', done => {
    const net = nn([2, 4, 1]);
    net.train(
      [
        { input: [1, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] }
      ],
      {
        iterations: 10000,
        done: () => {
          net.run([1, 1])[0].should.be.approximately(1, delta);
          net.run([1, 0])[0].should.be.approximately(0, delta);
          net.run([0, 1])[0].should.be.approximately(0, delta);
          net.run([0, 0])[0].should.be.approximately(0, delta);
          done();
        }
      }
    );
  });
  it('should learn OR', done => {
    const net = nn([2, 4, 1]);
    net.train(
      [
        { input: [1, 0], output: [1] },
        { input: [0, 1], output: [1] },
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [1] }
      ],
      {
        done: () => {
          net.run([1, 1])[0].should.be.approximately(1, delta);
          net.run([1, 0])[0].should.be.approximately(1, delta);
          net.run([0, 1])[0].should.be.approximately(1, delta);
          net.run([0, 0])[0].should.be.approximately(0, delta);
          done();
        }
      }
    );
  });
  it('should learn XOR', done => {
    const net = nn([2, 4, 1]);
    net.train(
      [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] }
      ],
      {
        stopError: 0.08,
        done: () => {
          net.run([1, 1])[0].should.be.approximately(0, delta);
          net.run([1, 0])[0].should.be.approximately(1, delta);
          net.run([0, 1])[0].should.be.approximately(1, delta);
          net.run([0, 0])[0].should.be.approximately(0, delta);
          done();
        }
      }
    );
  });
  it('should learn NOT', done => {
    const net = nn([1, 3, 1], { learningRate: 0.3 });
    net.train([{ input: [1], output: [0] }, { input: [0], output: [1] }], {
      done: () => {
        net.run([1])[0].should.be.approximately(0, delta);
        net.run([0])[0].should.be.approximately(1, delta);
        done();
      }
    });
  });
  it('should learn async', done => {
    const net = nn([2, 4, 1]);
    net.train(
      [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] }
      ],
      {
        stopError: 0.08
      }
    );
    setTimeout(() => {
      net.run([1, 0])[0].should.be.above(0.4);
    }, 1000);
    setTimeout(() => {
      net.run([1, 0])[0].should.be.above(0.6);
      net.stop();
      done();
    }, 7000);
  });
  it('should be able to export trained data', done => {
    const net = nn([1, 3, 1]);
    net.train([{ input: [1], output: [0] }, { input: [0], output: [1] }], {
      done: () => {
        net.export().should.have.properties('weights', 'biases');
        done();
      }
    });
  });
  it('should be able to import trained data', () => {
    const data =
      '{"weights":[null,[[-2.5555392951172373],[-4.410859091147952],[2.605878954921416]],[[2.6800414448987753,5.691000454534446,-3.626756893929223]]],"biases":[null,[0.6509664299737214,1.7661964050674284,-0.5557365136930482],[-1.3419994300048395]]}';
    const net = nn([1, 3, 1], JSON.parse(data));
    net.run([1])[0].should.be.approximately(0, delta);
    net.run([0])[0].should.be.approximately(1, delta);
  });
  it('should be able to use a custom activation function', done => {
    const net = nn([2, 4, 1], {
      activation: x => 2 / (1 + Math.exp(-2 * x)) - 1
    });
    net.train(
      [
        { input: [1, 0], output: [1] },
        { input: [0, 1], output: [1] },
        { input: [0, 0], output: [0] },
        { input: [1, 1], output: [0] }
      ],
      {
        done: () => {
          net.run([1, 1])[0].should.be.approximately(0, delta);
          net.run([1, 0])[0].should.be.approximately(1, delta);
          net.run([0, 1])[0].should.be.approximately(1, delta);
          net.run([0, 0])[0].should.be.approximately(0, delta);
          done();
        }
      }
    );
  });
});
