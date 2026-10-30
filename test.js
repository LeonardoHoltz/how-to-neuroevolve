// Self-check: node test.js
// Loads the browser scripts into one shared context (as <script> tags would) and checks the math.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const ctx = vm.createContext({ console, assert });
for (const f of ['js/core.js', 'js/algorithms.js']) vm.runInContext(fs.readFileSync(`${__dirname}/${f}`, 'utf8'), ctx, { filename: f });
vm.runInContext(`
  // backward() matches finite differences
  const data = makeData('xor'), t = initTheta(7), g = lossGrad(t, data);
  for (let k = 0; k < D; k++) {
    const tp = t.slice(), tm = t.slice(); tp[k] += 1e-5; tm[k] -= 1e-5;
    assert(Math.abs((loss(tp, data) - loss(tm, data)) / 2e-5 - g[k]) < 1e-6, 'grad ' + k);
  }
  // jacobiEig reconstructs a symmetric matrix
  const r = makeRng(1), M = Array.from({ length: 6 }, () => randn(r, 6)), S = M.map((_, i) => M.map((_, j) => M[i][j] + M[j][i]));
  const { values, V } = jacobiEig(S);
  S.forEach((row, i) => row.forEach((x, j) => assert(Math.abs(values.reduce((a, l, k) => a + V[i][k] * l * V[j][k], 0) - x) < 1e-8, 'eig')));
  // fitness shaping invariants
  const f = [3, 1, 2, 5];
  centeredRanks(f).forEach((v, i) => assert(Math.abs(v - [1 / 6, -0.5, -1 / 6, 0.5][i]) < 1e-12, "ranks"));
  assert(Math.abs(nesUtilities(f).reduce((a, b) => a + b)) < 1e-12 && nesUtilities(f)[1] > 0);
  assert(Math.abs(recombinationWeights(f).reduce((a, b) => a + b) - 1) < 1e-12);
  // Dust's node-perturbation estimate points along the true gradient
  {
    const A = ALGORITHMS.dust, hp = { popsize: 256, sigma: 0.2, lr: 0 }, rng = makeRng(5), s = A.init(t, hp);
    const asked = A.ask(s, hp, rng, data), fit = A.evaluate(asked, data, hp);
    A.tell(s, asked, fit, A.shape(fit, asked).w, hp, data);
    const cos = g.reduce((a, x, k) => a + x * s.grad[k], 0) / (norm(g) * norm(s.grad));
    assert(cos > 0.9, 'dust cosine ' + cos);
  }
  // every algorithm learns every task with its default hyperparameters
  for (const kind of ['linear', 'xor', 'circle']) {
    const d = makeData(kind);
    for (const [id, A] of Object.entries(ALGORITHMS)) {
      const hp = { ...A.defaults }, rng = makeRng(42), s = A.init(initTheta(7), hp, rng);
      const gens = 300;
      for (let i = 0; i < gens; i++) {
        const asked = A.ask(s, hp, rng, d), fit = A.evaluate ? A.evaluate(asked, d, hp) : asked.pop.map(x => loss(x, d)), sh = A.shape(fit, asked, s, hp, d);
        A.tell(s, asked, fit, sh.w, hp, d);
      }
      const acc = accuracy(A.center(s, hp), d), L = loss(A.center(s, hp), d);
      console.log(kind.padEnd(7), id.padEnd(10), 'loss', L.toFixed(3), 'acc', acc.toFixed(2));
      assert(acc >= 0.9, id + ' failed on ' + kind);
    }
  }
  console.log('ok');
`, ctx);
