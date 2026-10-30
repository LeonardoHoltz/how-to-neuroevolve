// Shared building blocks, mirroring evosax.core: RNG, the tiny network, the task,
// fitness shaping functions, optimizers and a little linear algebra.
// Classic script (no modules) so index.html opens straight from disk.

// ---------- Network: 2 inputs -> H tanh hidden -> 1 sigmoid output ----------
const H = 4;
const D = 4 * H + 1; // W1 (H×2) + b1 (H) + W2 (1×H) + b2 (1)
const LAYOUT = [
  { name: 'W1', start: 0, len: 2 * H, rows: H, cols: 2 },
  { name: 'b1', start: 2 * H, len: H },
  { name: 'W2', start: 3 * H, len: H, rows: 1, cols: H },
  { name: 'b2', start: 4 * H, len: 1 },
];
const w1 = (t, j, i) => t[2 * j + i];
const b1 = (t, j) => t[2 * H + j];
const w2 = (t, j) => t[3 * H + j];
const b2 = (t) => t[4 * H];

function forward(t, x1, x2) {
  const z1 = new Float64Array(H), h = new Float64Array(H);
  let z2 = b2(t);
  for (let j = 0; j < H; j++) {
    z1[j] = w1(t, j, 0) * x1 + w1(t, j, 1) * x2 + b1(t, j);
    h[j] = Math.tanh(z1[j]);
    z2 += w2(t, j) * h[j];
  }
  return { z1, h, z2, p: 1 / (1 + Math.exp(-z2)) };
}

// Forward pass with Gaussian jitter σ·a added to each neuron's pre-activation (Dust's node perturbation).
function noisyForward(t, x1, x2, n, sigma) {
  const h = new Float64Array(H);
  let z2 = b2(t) + sigma * n.a2;
  for (let j = 0; j < H; j++) {
    h[j] = Math.tanh(w1(t, j, 0) * x1 + w1(t, j, 1) * x2 + b1(t, j) + sigma * n.a1[j]);
    z2 += w2(t, j) * h[j];
  }
  return { h, z2, p: 1 / (1 + Math.exp(-z2)) };
}

// Backward pass for one example: returns dL/dθ and the per-neuron deltas dL/dz.
function backward(t, x1, x2, y) {
  const f = forward(t, x1, x2), g = new Float64Array(D), dz1 = new Float64Array(H);
  const dz2 = f.p - y; // sigmoid + binary cross-entropy
  g[4 * H] = dz2;
  for (let j = 0; j < H; j++) {
    g[3 * H + j] = dz2 * f.h[j];
    dz1[j] = dz2 * w2(t, j) * (1 - f.h[j] * f.h[j]);
    g[2 * j] = dz1[j] * x1;
    g[2 * j + 1] = dz1[j] * x2;
    g[2 * H + j] = dz1[j];
  }
  return { f, g, dz1, dz2 };
}

const bce = (p, y) => { p = Math.min(1 - 1e-7, Math.max(1e-7, p)); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); };
const loss = (t, data) => data.reduce((s, d) => s + bce(forward(t, d.x1, d.x2).p, d.y), 0) / data.length;
const accuracy = (t, data) => data.filter(d => (forward(t, d.x1, d.x2).p > 0.5) === (d.y === 1)).length / data.length;

function lossGrad(t, data) {
  const g = new Float64Array(D);
  for (const d of data) backward(t, d.x1, d.x2, d.y).g.forEach((v, k) => (g[k] += v / data.length));
  return g;
}

// ---------- RNG ----------
function makeRng(seed) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => { let u = 0; while (u === 0) u = uniform(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform()); };
  return { uniform, normal, int: n => Math.floor(uniform() * n) };
}
const randn = (rng, n) => Float64Array.from({ length: n }, rng.normal);

// ---------- Task: 2-D binary classification ----------
function makeData(kind, seed = 3, n = 80) {
  const rng = makeRng(seed), pts = [];
  for (let i = 0; i < n; i++) {
    const s1 = rng.uniform() < 0.5 ? -1 : 1, s2 = rng.uniform() < 0.5 ? -1 : 1;
    let x1, x2, y;
    if (kind === 'xor') {
      x1 = s1 * (0.15 + 0.75 * rng.uniform()); x2 = s2 * (0.15 + 0.75 * rng.uniform()); y = s1 * s2 > 0 ? 1 : 0;
    } else if (kind === 'circle') {
      y = i % 2;
      const r = y ? 0.45 * rng.uniform() : 0.6 + 0.35 * rng.uniform(), a = 2 * Math.PI * rng.uniform();
      x1 = r * Math.cos(a); x2 = r * Math.sin(a);
    } else { // linear
      x1 = 1.8 * rng.uniform() - 0.9; x2 = 1.8 * rng.uniform() - 0.9;
      if (Math.abs(x1 + 0.5 * x2) < 0.15) { i--; continue; }
      y = x1 + 0.5 * x2 > 0 ? 1 : 0;
    }
    pts.push({ x1, x2, y });
  }
  return pts;
}

// ---------- Fitness shaping (fitness = loss, lower is better) ----------
// rank 0 = best
function ranksOf(f) {
  const r = new Array(f.length);
  [...f.keys()].sort((a, b) => f[a] - f[b]).forEach((i, k) => (r[i] = k));
  return r;
}
// Best member → -0.5, worst → +0.5. Used as a "loss-like" signal by Open_ES, PGPE, EGGROLL.
const centeredRanks = f => ranksOf(f).map(k => k / (f.length - 1) - 0.5);
// NES utilities: best gets the largest positive weight, they sum to 0. Used by SNES/xNES.
function nesUtilities(f) {
  const n = f.length, raw = ranksOf(f).map(k => Math.max(0, Math.log(n / 2 + 1) - Math.log(k + 1)));
  const s = raw.reduce((a, b) => a + b, 0);
  return raw.map(u => u / s - 1 / n);
}
// Log-decreasing weights on the best half, zero elsewhere, summing to 1. Used by SimpleES and CMA-ES.
function recombinationWeights(f) {
  const mu = Math.floor(f.length / 2), raw = ranksOf(f).map(k => (k < mu ? Math.log(mu + 0.5) - Math.log(k + 1) : 0));
  const s = raw.reduce((a, b) => a + b, 0);
  return raw.map(w => w / s);
}

// ---------- Optimizers (evosax delegates this to optax) ----------
function adam(lr, beta1 = 0.9, beta2 = 0.999) {
  const m = new Float64Array(D), v = new Float64Array(D);
  let t = 0;
  return {
    step(theta, g) {
      t++;
      return theta.map((x, k) => {
        m[k] = beta1 * m[k] + (1 - beta1) * g[k];
        v[k] = beta2 * v[k] + (1 - beta2) * g[k] * g[k];
        return x - (lr * m[k]) / (1 - beta1 ** t) / (Math.sqrt(v[k] / (1 - beta2 ** t)) + 1e-8);
      });
    },
  };
}

// ---------- Small linear algebra ----------
const norm = v => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

// Symmetric eigendecomposition (cyclic Jacobi). V[k][i] is component k of eigenvector i.
function jacobiEig(A) {
  const n = A.length, a = A.map(r => Float64Array.from(r));
  const V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, k) => +(i === k)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (a[p][q] === 0) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const kp = a[k][p], kq = a[k][q]; a[k][p] = c * kp - s * kq; a[k][q] = s * kp + c * kq; }
      for (let k = 0; k < n; k++) { const pk = a[p][k], qk = a[q][k]; a[p][k] = c * pk - s * qk; a[q][k] = s * pk + c * qk; }
      for (let k = 0; k < n; k++) { const kp = V[k][p], kq = V[k][q]; V[k][p] = c * kp - s * kq; V[k][q] = s * kp + c * kq; }
    }
  }
  return { values: a.map((r, i) => r[i]), V };
}

// Same starting weights for every algorithm, so runs are comparable.
const initTheta = seed => randn(makeRng(seed), D).map(x => 0.5 * x);
