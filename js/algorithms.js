// Every optimizer is written as the same ask → evaluate → shape → tell loop that evosax uses.
// Each part points at a reusable "component" (CHIPS), so the UI can box together what algorithms share.

const CHIPS = {
  // state: what survives from one generation to the next
  point: 'One weight vector θ',
  iso_gauss: 'Gaussian N(μ, σ²I)',
  diag_gauss: 'Gaussian N(μ, diag σ²)',
  full_gauss: 'Gaussian N(m, σ²C)',
  population: 'Explicit population',
  swarm: 'Swarm: positions, velocities, memories',
  // ask: how candidates are produced
  none: 'No sampling',
  gauss: 'Gaussian sampling',
  gauss_corr: 'Correlated Gaussian sampling',
  mirrored: 'Mirrored (antithetic) pairs',
  lowrank: 'Low-rank matrix perturbations',
  crossover: 'Crossover',
  mutation: 'Gaussian mutation',
  diff_mut: 'Difference-vector mutation',
  velocity: 'Velocity update',
  node_noise: 'Noise on neuron outputs',
  // evaluate
  forward: 'Forward pass + loss',
  // shape: how raw fitness becomes weights
  backward: 'Backward pass (exact gradient)',
  centered_rank: 'Centered ranks',
  nes_util: 'NES utilities',
  recomb_w: 'Log recombination weights',
  truncation: 'Truncation selection',
  greedy: 'Greedy comparison',
  loss_change: 'Per-example loss change',
  // tell: how the state is updated
  sgd: 'Gradient step',
  mc_grad: 'Monte-Carlo gradient estimate',
  optimizer: 'Optimizer step (Adam)',
  std_grad: 'σ gradient',
  nat_grad: 'Natural-gradient μ, σ update',
  recomb: 'Weighted recombination',
  std_elite: 'σ from elite spread',
  cma_adapt: 'Evolution paths + C, σ adaptation',
  elitism: 'Elitism (keep best)',
  replace: 'Replace if better',
  memory: 'Update personal / global best',
  node_grad: 'Noise × reward → error, × input → ∇W',
  momentum: 'SGD with momentum',
};

// shared helpers
const sampleAround = (mean, std, z) => mean.map((m, k) => m + (typeof std === 'number' ? std : std[k]) * z[k]);
const argmin = f => f.reduce((b, v, i) => (v < f[b] ? i : b), 0);
const fmt = (x, d = 3) => (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(1) : x.toFixed(d));
// popsize is always even (the UI steps by 2)
function mirroredNoise(rng, n, make) {
  const half = Array.from({ length: n / 2 }, make);
  return [...half, ...half.map(e => e.map(x => -x))];
}
// Open_ES-style gradient estimate: ĝ = 1/(Nσ) Σ shaped_i · ε_i
function mcGradient(eps, w, sigma) {
  const g = new Float64Array(D);
  eps.forEach((e, i) => e.forEach((x, k) => (g[k] += (w[i] * x) / (eps.length * sigma))));
  return g;
}

const ALGORITHMS = {
  backprop: {
    name: 'Backprop + GD', family: 'gradient', evosax: null,
    defaults: { popsize: 1, sigma: 0, lr: 1 },
    parts: { state: ['point'], ask: ['none'], shape: ['backward'], tell: ['sgd'] },
    math: {
      ask: 'θ (no candidates: the model itself is evaluated)',
      shape: 'δ₂ = p − y,   δ₁ = δ₂ · W₂ ⊙ (1 − h²),   ∂L/∂W = δ · aᵀ',
      tell: 'θ ← θ − η · ∇L(θ)',
    },
    init: (theta0) => ({ theta: theta0.slice() }),
    ask: (s) => ({ pop: [s.theta.slice()], eps: [new Float64Array(D)], note: 'Nothing is sampled: backprop evaluates the current weights directly.' }),
    shape(f, asked, s, hp, data) {
      s.grad = lossGrad(s.theta, data);
      return { w: [1], note: `Backward pass over ${data.length} points: chain rule gives the exact gradient, |∇L| = ${fmt(norm(s.grad))}.` };
    },
    tell(s, asked, f, w, hp) {
      s.theta = s.theta.map((x, k) => x - hp.lr * s.grad[k]);
      return `θ moved by η·|∇L| = ${fmt(hp.lr * norm(s.grad))} straight downhill.`;
    },
    center: s => s.theta,
  },

  simple_es: {
    name: 'SimpleES', family: 'recomb', evosax: 'SimpleES',
    defaults: { popsize: 32, sigma: 0.5, lr: 1 },
    parts: { state: ['diag_gauss'], ask: ['gauss'], shape: ['recomb_w'], tell: ['recomb', 'std_elite'] },
    math: {
      ask: 'x_i = μ + σ ⊙ ε_i,   ε_i ~ N(0, I)',
      shape: 'w_i ∝ log(μ+½) − log(rank_i)  for the best half, else 0',
      tell: 'μ ← Σ w_i x_i,   σ ← (1−c)σ + c·√(Σ w_i (x_i − μ_old)²)',
    },
    init: (theta0, hp) => ({ mean: theta0.slice(), std: new Float64Array(D).fill(hp.sigma) }),
    ask(s, hp, rng) {
      const eps = Array.from({ length: hp.popsize }, () => randn(rng, D));
      return { eps, pop: eps.map(z => sampleAround(s.mean, s.std, z)), note: `Sampled ${hp.popsize} members independently around μ, each dimension with its own σ.` };
    },
    shape: f => ({ w: recombinationWeights(f), note: `The best ${Math.floor(f.length / 2)} get log-decreasing weights; the rest get 0.` }),
    tell(s, asked, f, w) {
      const old = s.mean;
      s.mean = old.map((_, k) => asked.pop.reduce((acc, x, i) => acc + w[i] * x[k], 0));
      s.std = s.std.map((sd, k) => Math.max(1e-3, 0.9 * sd + 0.1 * Math.sqrt(asked.pop.reduce((acc, x, i) => acc + w[i] * (x[k] - old[k]) ** 2, 0))));
      return `μ jumped to the weighted average of the elite (moved ${fmt(norm(s.mean.map((m, k) => m - old[k])))}). σ shrinks or grows to match the elite's spread.`;
    },
    center: s => s.mean, spread: s => s.std,
  },

  open_es: {
    name: 'OpenES', family: 'grad_est', evosax: 'Open_ES',
    defaults: { popsize: 32, sigma: 0.1, lr: 0.05 },
    parts: { state: ['iso_gauss'], ask: ['gauss', 'mirrored'], shape: ['centered_rank'], tell: ['mc_grad', 'optimizer'] },
    math: {
      ask: 'x_i = μ ± σ ε_i,   ε_i ~ N(0, I)',
      shape: 'r_i = rank_i / (N−1) − ½   (best → −½, worst → +½)',
      tell: 'ĝ = 1/(Nσ) Σ r_i ε_i,   μ ← Adam(μ, ĝ)',
    },
    init: (theta0, hp) => ({ mean: theta0.slice(), opt: adam(hp.lr) }),
    ask(s, hp, rng) {
      const eps = mirroredNoise(rng, hp.popsize, () => randn(rng, D));
      return { eps, pop: eps.map(e => sampleAround(s.mean, hp.sigma, e)), note: `Sampled ${eps.length / 2} noise vectors and used each twice, as +ε and −ε. Mirroring cancels noise in the gradient estimate.` };
    },
    shape: f => ({ w: centeredRanks(f), note: 'Raw losses are replaced by their rank. One lucky outlier can no longer dominate the update.' }),
    tell(s, asked, f, w, hp) {
      s.grad = mcGradient(asked.eps, w, hp.sigma);
      s.mean = s.opt.step(s.mean, s.grad);
      return `Each ε_i was weighted by its rank. Their sum estimates the gradient of the smoothed loss: |ĝ| = ${fmt(norm(s.grad))}. Adam took one step against ĝ.`;
    },
    center: s => s.mean, spread: (s, hp) => new Float64Array(D).fill(hp.sigma),
  },

  pgpe: {
    name: 'PGPE', family: 'grad_est', evosax: 'PGPE',
    defaults: { popsize: 32, sigma: 0.3, lr: 0.05 },
    parts: { state: ['diag_gauss'], ask: ['gauss', 'mirrored'], shape: ['centered_rank'], tell: ['mc_grad', 'std_grad', 'optimizer'] },
    math: {
      ask: 'x± = μ ± σ ⊙ z,   z ~ N(0, I)',
      shape: 'r_i = rank_i / (N−1) − ½',
      tell: 'g_μ = mean[(r₊ − r₋)/2 · z],   g_σ = mean[(r₊ + r₋)/2 · (z² − 1)],   σ ← σ(1 − η_σ g_σ)',
    },
    init: (theta0, hp) => ({ mean: theta0.slice(), std: new Float64Array(D).fill(hp.sigma), opt: adam(hp.lr) }),
    ask(s, hp, rng) {
      const eps = mirroredNoise(rng, hp.popsize, () => randn(rng, D));
      return { eps, pop: eps.map(z => sampleAround(s.mean, s.std, z)), note: `${eps.length / 2} symmetric pairs μ ± σ⊙z. Each pair is a finite-difference probe along z.` };
    },
    shape: f => ({ w: centeredRanks(f), note: 'Losses become centered ranks, as in OpenES.' }),
    tell(s, asked, f, w) {
      const K = asked.eps.length / 2, gm = new Float64Array(D), gs = new Float64Array(D);
      for (let i = 0; i < K; i++) {
        const z = asked.eps[i], diff = (w[i] - w[i + K]) / 2, avg = (w[i] + w[i + K]) / 2;
        z.forEach((x, k) => { gm[k] += (diff * x) / K; gs[k] += (avg * (x * x - 1)) / K; });
      }
      s.grad = gm;
      s.mean = s.opt.step(s.mean, gm);
      s.std = s.std.map((sd, k) => Math.min(2, Math.max(1e-3, sd * (1 - 0.2 * gs[k]))));
      return `Pair differences gave the μ gradient (|g_μ| = ${fmt(norm(gm))}). Pair averages told each σ_k whether wider probes helped: mean σ is now ${fmt(s.std.reduce((a, b) => a + b, 0) / D)}.`;
    },
    center: s => s.mean, spread: s => s.std,
  },

  eggroll: {
    name: 'EGGROLL', family: 'grad_est', evosax: null,
    defaults: { popsize: 32, sigma: 0.15, lr: 0.05 },
    parts: { state: ['iso_gauss'], ask: ['lowrank', 'mirrored'], shape: ['centered_rank'], tell: ['mc_grad', 'optimizer'] },
    math: {
      ask: 'E_i = A_i B_iᵀ / √r   (A ∈ ℝ^{m×r}, B ∈ ℝ^{n×r}),   W_i = W ± σ E_i',
      shape: 'r_i = rank_i / (N−1) − ½',
      tell: 'ĝ = 1/(Nσ) Σ r_i E_i   (rank ≤ min(N·r, m, n)),   μ ← Adam(μ, ĝ)',
    },
    init: (theta0, hp) => ({ mean: theta0.slice(), opt: adam(hp.lr) }),
    ask(s, hp, rng) {
      const half = Array.from({ length: hp.popsize / 2 }, () => {
        const e = randn(rng, D), fac = {}; // bias slots keep full Gaussian noise
        for (const L of LAYOUT.filter(L => L.rows)) {
          const A = randn(rng, L.rows), B = randn(rng, L.cols); // rank r = 1
          for (let r = 0; r < L.rows; r++) for (let c = 0; c < L.cols; c++) e[L.start + r * L.cols + c] = A[r] * B[c];
          fac[L.name] = { A, B };
        }
        return { e, fac };
      });
      const neg = fac => Object.fromEntries(Object.entries(fac).map(([k, v]) => [k, { A: v.A.map(x => -x), B: v.B }]));
      const eps = [...half.map(h => h.e), ...half.map(h => h.e.map(x => -x))];
      const factors = [...half.map(h => h.fac), ...half.map(h => neg(h.fac))];
      return { eps, factors, pop: eps.map(e => sampleAround(s.mean, hp.sigma, e)), note: 'Each weight-matrix perturbation is an outer product a·bᵀ (rank 1). For an m×n layer that needs m+n random numbers instead of m·n.' };
    },
    shape: f => ({ w: centeredRanks(f), note: 'Losses become centered ranks, the same shaping as OpenES.' }),
    tell(s, asked, f, w, hp) {
      s.grad = mcGradient(asked.eps, w, hp.sigma);
      s.mean = s.opt.step(s.mean, s.grad);
      return `Every member was a rank-1 nudge, but their weighted sum is not rank-1. The W1 update has rank ${matrixRank(s.grad, LAYOUT[0])}, the most a ${H}×2 matrix can have. |ĝ| = ${fmt(norm(s.grad))}.`;
    },
    center: s => s.mean, spread: (s, hp) => new Float64Array(D).fill(hp.sigma),
  },

  dust: {
    name: 'Dust', family: 'node', evosax: null,
    defaults: { popsize: 16, sigma: 0.2, lr: 0.1 },
    parts: { state: ['point'], ask: ['node_noise'], shape: ['loss_change'], tell: ['node_grad', 'momentum'] },
    math: {
      ask: 'z_t → z_t + σ a_t,   a_t ~ N(0, I)   fresh for every neuron, data point t and draw k',
      shape: 'c_t = mean_k ℓ_t − ℓ_t⁽ᵏ⁾   (our net has no sequence, so the credit decay γ = 0 and r_t = c_t)',
      tell: 'ĝ_t = −1/(Kσ) Σ_k r_t a_t,   G_W = Σ_t ĝ_t x_tᵀ,   θ ← SGD-momentum(θ, G)',
    },
    init: (theta0) => ({ theta: theta0.slice(), vel: new Float64Array(D) }),
    ask(s, hp, rng, data) {
      const noise = Array.from({ length: hp.popsize }, () => data.map(() => ({ a1: randn(rng, H), a2: rng.normal() })));
      return { pop: noise.map(() => s.theta), eps: noise.map(() => new Float64Array(D)), noise, note: `${hp.popsize} draws × ${data.length} points = ${hp.popsize * data.length} virtual members from only ${hp.popsize} forward passes over the data. The weights stay untouched: each draw jitters every neuron's output, separately for each point.` };
    },
    // Dust scores each (draw, point) pair, so it overrides the default "loss of each member".
    evaluate(asked, data, hp) {
      asked.losses = asked.noise.map(draw => draw.map((n, t) => bce(noisyForward(asked.pop[0], data[t].x1, data[t].x2, n, hp.sigma).p, data[t].y)));
      return asked.losses.map(l => l.reduce((a, b) => a + b, 0) / l.length);
    },
    shape(f, asked) {
      const K = asked.losses.length, T = asked.losses[0].length;
      const mean = Array.from({ length: T }, (_, t) => asked.losses.reduce((a, l) => a + l[t], 0) / K);
      asked.reward = asked.losses.map(l => l.map((v, t) => mean[t] - v));
      return { w: asked.reward.map(r => r.reduce((a, b) => a + b, 0) / T), note: 'Each jitter is rewarded per point: how much lower that point\'s loss was than the average draw on the same point. The w shown below is each draw\'s reward averaged over points.' };
    },
    tell(s, asked, f, w, hp, data) {
      const K = asked.noise.length, T = data.length, g = new Float64Array(D);
      data.forEach((d, t) => {
        const h = forward(s.theta, d.x1, d.x2).h, g1 = new Float64Array(H);
        let g2 = 0;
        asked.noise.forEach((draw, k) => {
          const r = asked.reward[k][t];
          draw[t].a1.forEach((a, j) => (g1[j] -= (r * a) / (K * hp.sigma)));
          g2 -= (r * draw[t].a2) / (K * hp.sigma);
        });
        // error signal × layer input, the same local rule backprop ends with
        for (let j = 0; j < H; j++) {
          g[2 * j] += (g1[j] * d.x1) / T; g[2 * j + 1] += (g1[j] * d.x2) / T; g[2 * H + j] += g1[j] / T;
          g[3 * H + j] += (g2 * h[j]) / T;
        }
        g[4 * H] += g2 / T;
      });
      s.grad = g;
      s.vel = s.vel.map((v, k) => 0.9 * v + g[k]);
      s.theta = s.theta.map((x, k) => x - hp.lr * s.vel[k]);
      return `For every neuron, the jitters that helped were averaged into an estimate of its error signal. Multiplying by the neuron's inputs gave a gradient for every weight (|G| = ${fmt(norm(g))}), and SGD with momentum took the step.`;
    },
    center: s => s.theta,
  },

  snes: {
    name: 'SNES', family: 'nes', evosax: 'SNES',
    defaults: { popsize: 16, sigma: 0.5, lr: 1 },
    parts: { state: ['diag_gauss'], ask: ['gauss'], shape: ['nes_util'], tell: ['nat_grad'] },
    math: {
      ask: 'x_i = μ + σ ⊙ z_i,   z_i ~ N(0, I)',
      shape: 'u_i = max(0, log(N/2+1) − log(rank_i)) / Σ − 1/N',
      tell: 'μ ← μ + η_μ σ ⊙ Σ u_i z_i,   σ ← σ ⊙ exp(η_σ/2 · Σ u_i (z_i² − 1))',
    },
    init: (theta0, hp) => ({ mean: theta0.slice(), std: new Float64Array(D).fill(hp.sigma) }),
    ask(s, hp, rng) {
      const eps = Array.from({ length: hp.popsize }, () => randn(rng, D));
      return { eps, pop: eps.map(z => sampleAround(s.mean, s.std, z)), note: `Sampled ${hp.popsize} members from the separable Gaussian.` };
    },
    shape: f => ({ w: nesUtilities(f), note: 'NES utilities: the best members get positive weight, the rest share a small negative weight, and everything sums to 0.' }),
    tell(s, asked, f, w, hp) {
      const etaS = (3 + Math.log(D)) / (5 * Math.sqrt(D)), gm = new Float64Array(D), gs = new Float64Array(D);
      asked.eps.forEach((z, i) => z.forEach((x, k) => { gm[k] += w[i] * x; gs[k] += w[i] * (x * x - 1); }));
      s.grad = gm.map(x => -x);
      s.mean = s.mean.map((m, k) => m + hp.lr * s.std[k] * gm[k]);
      s.std = s.std.map((sd, k) => sd * Math.exp((etaS / 2) * gs[k]));
      return `This is a natural-gradient step. μ moves in units of σ, and log σ moves by η_σ/2 · Σ u(z² − 1) (η_σ = ${fmt(etaS)}). Mean σ is now ${fmt(s.std.reduce((a, b) => a + b, 0) / D)}.`;
    },
    center: s => s.mean, spread: s => s.std,
  },

  cma_es: {
    name: 'CMA-ES', family: 'recomb', evosax: 'CMA_ES',
    defaults: { popsize: 16, sigma: 0.5, lr: 1 },
    parts: { state: ['full_gauss'], ask: ['gauss_corr'], shape: ['recomb_w'], tell: ['recomb', 'cma_adapt'] },
    math: {
      ask: 'y_i = B·D·z_i,   x_i = m + σ y_i   (so y_i ~ N(0, C))',
      shape: 'w_i ∝ log(μ+½) − log(rank_i)  for the best half',
      tell: 'm ← m + σ Σ w_i y_i,   p_σ, p_c ← paths,   C ← (1−c₁−c_μ)C + c₁ p_c p_cᵀ + c_μ Σ w_i y_i y_iᵀ,   σ ← σ·exp(c_σ/d_σ (|p_σ|/E|N| − 1))',
    },
    init(theta0, hp) {
      const lam = hp.popsize, mu = Math.floor(lam / 2);
      const raw = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1)), sw = raw.reduce((a, b) => a + b, 0);
      const mueff = 1 / raw.reduce((a, w) => a + (w / sw) ** 2, 0), n = D;
      const cs = (mueff + 2) / (n + mueff + 5), c1 = 2 / ((n + 1.3) ** 2 + mueff);
      return {
        mean: theta0.slice(), sigma: hp.sigma, C: Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, k) => +(i === k))),
        B: null, Dv: new Float64Array(n).fill(1), pc: new Float64Array(n), ps: new Float64Array(n), g: 0,
        mueff, cs, c1, cc: (4 + mueff / n) / (n + 4 + (2 * mueff) / n),
        cmu: Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) ** 2 + mueff)),
        damps: 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs,
        chiN: Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n)),
      };
    },
    ask(s, hp, rng) {
      const eps = Array.from({ length: hp.popsize }, () => {
        const z = randn(rng, D);
        return s.B ? Float64Array.from({ length: D }, (_, k) => s.B[k].reduce((a, b, i) => a + b * s.Dv[i] * z[i], 0)) : z;
      });
      return { eps, pop: eps.map(y => sampleAround(s.mean, s.sigma, y)), note: `Sampled ${hp.popsize} members from N(m, σ²C). C has learned correlations between weights, so the cloud can be a tilted ellipsoid, not just a ball.` };
    },
    shape: f => ({ w: recombinationWeights(f), note: `The best ${Math.floor(f.length / 2)} get log-decreasing recombination weights.` }),
    tell(s, asked, f, w) {
      const n = D, ys = asked.eps, yw = new Float64Array(n);
      ys.forEach((y, i) => w[i] && y.forEach((x, k) => (yw[k] += w[i] * x)));
      s.mean = s.mean.map((m, k) => m + s.sigma * yw[k]);
      // C^{-1/2} yw = B D^{-1} Bᵀ yw
      const Bt = s.B ? Array.from({ length: n }, (_, i) => s.B.reduce((a, row, k) => a + row[i] * yw[k], 0) / s.Dv[i]) : yw;
      const invY = s.B ? Float64Array.from({ length: n }, (_, k) => s.B[k].reduce((a, b, i) => a + b * Bt[i], 0)) : yw;
      s.ps = s.ps.map((p, k) => (1 - s.cs) * p + Math.sqrt(s.cs * (2 - s.cs) * s.mueff) * invY[k]);
      s.g++;
      const hsig = norm(s.ps) / Math.sqrt(1 - (1 - s.cs) ** (2 * s.g)) / s.chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
      s.pc = s.pc.map((p, k) => (1 - s.cc) * p + hsig * Math.sqrt(s.cc * (2 - s.cc) * s.mueff) * yw[k]);
      const a = 1 - s.c1 - s.cmu + (1 - hsig) * s.c1 * s.cc * (2 - s.cc);
      for (let i = 0; i < n; i++) for (let k = 0; k <= i; k++) {
        let v = a * s.C[i][k] + s.c1 * s.pc[i] * s.pc[k];
        ys.forEach((y, j) => w[j] && (v += s.cmu * w[j] * y[i] * y[k]));
        s.C[i][k] = s.C[k][i] = v;
      }
      s.sigma *= Math.exp((s.cs / s.damps) * (norm(s.ps) / s.chiN - 1));
      const { values, V } = jacobiEig(s.C);
      s.B = V; s.Dv = Float64Array.from(values, v => Math.sqrt(Math.max(v, 1e-20)));
      return `m moved to the weighted elite mean. The evolution paths update C (the cloud's shape) and σ (its size): σ = ${fmt(s.sigma)}, condition number of C = ${fmt(Math.max(...values) / Math.max(1e-20, Math.min(...values)), 1)}.`;
    },
    center: s => s.mean,
    spread: s => Float64Array.from({ length: D }, (_, k) => s.sigma * Math.sqrt(s.C[k][k])),
    cov: s => s.C.map(r => r.map(x => x * s.sigma * s.sigma)),
  },

  simple_ga: {
    name: 'SimpleGA', family: 'population', evosax: 'SimpleGA',
    defaults: { popsize: 32, sigma: 0.15, lr: 1 },
    parts: { state: ['population'], ask: ['crossover', 'mutation'], shape: ['truncation'], tell: ['elitism'] },
    math: {
      ask: 'child = crossover(parent_a, parent_b) + σ ε',
      shape: 'keep the top 25 % (selected = 1, others = 0)',
      tell: 'elite ← best members;  next generation is bred only from them',
    },
    init: (theta0, hp, rng) => ({ elite: null, init: Array.from({ length: hp.popsize }, () => sampleAround(theta0, 0.5, randn(rng, D))), best: theta0.slice() }),
    ask(s, hp, rng) {
      if (!s.elite) return { pop: s.init, eps: s.init.map(x => x.map((v, k) => v - s.best[k])), note: 'Generation 0 is a random population scattered around the initial weights.' };
      const pop = [], eps = [];
      for (let i = 0; i < hp.popsize; i++) {
        if (i < s.elite.length) { pop.push(s.elite[i].slice()); eps.push(new Float64Array(D)); continue; }
        const a = s.elite[rng.int(s.elite.length)], b = s.elite[rng.int(s.elite.length)];
        const child = a.map((v, k) => (rng.uniform() < 0.5 ? v : b[k]) + hp.sigma * rng.normal());
        pop.push(child); eps.push(child.map((v, k) => v - a[k]));
      }
      return { pop, eps, note: `${s.elite.length} elites are copied unchanged. The other ${hp.popsize - s.elite.length} are children: uniform crossover of two elite parents, then Gaussian mutation.` };
    },
    shape(f) {
      const k = Math.max(2, Math.floor(f.length / 4));
      return { w: ranksOf(f).map(r => +(r < k)), note: `Truncation selection: the best ${k} survive and everyone else is discarded. No gradient, no averaging.` };
    },
    tell(s, asked, f, w) {
      const order = [...f.keys()].sort((a, b) => f[a] - f[b]);
      s.elite = order.filter(i => w[i]).map(i => asked.pop[i]);
      s.best = s.elite[0];
      return `Kept ${s.elite.length} elites. The displayed network is now the best individual (loss ${fmt(f[order[0]])}).`;
    },
    center: s => s.best,
  },

  de: {
    name: 'Differential Evolution', family: 'population', evosax: 'DifferentialEvolution',
    defaults: { popsize: 32, sigma: 0.5, lr: 0.5 },
    parts: { state: ['population'], ask: ['diff_mut', 'crossover'], shape: ['greedy'], tell: ['replace'] },
    math: {
      ask: 'v = x_a + F (x_b − x_c),   trial = crossover(x_i, v, CR = 0.9)   (F = learning rate)',
      shape: 'selected_i = [ L(trial_i) ≤ L(x_i) ]',
      tell: 'x_i ← trial_i if it was at least as good',
    },
    init: (theta0, hp, rng) => ({ pop: Array.from({ length: hp.popsize }, () => sampleAround(theta0, hp.sigma, randn(rng, D))), fit: null }),
    ask(s, hp, rng) {
      if (!s.fit) return { pop: s.pop, eps: s.pop.map(() => new Float64Array(D)), note: 'Generation 0: evaluate the random starting population.' };
      const N = s.pop.length, pick = ex => { let r; do r = rng.int(N); while (ex.includes(r)); return r; };
      const pop = s.pop.map((x, i) => {
        const a = pick([i]), b = pick([i, a]), c = pick([i, a, b]), jr = rng.int(D);
        return x.map((v, k) => (k === jr || rng.uniform() < 0.9 ? s.pop[a][k] + hp.lr * (s.pop[b][k] - s.pop[c][k]) : v));
      });
      return { pop, eps: pop.map((t, i) => t.map((v, k) => v - s.pop[i][k])), note: 'Each trial adds a scaled difference between two other members to a third. The population\'s own spread sets the step size.' };
    },
    shape(f, asked, s) {
      const w = s.fit ? f.map((v, i) => +(v <= s.fit[i])) : f.map(() => 1);
      return { w, note: s.fit ? `${w.filter(Boolean).length} of ${f.length} trials beat the parent they compete with.` : 'First evaluation: every member is kept.' };
    },
    tell(s, asked, f, w) {
      if (!s.fit) s.fit = f.slice();
      w.forEach((keep, i) => keep && ((s.pop[i] = asked.pop[i]), (s.fit[i] = f[i])));
      return `Replaced ${w.filter(Boolean).length} parents. Best loss in the population: ${fmt(Math.min(...s.fit))}.`;
    },
    center: s => (s.fit ? s.pop[argmin(s.fit)] : s.pop[0]),
  },

  pso: {
    name: 'PSO', family: 'population', evosax: 'PSO',
    defaults: { popsize: 32, sigma: 0.5, lr: 1 },
    parts: { state: ['swarm'], ask: ['velocity'], shape: ['greedy'], tell: ['memory'] },
    math: {
      ask: 'v ← ωv + c₁r₁(p_best − x) + c₂r₂(g_best − x),   x ← x + v',
      shape: 'improved_i = [ L(x_i) < L(p_best,i) ]',
      tell: 'p_best,i ← x_i if improved;   g_best ← best p_best',
    },
    init: (theta0, hp, rng) => ({ x: Array.from({ length: hp.popsize }, () => sampleAround(theta0, hp.sigma, randn(rng, D))), v: Array.from({ length: hp.popsize }, () => new Float64Array(D)), pbest: null }),
    ask(s, hp, rng) {
      if (s.pbest) s.x = s.x.map((x, i) => {
        s.v[i] = s.v[i].map((v, k) => 0.7 * v + 1.5 * rng.uniform() * (s.pbest[i][k] - x[k]) + 1.5 * rng.uniform() * (s.gbest[k] - x[k]));
        return x.map((xk, k) => xk + s.v[i][k]);
      });
      return { pop: s.x, eps: s.v, note: s.pbest ? 'Each particle keeps some momentum and is pulled toward its own best spot and the swarm\'s best spot.' : 'Generation 0: particles start at random positions with zero velocity.' };
    },
    shape(f, asked, s) {
      const w = s.pbest ? f.map((v, i) => +(v < s.pfit[i])) : f.map(() => 1);
      return { w, note: `${w.filter(Boolean).length} particles found a new personal best.` };
    },
    tell(s, asked, f, w) {
      if (!s.pbest) { s.pbest = s.x.map(x => x.slice()); s.pfit = f.slice(); }
      w.forEach((imp, i) => imp && ((s.pbest[i] = s.x[i].slice()), (s.pfit[i] = f[i])));
      s.gbest = s.pbest[argmin(s.pfit)];
      return `The global best has loss ${fmt(Math.min(...s.pfit))}. Every particle will be pulled toward it.`;
    },
    center: s => s.gbest || s.x[0],
  },
};

// Families: the "boxes" that group algorithms by how they think.
const FAMILIES = [
  { id: 'gradient', title: 'Exact gradient', blurb: 'Differentiate the loss through the network.', algos: ['backprop'] },
  { id: 'grad_est', title: 'ES · gradient estimators', blurb: 'Perturb, score, and average the noise into a gradient estimate.', algos: ['open_es', 'pgpe', 'eggroll'] },
  { id: 'node', title: 'Zeroth-order · node perturbation', blurb: 'Jitter neurons, not weights. One pass gives many virtual members.', algos: ['dust'] },
  { id: 'nes', title: 'ES · natural gradient', blurb: 'Ascend the search distribution\'s own parameters.', algos: ['snes'] },
  { id: 'recomb', title: 'ES · recombination & adaptation', blurb: 'Move to the elite average and reshape the cloud.', algos: ['simple_es', 'cma_es'] },
  { id: 'population', title: 'Population-based', blurb: 'Keep explicit individuals and select among them.', algos: ['simple_ga', 'de', 'pso'] },
  { id: 'topology', title: 'Beyond fixed weights', blurb: 'Evolves the network\'s shape too. Explained below, not runnable here.', algos: [], links: [['NEAT', '#neat']] },
];

// Numerical rank of a weight-matrix slice of a flat vector (Gram matrix eigenvalues).
function matrixRank(v, L) {
  const M = Array.from({ length: L.rows }, (_, r) => Array.from({ length: L.cols }, (_, c) => v[L.start + r * L.cols + c]));
  const G = Array.from({ length: L.cols }, (_, i) => Array.from({ length: L.cols }, (_, j) => M.reduce((a, row) => a + row[i] * row[j], 0)));
  const ev = jacobiEig(G).values, top = Math.max(...ev);
  return ev.filter(e => e > top * 1e-10).length;
}

// Plain-language version of every stage, shown while "Show math" is off.
const PLAIN = {
  eval: 'Runs every candidate on every dot (forward passes only) and gives it a score. Lower loss is better.',
  backprop: {
    state: 'One network. That is all it remembers.',
    ask: 'No variations. It tests the network itself.',
    shape: 'Runs the network backwards (calculus) to get the exact direction in which to turn every knob.',
    tell: 'Turns every knob a small step in that direction.',
  },
  simple_es: {
    state: 'A center network, plus how far to spread the copies for each knob.',
    ask: 'Makes random copies around the center.',
    shape: 'Only the best half get a vote. The very best gets the biggest one.',
    tell: 'The new center is the vote-weighted average of the winners. The spread adapts to how spread out the winners were.',
  },
  open_es: {
    state: 'A center network and a fixed noise size.',
    ask: 'Makes copies with random nudges, in pairs: +nudge and −nudge.',
    shape: 'Only the ranking matters. The best copy pulls hardest, the worst pushes away.',
    tell: 'Adds up all the nudges, weighted by their votes, to estimate the downhill direction, then takes a step.',
  },
  pgpe: {
    state: 'A center network and a separate noise size for every knob.',
    ask: 'Tries each random nudge both ways (+ and −), like testing a slope with both feet.',
    shape: 'Ranking-based votes, as in OpenES.',
    tell: 'Moves toward whichever side of each pair won. Widens the search for knobs where big nudges helped, and narrows it where they hurt.',
  },
  eggroll: {
    state: 'Same as OpenES: a center network and a noise size.',
    ask: 'Nudges each weight table with a cheap, structured pattern (a column times a row) instead of fully random noise.',
    shape: 'Ranking-based votes, as in OpenES.',
    tell: 'Combines all the nudges into one step. Each nudge is simple, but their sum is not.',
  },
  dust: {
    state: 'One network, like backprop. There is no population of weights.',
    ask: 'Leaves the weights alone and adds random jitter to every neuron\'s output, separately for each dot. That gives thousands of "virtual" variations at once.',
    shape: 'For each dot, checks whether the jitter made that dot\'s loss lower or higher than the average.',
    tell: 'Jitter that helped shows what each neuron should have output. Combined with the neuron\'s inputs, that gives a direction for every weight.',
  },
  snes: {
    state: 'A center network and a spread for each knob.',
    ask: 'Makes random copies around the center.',
    shape: 'Winners get positive votes, losers get small negative ones.',
    tell: 'Moves the center toward the winners. Grows or shrinks each knob\'s spread depending on whether far-out copies won.',
  },
  cma_es: {
    state: 'A center network, a step size, and a learned "shape" for the noise cloud: which knobs should move together.',
    ask: 'Makes copies from a stretched, tilted cloud that has learned good directions.',
    shape: 'Only the best half get a vote.',
    tell: 'Moves to the winners\' average, stretches the cloud along directions that keep working, and adjusts the step size.',
  },
  simple_ga: {
    state: 'A group of individual networks: the survivors.',
    ask: 'Breeds children: mixes the knobs of two survivors, then adds random mutations.',
    shape: 'Survival of the fittest. The top quarter survive; the rest are discarded.',
    tell: 'The survivors become the parents of the next generation.',
  },
  de: {
    state: 'A group of individual networks.',
    ask: 'For each network, builds a challenger by adding the difference between two other networks to a third.',
    shape: 'Each challenger competes head-to-head with the network it would replace.',
    tell: 'A challenger replaces its rival only if it is at least as good.',
  },
  pso: {
    state: 'A swarm. Each network has a position, a velocity, and a memory of its best spot so far.',
    ask: 'Each network flies a little, pulled toward its own best spot and toward the swarm\'s best.',
    shape: 'Checks which networks found a new personal best.',
    tell: 'Updates the memories. Everyone will now be pulled toward the new swarm best.',
  },
};
