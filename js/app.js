// UI: renders the ask → evaluate → shape → tell loop and the network's forward pass.

const $ = s => document.querySelector(s);
const PHASES = ['ask', 'eval', 'shape', 'tell'];
const PHASE_INFO = {
  state: { title: 'State', sub: 'what it remembers', call: 'state = es.init(key, θ₀, params)' },
  ask: { title: 'Ask', sub: 'make candidates', call: 'population, state = es.ask(key, state, params)' },
  eval: { title: 'Evaluate', sub: 'test them', call: 'fitness = vmap(loss)(population)' },
  shape: { title: 'Shape', sub: 'scores → votes', call: 'fitness_shaping_fn(population, fitness, …)' },
  tell: { title: 'Tell', sub: 'update', call: 'state, metrics = es.tell(key, population, fitness, state, params)' },
};
const EVAL_MATH = 'L(θ) = −mean[ y log p + (1−y) log(1−p) ],   p = σ(W₂ tanh(W₁x + b₁) + b₂)';
// Centered ranks are loss-like (best = −½), so their pull toward a member is −w.
const PULL_SIGN = { open_es: -1, pgpe: -1, eggroll: -1 };
const EPS_LABEL = { simple_ga: 'child − parent', de: 'trial − parent', pso: 'velocity v_i', backprop: '—', dust: 'weight change (none)' };
const SOURCE = { backprop: 'jax.grad + optax', eggroll: 'Sarkar et al. 2025', dust: 'Q Labs 2026' };
const S = { algo: 'open_es', task: 'xor', seed: 7, selected: -1, point: 0, showGrad: false, showMath: false, timer: null };

// ---------- colors ----------
const POS = [33, 113, 181], NEG = [230, 110, 40], MID = [246, 246, 244];
function divColor(v) {
  const t = Math.max(-1, Math.min(1, v)), c = t > 0 ? POS : NEG, a = Math.abs(t);
  return `rgb(${MID.map((m, i) => Math.round(m + (c[i] - m) * a)).join(',')})`;
}
const chipHue = id => (Object.keys(CHIPS).indexOf(id) * 137.5) % 360;
const chip = id => `<span class="chip" data-chip="${id}" style="--h:${chipHue(id)}">${CHIPS[id]}</span>`;

// Paint fn(x1, x2) ∈ [-1, 1] over the input square [-1.1, 1.1]².
function paintMap(canvas, fn, res) {
  const ctx = canvas.getContext('2d'), cell = canvas.width / res;
  for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
    ctx.fillStyle = divColor(fn(-1.1 + (2.2 * (i + 0.5)) / res, 1.1 - (2.2 * (j + 0.5)) / res));
    ctx.fillRect(i * cell, j * cell, cell + 0.5, cell + 0.5);
  }
}
function mapURL(fn) {
  const c = document.createElement('canvas');
  c.width = c.height = 48;
  paintMap(c, fn, 24);
  return c.toDataURL();
}

// ---------- loop ----------
function reset() {
  stop();
  S.animId = (S.animId || 0) + 1; // cancels a running step animation
  const A = ALGORITHMS[S.algo];
  S.data = makeData(S.task);
  S.rng = makeRng(S.seed * 7919 + 1);
  S.es = A.init(initTheta(S.seed), S.hp, S.rng);
  Object.assign(S, { gen: 0, phase: null, asked: null, fit: null, shaped: null, notes: {}, selected: -1 });
  S.history = [{ gen: 0, center: loss(A.center(S.es, S.hp), S.data), best: null }];
  render();
}

function stepPhase() {
  const A = ALGORITHMS[S.algo];
  const next = S.phase === null || S.phase === 'tell' ? 'ask' : PHASES[PHASES.indexOf(S.phase) + 1];
  if (next === 'ask') {
    S.asked = A.ask(S.es, S.hp, S.rng, S.data);
    S.fit = S.shaped = null;
    S.notes = { ask: S.asked.note };
    if (S.selected >= S.asked.pop.length) S.selected = -1;
  } else if (next === 'eval') {
    S.fit = A.evaluate ? A.evaluate(S.asked, S.data, S.hp) : S.asked.pop.map(x => loss(x, S.data));
    S.notes.eval = `${S.fit.length} member(s) × ${S.data.length} points = ${S.fit.length * S.data.length} forward passes. Best loss ${fmt(Math.min(...S.fit))}, worst ${fmt(Math.max(...S.fit))}.`;
  } else if (next === 'shape') {
    S.shaped = A.shape(S.fit, S.asked, S.es, S.hp, S.data);
    S.notes.shape = S.shaped.note;
  } else {
    const before = A.center(S.es, S.hp).slice();
    S.notes.tell = A.tell(S.es, S.asked, S.fit, S.shaped.w, S.hp, S.data);
    if (S.es.grad && S.algo !== 'backprop') {
      const g = lossGrad(before, S.data), dot = g.reduce((a, x, k) => a + x * S.es.grad[k], 0);
      S.notes.tell += ` Cosine similarity with the true backprop gradient: ${fmt(dot / (norm(g) * norm(S.es.grad) + 1e-12), 2)}.`;
    }
    S.gen++;
    S.history.push({ gen: S.gen, center: loss(A.center(S.es, S.hp), S.data), best: Math.min(...S.fit) });
  }
  S.phase = next;
}
function stepGeneration() { do stepPhase(); while (S.phase !== 'tell'); }
function stop() { clearInterval(S.timer); S.timer = null; }
function toggleRun() {
  if (S.timer) return stop(), render();
  S.selected = -1;
  S.timer = setInterval(() => { stepGeneration(); render(); }, 60);
  render();
}

// What the network views display: a selected member, or the strategy's center.
function shown() {
  const A = ALGORITHMS[S.algo];
  if (S.selected >= 0 && S.asked) return { theta: S.asked.pop[S.selected], label: S.algo === 'dust' ? `draw #${S.selected} (same weights, jittered neurons)` : `member #${S.selected}` };
  const lbl = { gradient: 'θ', population: 'best individual' }[A.family] || 'mean μ';
  return { theta: A.center(S.es, S.hp), label: lbl };
}

// ---------- rendering ----------
function render() {
  const A = ALGORITHMS[S.algo];
  const nx = PHASE_INFO[S.phase === null || S.phase === 'tell' ? 'ask' : PHASES[PHASES.indexOf(S.phase) + 1]];
  $('#step').textContent = `Next: ${nx.title} (${nx.sub}) ▸`;
  $('#run').textContent = S.timer ? '❚❚ Pause' : '▶ Run';
  $('#gen').textContent = `generation ${S.gen}`;
  $('#popsize').disabled = S.algo === 'backprop';
  document.querySelectorAll('.algo-btn').forEach(b => b.classList.toggle('active', b.dataset.algo === S.algo));
  renderPipeline(A);
  const { theta, label } = shown();
  $('#net-title').textContent = `${A.name} · ${label}`;
  renderData(theta);
  renderNetwork(theta);
  renderDetail(theta, S.detail || 'out');
  renderPopulation(A);
  renderParams(A, theta);
  renderChart();
}

function renderPipeline(A) {
  const spread = A.spread && A.spread(S.es, S.hp);
  const stats = spread ? `mean σ = ${fmt(spread.reduce((a, b) => a + b, 0) / D)}` : '';
  const desc = p => (S.showMath ? (p === 'state' ? `${D} parameters. ${stats}` : p === 'eval' ? EVAL_MATH : A.math[p]) : p === 'eval' ? PLAIN.eval : PLAIN[S.algo][p]);
  const boxes = [['state', A.parts.state, desc('state'), ''], ...PHASES.map(p => [p, p === 'eval' ? ['forward'] : A.parts[p], desc(p), S.notes[p] || ''])];
  $('#pipeline').innerHTML = boxes.map(([p, chips, math, note], i) => `
    <div class="pbox ${p === S.phase ? 'active' : ''} ${p !== 'state' && S.phase && PHASES.indexOf(p) < PHASES.indexOf(S.phase) ? 'done' : ''}">
      <div class="pbox-head"><span class="pnum">${i}</span>${PHASE_INFO[p].title}<span class="psubt">${PHASE_INFO[p].sub}</span></div>
      ${S.showMath ? `<code class="call">${PHASE_INFO[p].call}</code>` : ''}
      <div class="chips">${chips.map(chip).join('')}</div>
      <div class="${S.showMath ? 'math' : 'plain'}">${math}</div>
      ${note ? `<div class="note">${note}</div>` : ''}
    </div>`).join('<div class="arrow">→</div>');
}

function renderData(theta) {
  const c = $('#dataplot'), ctx = c.getContext('2d'), W = c.width;
  paintMap(c, (x1, x2) => 2 * forward(theta, x1, x2).p - 1, 70);
  const px = x => ((x + 1.1) / 2.2) * W, py = y => ((1.1 - y) / 2.2) * W;
  S.data.forEach((d, i) => {
    ctx.beginPath();
    ctx.arc(px(d.x1), py(d.x2), i === S.point ? 7 : 4.5, 0, 2 * Math.PI);
    ctx.fillStyle = `rgb(${(d.y ? POS : NEG).join(',')})`;
    ctx.fill();
    ctx.lineWidth = i === S.point ? 3 : 1.2;
    ctx.strokeStyle = i === S.point ? '#111' : '#fff';
    ctx.stroke();
  });
  $('#data-stats').innerHTML = `loss <b>${fmt(loss(theta, S.data))}</b> · accuracy <b>${(100 * accuracy(theta, S.data)).toFixed(0)}%</b>`;
}

const NET = { in: [[70, 140], [70, 250]], hid: [50, 140, 230, 320].map(y => [300, y]), out: [530, 190], s: 56, so: 76 };
function renderNetwork(theta) {
  const d = S.data[S.point], f = forward(theta, d.x1, d.x2), bw = backward(theta, d.x1, d.x2, d.y);
  const g = S.showGrad ? lossGrad(theta, S.data) : null;
  const val = (k) => (g ? g[k] : theta[k]);
  const scale = Math.max(...Array.from({ length: D }, (_, k) => Math.abs(val(k)))) || 1;
  const edge = (x1, y1, x2, y2, k, layer) => {
    const v = val(k), mx = (x1 + x2) / 2;
    return `<path class="edge l${layer} ${v > 0 ? 'pos' : 'neg'}" data-k="${k}" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" style="stroke-width:${0.6 + (5 * Math.abs(v)) / scale}"><title>${g ? '∂L/∂' : ''}${paramName(k)} = ${fmt(v)}</title></path>`;
  };
  let html = '';
  NET.in.forEach(([x, y], i) => NET.hid.forEach(([hx, hy], j) => (html += edge(x + NET.s / 2, y, hx - NET.s / 2, hy, 2 * j + i, 1))));
  NET.hid.forEach(([x, y], j) => (html += edge(x + NET.s / 2, y, NET.out[0] - NET.so / 2, NET.out[1], 3 * H + j, 2)));
  const node = (id, x, y, s, fn, label, sub) => `
    <g class="node ${S.detail === id ? 'sel' : ''}" data-node="${id}">
      <image href="${mapURL(fn)}" x="${x - s / 2}" y="${y - s / 2}" width="${s}" height="${s}" preserveAspectRatio="none"/>
      <rect x="${x - s / 2}" y="${y - s / 2}" width="${s}" height="${s}" rx="4"/>
      <text x="${x}" y="${y + s / 2 + 13}" class="nlabel">${label}</text>
      ${sub ? `<text x="${x}" y="${y + s / 2 + 25}" class="nsub">${sub}</text>` : ''}
    </g>`;
  const xs = [d.x1, d.x2];
  // Dust: show the jitter this draw added to each neuron for the selected point
  const jit = S.algo === 'dust' && S.selected >= 0 && S.asked ? S.asked.noise[S.selected][S.point] : null;
  const hsub = j => (jit ? `jitter ${fmt(S.hp.sigma * jit.a1[j], 2)}` : S.showGrad ? `δ = ${fmt(bw.dz1[j], 3)}` : '');
  const osub = jit ? `jitter ${fmt(S.hp.sigma * jit.a2, 2)}` : S.showGrad ? `δ = p − y = ${fmt(bw.dz2, 3)}` : `target y = ${d.y}`;
  NET.in.forEach(([x, y], i) => (html += node(`in${i}`, x, y, NET.s, (a, b) => [a, b][i], `x${'₁₂'[i]} = ${fmt(xs[i], 2)}`)));
  NET.hid.forEach(([x, y], j) => (html += node(`h${j}`, x + 0, y, NET.s, (a, b) => forward(theta, a, b).h[j], `h${'₁₂₃₄'[j]} = ${fmt(f.h[j], 2)}`, hsub(j))));
  html += node('out', NET.out[0], NET.out[1], NET.so, (a, b) => 2 * forward(theta, a, b).p - 1, `p = ${fmt(f.p, 2)} → class ${+(f.p > 0.5)}`, osub);
  const hdr = (x, t) => `<text x="${x}" y="16" class="lhead">${t}</text>`;
  $('#net').innerHTML = hdr(70, 'input') + hdr(300, 'hidden · tanh') + hdr(530, 'output · sigmoid') + html;
  $('#net-legend').innerHTML = jit
    ? 'Dust never changes the weights while exploring. This draw added the purple jitter to each neuron\'s input sum, for the selected dot only; other dots get their own jitter. Jitter that lowered the loss tells each neuron which way it should have moved.'
    : S.showGrad
    ? 'Edge width is |∂L/∂w|: what backprop would compute for this network over the whole dataset. <span class="swatch pos"></span> positive gradient (decreasing w lowers the loss) <span class="swatch neg"></span> negative gradient.'
    : 'Each square shows what that neuron outputs across the whole input plane. Edge width is |w|. <span class="swatch pos"></span> positive weight <span class="swatch neg"></span> negative weight. Hover or click a neuron to see its arithmetic.';
}

function paramName(k) {
  if (k < 2 * H) return `W1[${Math.floor(k / 2) + 1},${(k % 2) + 1}]`;
  if (k < 3 * H) return `b1[${k - 2 * H + 1}]`;
  if (k < 4 * H) return `W2[${k - 3 * H + 1}]`;
  return 'b2';
}

function renderDetail(theta, id) {
  const d = S.data[S.point], f = forward(theta, d.x1, d.x2), bw = backward(theta, d.x1, d.x2, d.y);
  const row = (a, b, c) => `<tr><td>${a}</td><td>${b}</td><td class="num">${c}</td></tr>`;
  let html;
  if (id.startsWith('in')) {
    const i = +id[2];
    html = `<p><b>Input x${'₁₂'[i]}</b> is coordinate ${i + 1} of the selected point: <b>${fmt([d.x1, d.x2][i], 3)}</b>.</p><p>Its square is just a gradient across the plane. Every hidden neuron starts from these two ramps.</p>`;
  } else if (id.startsWith('h')) {
    const j = +id[1];
    html = `<p><b>Hidden neuron h${'₁₂₃₄'[j]}</b>: multiply each input by its weight, add a bias, then squash with tanh.</p><table>
      ${row('w₁·x₁', `${fmt(w1(theta, j, 0), 2)} × ${fmt(d.x1, 2)}`, fmt(w1(theta, j, 0) * d.x1))}
      ${row('w₂·x₂', `${fmt(w1(theta, j, 1), 2)} × ${fmt(d.x2, 2)}`, fmt(w1(theta, j, 1) * d.x2))}
      ${row('+ b', '', fmt(b1(theta, j)))}
      <tr class="sum"><td>z</td><td>sum</td><td class="num">${fmt(f.z1[j])}</td></tr>
      <tr class="sum"><td>h</td><td>tanh(z)</td><td class="num">${fmt(f.h[j])}</td></tr>
      ${S.showGrad ? row('δ = ∂L/∂z', '(p−y)·w₂·(1−h²)', fmt(bw.dz1[j])) : ''}</table>
      <p>The white band in its square is the line ${fmt(w1(theta, j, 0), 2)}·x₁ + ${fmt(w1(theta, j, 1), 2)}·x₂ + ${fmt(b1(theta, j), 2)} = 0. On one side tanh saturates to +1 (blue), on the other to −1 (orange). Every optimizer on this page is just moving these lines around.</p>`;
  } else {
    html = `<p><b>Output neuron</b>: weigh the four hidden neurons, add a bias, then squash to a probability with σ.</p><table>
      ${Array.from({ length: H }, (_, j) => row(`w·h${'₁₂₃₄'[j]}`, `${fmt(w2(theta, j), 2)} × ${fmt(f.h[j], 2)}`, fmt(w2(theta, j) * f.h[j]))).join('')}
      ${row('+ b', '', fmt(b2(theta)))}
      <tr class="sum"><td>z</td><td>sum</td><td class="num">${fmt(f.z2)}</td></tr>
      <tr class="sum"><td>p</td><td>σ(z)</td><td class="num">${fmt(f.p)}</td></tr>
      ${row('loss', `−log ${d.y ? 'p' : '(1−p)'}`, fmt(bce(f.p, d.y)))}</table>
      <p>The output is a vote of the four hidden maps, weighted by W₂. Its square is the classifier's decision map, the same one shown on the left.</p>`;
  }
  $('#detail').innerHTML = html;
}

function renderPopulation(A) {
  const pop = S.asked ? S.asked.pop : [];
  const order = S.fit && !S.unsorted ? [...pop.keys()].sort((a, b) => S.fit[a] - S.fit[b]) : [...pop.keys()];
  const wmax = S.shaped ? Math.max(...S.shaped.w.map(Math.abs)) || 1 : 1;
  const card = (i, theta, sub, bar, k = 0, vote = '') => `<div class="pcard ${S.selected === i ? 'sel' : ''} ${vote}" data-i="${i}" style="--k:${k}"><canvas width="52" height="52"></canvas><div class="psub">${sub}</div>${bar}</div>`;
  const centerLbl = { gradient: 'θ', population: 'best' }[A.family] || 'μ';
  let html = card(-1, null, `<b>${centerLbl}</b> now`, '');
  for (const [k, i] of order.entries()) {
    const w = S.shaped && S.shaped.w[i], pull = w * (PULL_SIGN[S.algo] || 1);
    const vote = !S.shaped ? '' : pull > 0 ? 'vote-up' : pull < 0 ? 'vote-down' : 'vote-none';
    const bar = S.shaped ? `<div class="wbar"><div style="width:${(50 * Math.abs(w)) / wmax}%;${pull >= 0 ? 'left:50%' : 'right:50%'};background:${divColor(pull >= 0 ? 0.9 : -0.9)}"></div></div><div class="psub">w ${fmt(w, 2)}</div>` : '';
    html += card(i, pop[i], S.fit ? `L ${fmt(S.fit[i], 2)}` : `#${i}`, bar, k, vote);
  }
  $('#pop').innerHTML = html;
  document.querySelectorAll('#pop .pcard').forEach(el => {
    const i = +el.dataset.i, theta = i < 0 ? A.center(S.es, S.hp) : pop[i];
    paintMap(el.querySelector('canvas'), (x1, x2) => 2 * forward(theta, x1, x2).p - 1, 26);
  });
  $('#pop-caption').textContent = S.algo === 'dust' && S.asked ? 'Dust\'s draws all share the same weights, so their maps look identical. The randomness lives on the neurons, per dot. Click a draw and look at the network: each neuron shows the jitter it received for the selected dot.'
    : S.unsorted ? 'Testing each candidate: one forward pass per dot, then a loss score. Next they get sorted, best first.'
    : !S.asked ? 'Press "Next: Ask" to sample the first generation.'
    : S.shaped ? `Sorted by loss. w is the shaped fitness. The bar shows its pull on the next update: blue pulls toward that member, orange pushes away${PULL_SIGN[S.algo] ? ' (centered ranks are loss-like, so the best members have w < 0 but pull hardest)' : ''}. For selection methods, 1 = kept.`
    : S.fit ? 'Sorted by loss (lower is better). Click a member to inspect its forward pass.'
    : 'Freshly sampled, not evaluated yet. Each thumbnail is a whole network\'s decision map.';
}

function renderParams(A, theta) {
  const center = A.center(S.es, S.hp), spread = A.spread && A.spread(S.es, S.hp);
  const rows = [['center', center, 'what the strategy believes now']];
  if (spread) rows.push(['σ', spread, 'search width per parameter']);
  if (S.selected >= 0 && S.asked) {
    rows.push([EPS_LABEL[S.algo] || 'noise ε_i', S.asked.eps[S.selected], 'how this member differs']);
    rows.push([`θ_${S.selected}`, theta, 'the member itself']);
  }
  if (S.showGrad) rows.push(['∇L', lossGrad(theta, S.data), 'true gradient (backprop)']);
  const head = `<div class="prow head"><span class="plabel"></span>${LAYOUT.map(L => `<span class="pgroup" style="flex:${L.len}">${L.name}</span>`).join('')}<span class="pscale"></span></div>`;
  const body = rows.map(([name, v, hint]) => {
    const sc = Math.max(...v.map(Math.abs)) || 1;
    return `<div class="prow"><span class="plabel" title="${hint}">${name}</span>${Array.from(v, (x, k) => `<span class="pcell ${LAYOUT.some(L => L.start === k && k) ? 'gap' : ''}" style="background:${divColor(x / sc)}" title="${paramName(k)} = ${fmt(x)}"></span>`).join('')}<span class="pscale">±${fmt(sc, 2)}</span></div>`;
  }).join('');
  let extra = '';
  if (S.algo === 'eggroll' && S.selected >= 0 && S.asked) {
    const { A: a, B: b } = S.asked.factors[S.selected].W1, sa = Math.max(...a.map(Math.abs)), sb = Math.max(...b.map(Math.abs));
    const cell = (x, s) => `<span class="mcell" style="background:${divColor(x / s)}" title="${fmt(x)}"></span>`;
    extra = `<div class="lowrank"><div><div class="mlabel">a (4×1)</div>${Array.from(a, x => `<div>${cell(x, sa)}</div>`).join('')}</div><div class="op">×</div>
      <div><div class="mlabel">bᵀ (1×2)</div><div>${Array.from(b, x => cell(x, sb)).join('')}</div></div><div class="op">=</div>
      <div><div class="mlabel">E_W1 = a bᵀ</div>${Array.from(a, x => `<div>${Array.from(b, y => cell(x * y, sa * sb)).join('')}</div>`).join('')}</div>
      <p>EGGROLL builds this member's W1 noise from 4 + 2 numbers instead of 8. Every row is a multiple of the same pattern bᵀ, which is what rank 1 means.</p></div>`;
  }
  if (A.cov) {
    const C = A.cov(S.es);
    extra = `<div class="covwrap"><canvas id="cov" width="170" height="170"></canvas><p>CMA-ES correlation matrix: C<sub>ij</sub>/√(C<sub>ii</sub>C<sub>jj</sub>). Off-diagonal colour means the search has learned that two weights should move together (blue) or opposite (orange).</p></div>`;
    $('#params').innerHTML = head + body + extra;
    const ctx = $('#cov').getContext('2d'), c = 170 / D;
    C.forEach((r, i) => r.forEach((x, j) => { ctx.fillStyle = divColor(x / Math.sqrt(C[i][i] * C[j][j])); ctx.fillRect(j * c, i * c, c, c); }));
    return;
  }
  $('#params').innerHTML = head + body + extra;
}

function renderChart() {
  const W = 320, Hh = 150, P = 28, h = S.history, n = Math.max(10, h.length - 1);
  const ymax = Math.max(0.1, ...h.flatMap(r => [r.center, r.best ?? 0])) * 1.05;
  const X = g => P + ((W - P - 8) * g) / n, Y = v => 8 + (Hh - 8 - 20) * (1 - v / ymax);
  const line = key => h.filter(r => r[key] != null).map((r, i) => `${i ? 'L' : 'M'}${X(r.gen).toFixed(1)},${Y(r[key]).toFixed(1)}`).join('');
  $('#chart').innerHTML = `
    <line x1="${P}" y1="${Y(0)}" x2="${W - 8}" y2="${Y(0)}" class="axis"/><line x1="${P}" y1="8" x2="${P}" y2="${Y(0)}" class="axis"/>
    <text x="${P - 4}" y="${Y(ymax / 1.05) + 4}" class="tick" text-anchor="end">${fmt(ymax / 1.05, 1)}</text>
    <text x="${P - 4}" y="${Y(0) + 4}" class="tick" text-anchor="end">0</text>
    <text x="${W - 8}" y="${Hh - 4}" class="tick" text-anchor="end">gen ${n}</text>
    <path d="${line('best')}" class="lbest"/><path d="${line('center')}" class="lcenter"/>`;
}

function renderFamilies() {
  $('#families').innerHTML = FAMILIES.map(F => `
    <div class="family">
      <div class="ftitle">${F.title}</div><div class="fblurb">${F.blurb}</div>
      <div class="fbtns">${F.algos.map(id => `<button class="algo-btn" data-algo="${id}">${ALGORITHMS[id].name}<small>${ALGORITHMS[id].evosax ? `evosax.${ALGORITHMS[id].evosax}` : SOURCE[id]}</small></button>`).join('')}
      ${(F.links || []).map(([t, href]) => `<a class="algo-btn link" href="${href}">${t}<small>read ↓</small></a>`).join('')}</div>
    </div>`).join('');
}

function renderComponentMap() {
  const cols = ['state', 'ask', 'shape', 'tell'];
  $('#cmap').innerHTML = `<thead><tr><th></th>${cols.map(c => `<th>${PHASE_INFO[c].title}</th>`).join('')}</tr></thead><tbody>${FAMILIES.filter(F => F.algos.length).map(F =>
    `<tr class="frow"><td colspan="5">${F.title}</td></tr>` + F.algos.map(id => `<tr><td class="aname">${ALGORITHMS[id].name}</td>${cols.map(c => `<td>${ALGORITHMS[id].parts[c].map(chip).join('')}</td>`).join('')}</tr>`).join('')).join('')}
    <tr class="frow"><td colspan="5">Shared by all</td></tr><tr><td class="aname">Evaluate</td><td colspan="4">${chip('forward')} The forward pass is the same for every method. Only what happens around it changes.</td></tr></tbody>`;
}

function animateForward() {
  const steps = [['.edge.l1', 'flow'], ['[data-node^="h"]', 'lit'], ['.edge.l2', 'flow'], ['[data-node="out"]', 'lit']];
  steps.forEach(([sel, cls], i) => setTimeout(() => {
    document.querySelectorAll('#net .flow, #net .lit').forEach(el => el.classList.remove('flow', 'lit'));
    document.querySelectorAll(`#net ${sel}`).forEach(el => el.classList.add(cls));
  }, i * 650));
  setTimeout(() => document.querySelectorAll('#net .flow, #net .lit').forEach(el => el.classList.remove('flow', 'lit')), steps.length * 650 + 400);
}

// ---------- step animations ----------
// Generic: they only use the population, fitness, shaped weights and the center θ before/after,
// so every algorithm gets them for free.
const sleep = ms => new Promise(r => setTimeout(r, ms));
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const ease = t => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
// Time-based (not rAF) so a hidden tab still finishes the animation instead of stalling the buttons.
const tween = (ms, f) => new Promise(done => {
  const t0 = performance.now();
  const tick = () => { const t = Math.min(1, (performance.now() - t0) / ms); f(ease(t)); t < 1 ? setTimeout(tick, 16) : done(); };
  tick();
});

// FLIP: slide population cards from their old spots to their new (sorted) spots.
function flipPopulation(update) {
  const first = new Map([...document.querySelectorAll('#pop .pcard')].map(el => [el.dataset.i, el.getBoundingClientRect()]));
  update();
  document.querySelectorAll('#pop .pcard').forEach(el => {
    const a = first.get(el.dataset.i), b = el.getBoundingClientRect();
    if (a) el.animate([{ transform: `translate(${a.left - b.left}px, ${a.top - b.top}px)` }, { transform: 'none' }], { duration: 650, easing: 'ease-in-out' });
  });
}

// Dots fly from every member that pulls the update into the center card.
function flyVotes() {
  const box = $('#pop').getBoundingClientRect(), to = $('#pop .pcard[data-i="-1"]').getBoundingClientRect();
  const tx = to.left + to.width / 2, ty = to.top + to.height / 2;
  [...document.querySelectorAll('#pop .pcard.vote-up')].filter(c => c.getBoundingClientRect().top < box.bottom).slice(0, 16).forEach((c, n) => {
    const r = c.getBoundingClientRect(), dot = document.createElement('div');
    dot.className = 'vote-dot';
    document.body.appendChild(dot);
    dot.animate([
      { transform: `translate(${r.left + r.width / 2}px, ${r.top + r.height / 2}px) scale(1)`, opacity: 1 },
      { transform: `translate(${tx}px, ${ty}px) scale(.35)`, opacity: 0.3 },
    ], { duration: 650, delay: n * 45, easing: 'ease-in', fill: 'both' }).onfinish = () => dot.remove();
  });
}

async function animatePhase() {
  const A = ALGORITHMS[S.algo], id = S.animId, before = A.center(S.es, S.hp).slice();
  const next = S.phase === null || S.phase === 'tell' ? 'ask' : PHASES[PHASES.indexOf(S.phase) + 1];
  if (next === 'tell') S.selected = -1; // watch the center move
  stepPhase();
  if (reduceMotion) return render();
  document.body.dataset.anim = S.phase;
  if (S.phase === 'ask') {
    render();
    await sleep(Math.min(1600, 500 + 25 * S.asked.pop.length));
  } else if (S.phase === 'eval') {
    S.unsorted = true; render(); S.unsorted = false; // score them in sampled order first
    animateForward();
    await sleep(Math.min(1800, 500 + 35 * S.asked.pop.length));
    if (id !== S.animId) return;
    document.body.dataset.anim = 'sort';
    flipPopulation(render);
    await sleep(750);
  } else if (S.phase === 'shape') {
    render();
    await sleep(Math.min(1600, 800 + 20 * S.asked.pop.length));
  } else {
    render();
    renderData(before); renderNetwork(before);
    flyVotes();
    await sleep(800);
    if (id !== S.animId) return;
    const after = A.center(S.es, S.hp);
    await tween(1000, t => { if (id !== S.animId) return; const th = before.map((x, k) => x + (after[k] - x) * t); renderData(th); renderNetwork(th); });
    if (id !== S.animId) return;
    render();
    $('#pop .pcard[data-i="-1"]').classList.add('pulse');
    await sleep(500);
  }
}

// Animations queue up; a reset (new algorithm, tour step…) makes the running one bail out early.
function runAnimated(fn) {
  S.animChain = (S.animChain || Promise.resolve()).then(async () => {
    stop();
    ['#step', '#stepgen', '#run'].forEach(b => ($(b).disabled = true));
    try { await fn(); } finally {
      ['#step', '#stepgen', '#run'].forEach(b => ($(b).disabled = false));
      delete document.body.dataset.anim;
    }
  });
}
async function animateGeneration() {
  const id = S.animId;
  do await animatePhase(); while (S.phase !== 'tell' && id === S.animId);
}

// ---------- guided tour ----------
// Each step rebuilds its state from scratch (seeded), so Back/Next always show the same thing.
function goto(algo, phases) {
  if (S.algo !== algo || S.gen || S.phase) selectAlgo(algo);
  for (let i = 0; i < phases - 1; i++) stepPhase();
  render();
  if (phases > 0) runAnimated(animatePhase);
}
const TOUR = [
  { t: 'Welcome', focus: null, run: () => goto('open_es', 0),
    x: 'This page shows, step by step, how a computer can <b>learn</b>. First the classic way (backpropagation), then by <b>evolution</b>: try lots of random variations and keep what works. No math needed. Press <b>Next</b> to walk through, or close the guide at any time and explore on your own.' },
  { t: 'The task', focus: '#panel-data', run: () => goto('open_es', 0),
    x: 'Each dot is blue or orange. We want a program that looks at a dot\'s position and guesses its colour. The background shows the program\'s current guess everywhere: a blue area means "I think blue here". Right now many guesses are wrong.' },
  { t: 'The network', focus: '#panel-net', run: () => { goto('open_es', 0); setTimeout(animateForward, 400); },
    x: 'The guessing program is a tiny <b>neural network</b>. A dot\'s two coordinates go in on the left. Four "neurons" in the middle each draw one line across the map, and the right side combines them into a final guess. Running a dot through the network from left to right is a <b>forward pass</b>. Click any dot on the map to trace it.' },
  { t: 'Inside a neuron', focus: '#panel-detail', run: () => { goto('open_es', 0); S.detail = 'h0'; render(); },
    x: 'A neuron does three small things: multiply each input by a <b>weight</b>, add them up together with a <b>bias</b>, then squash the result into the range −1 to +1. Hover over any square in the network to see its numbers.' },
  { t: 'Learning = tuning 17 knobs', focus: '#panel-params', run: () => goto('open_es', 0),
    x: 'All the weights and biases together are just 17 numbers. Think of them as knobs; each coloured cell here is one knob. Learning means finding knob settings that make the guesses right. The <b>loss</b> is one number for how wrong the network is: lower is better.' },
  { t: 'Way 1 · Backpropagation', focus: '#pipeline', run: () => goto('backprop', 0),
    x: 'If we can do calculus through the network, we can compute exactly which way to turn every knob to lower the loss. That is <b>backpropagation</b>, the standard method. Press <b>Full generation</b> a few times and watch the map improve.' },
  { t: 'Way 2 · Evolution strategies', focus: '#families', run: () => goto('open_es', 0),
    x: 'Now pretend the network is a sealed black box: no calculus, we can only run it and read its score. Evolution strategies work like walking downhill in thick fog. Send friends a few steps in random directions, ask who ended up lowest, and walk that way. We will use <b>OpenES</b>, a classic evolution strategy.' },
  { t: 'Step 1 · Ask: make variations', focus: '#panel-pop', run: () => goto('open_es', 1),
    x: 'The algorithm keeps one "center" network (μ). It makes 32 copies, each with its knobs nudged by a little random noise. Each thumbnail is one copy\'s guess map, and they all differ slightly. Click one to load it into the network view.' },
  { t: 'Step 2 · Evaluate: test them', focus: '#panel-pop', run: () => goto('open_es', 2),
    x: 'Every copy does a forward pass on every dot and gets a loss score. The copies are now sorted, best (lowest loss) first. Notice that no calculus happened. Only forward passes.' },
  { t: 'Step 3 · Shape: scores become votes', focus: '#panel-pop', run: () => goto('open_es', 3),
    x: 'The raw scores are turned into votes. OpenES only uses the <b>ranking</b>: the best copy pulls hardest and the worst pushes away. The little bar under each copy is its vote, blue for a pull and orange for a push.' },
  { t: 'Step 4 · Tell: move the center', focus: '#pipeline', run: () => goto('open_es', 4),
    x: 'The center moves toward the copies that won and away from the ones that lost. That completes one <b>generation</b>. The "Tell" box also compares this guessed direction with the exact one backpropagation would compute (cosine similarity: 1 means identical).' },
  { t: 'Repeat', focus: '.toolbar', run: () => goto('open_es', 4),
    x: 'Press <b>Run</b> to repeat ask → evaluate → shape → tell over and over, and watch the map and the loss curve. Evolution needs many more forward passes than backprop, but it never needs calculus. That means it also works on things you cannot differentiate.' },
  { t: 'Explore the family', focus: '#families', run: () => goto('open_es', 0),
    x: 'Every other algorithm changes only one or two of the four steps; the coloured tags in the pipeline show which. <b>CMA-ES</b> learns the shape of its noise. A <b>genetic algorithm</b> keeps only the winners and breeds them. <b>Dust</b> jitters neurons instead of weights. Pick one and step through it. Turn on <b>Show math</b> for the formulas, and scroll down for the full explanations.' },
];
function showTour(i) {
  S.tour = Math.max(0, Math.min(TOUR.length - 1, i));
  const step = TOUR[S.tour];
  $('#tour').hidden = false; $('#tour-open').hidden = true;
  $('#tour-n').textContent = `${S.tour + 1} / ${TOUR.length}`;
  $('#tour-title').textContent = step.t;
  $('#tour-text').innerHTML = step.x;
  $('#tour-prev').disabled = S.tour === 0;
  $('#tour-next').textContent = S.tour === TOUR.length - 1 ? 'Finish ✓' : 'Next ▸';
  step.run();
  document.querySelectorAll('.spotlight').forEach(el => el.classList.remove('spotlight'));
  if (step.focus) { const el = $(step.focus); el.classList.add('spotlight'); el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
}
function closeTour() {
  $('#tour').hidden = true; $('#tour-open').hidden = false;
  document.querySelectorAll('.spotlight').forEach(el => el.classList.remove('spotlight'));
}

// ---------- wiring ----------
function readHp() {
  S.hp = { popsize: +$('#popsize').value, sigma: +$('#sigma').value, lr: +$('#lr').value };
  $('#popsize-val').textContent = S.hp.popsize;
}
function selectAlgo(id) {
  S.algo = id;
  const d = ALGORITHMS[id].defaults;
  $('#popsize').value = Math.max(4, d.popsize); $('#sigma').value = d.sigma; $('#lr').value = d.lr;
  $('#sigma').disabled = id === 'backprop';
  $('#lr-label').textContent = { de: 'F', simple_es: 'lr (unused)', cma_es: 'lr (unused)', simple_ga: 'lr (unused)', pso: 'lr (unused)' }[id] || 'learning rate';
  readHp();
  if (id === 'backprop') S.hp.popsize = 1;
  reset();
}

document.addEventListener('DOMContentLoaded', () => {
  renderFamilies();
  renderComponentMap();
  $('#families').addEventListener('click', e => { const b = e.target.closest('button.algo-btn'); if (b) selectAlgo(b.dataset.algo); });
  $('#step').onclick = () => runAnimated(animatePhase);
  $('#stepgen').onclick = () => runAnimated(animateGeneration);
  $('#run').onclick = toggleRun;
  $('#reset').onclick = reset;
  $('#animate').onclick = animateForward;
  $('#showgrad').onchange = e => { S.showGrad = e.target.checked; render(); };
  $('#showmath').onchange = e => { S.showMath = e.target.checked; render(); };
  $('#tour-next').onclick = () => (S.tour === TOUR.length - 1 ? closeTour() : showTour(S.tour + 1));
  $('#tour-prev').onclick = () => showTour(S.tour - 1);
  $('#tour-close').onclick = closeTour;
  $('#tour-open').onclick = () => showTour(0);
  $('#task').onchange = e => { S.task = e.target.value; reset(); };
  $('#seed').onchange = e => { S.seed = +e.target.value || 1; reset(); };
  ['#popsize', '#sigma', '#lr'].forEach(sel => ($(sel).oninput = () => { readHp(); if (S.algo === 'backprop') S.hp.popsize = 1; reset(); }));
  $('#dataplot').onclick = e => {
    const r = e.target.getBoundingClientRect(), x1 = ((e.clientX - r.left) / r.width) * 2.2 - 1.1, x2 = 1.1 - ((e.clientY - r.top) / r.height) * 2.2;
    S.point = argmin(S.data.map(d => (d.x1 - x1) ** 2 + (d.x2 - x2) ** 2));
    render();
  };
  $('#pop').onclick = e => { const c = e.target.closest('.pcard'); if (c) { S.selected = +c.dataset.i; render(); } };
  const pickNode = e => { const n = e.target.closest('.node'); if (n) { S.detail = n.dataset.node; renderDetail(shown().theta, S.detail); document.querySelectorAll('#net .node').forEach(el => el.classList.toggle('sel', el === n)); } };
  $('#net').addEventListener('mouseover', pickNode);
  $('#net').addEventListener('click', pickNode);
  // hover a component chip anywhere → highlight every algorithm that shares it
  document.addEventListener('mouseover', e => {
    const c = e.target.closest('.chip');
    document.querySelectorAll('.chip.hl').forEach(el => el.classList.remove('hl'));
    if (c) document.querySelectorAll(`.chip[data-chip="${c.dataset.chip}"]`).forEach(el => el.classList.add('hl'));
  });
  selectAlgo(S.algo);
  showTour(0);
});
