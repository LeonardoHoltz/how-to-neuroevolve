# How to Neuroevolve

An interactive explainer of evolution strategies, in the spirit of [CNN Explainer](https://poloclub.github.io/cnn-explainer/).
Eleven optimizers train the same tiny neural network (2 inputs → 4 tanh neurons → 1 sigmoid output, 17 weights) on a 2-D classification task.
You can step through each one stage by stage (**ask → evaluate → shape → tell**) and watch every forward pass.

Algorithms: backprop + gradient descent, OpenES, PGPE, EGGROLL, Dust, SNES, SimpleES, CMA-ES, SimpleGA, Differential Evolution and PSO. NEAT is explained but not runnable.
Most are modeled on [evosax](https://github.com/RobertTLange/evosax).

## Run it

There is nothing to install or build. Either open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. A guided tour starts automatically. Reopen it any time with **? Guided tour** in the bottom-right corner.

## Test

Requires Node.js:

```bash
node test.js
```

This checks the backprop gradient against finite differences and the eigensolver used by CMA-ES. It also checks that Dust's gradient estimate points in the same direction as the true gradient, and that every algorithm reaches at least 90 % accuracy on every task.

## Files

| File | What it holds |
|---|---|
| `index.html` | Page layout and the written explanations |
| `style.css` | Styles and step animations |
| `js/core.js` | Network, backward pass, datasets, fitness shaping, optimizers |
| `js/algorithms.js` | Every algorithm as `init / ask / shape / tell`, plus the component and family tables |
| `js/app.js` | Rendering, step animations and the guided tour |
| `test.js` | Self-check (above) |

## Adding an algorithm

1. Add an entry to `ALGORITHMS` in `js/algorithms.js` with `init`, `ask`, `shape`, `tell` and `center`.
2. List the components it uses in `parts`, and add any new component to `CHIPS`.
3. Put it in a family in `FAMILIES`, and write its plain-language text in `PLAIN`.

The visualizations and animations only use those functions, so the new algorithm works with them automatically. `node test.js` will then check that it learns every task.

## Notes

The algorithms are simplified JavaScript re-implementations for teaching. Their default settings are tuned for this toy task, so use evosax for real work.
EGGROLL ([Sarkar et al., 2025](https://arxiv.org/abs/2511.16652)) and Dust ([Q Labs, 2026](https://qlabs.sh/research/dust)) are not part of evosax. Their versions here are minimal adaptations to the 2-4-1 network.
