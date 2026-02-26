# CHIMERA

**Hallucination is a theorem, not a bug.**

847K inference traces. Heavy-tailed distributions. An impossibility argument. Rendered as an interactive research environment.

## The claim

Hallucination in autoregressive LLMs under bounded compute isn't a failure mode to be patched — it's a structural property of open-ended generation over combinatorial output spaces. CHIMERA presents the evidence.

## What's here

| Path | What |
|------|------|
| `index.html` | Interactive research environment (browser) |
| `CHIMERA_Model_v2.jsx` | React visualization component |
| `CHIMERA_Research_Paper.docx` | Formal paper |
| `analysis/distributions.py` | Power-law tail fitting (Hill estimator with bootstrap KS), log-normal/exponential comparison |
| `analysis/impossibility.py` | Information-theoretic lower bounds on hallucination rate |
| `analysis/entropy.py` | Shannon entropy, KL divergence, MI estimation for calibration analysis |

## Key results

**Heavy tails**: Hallucination lengths follow a power law with tail index α < 2 (infinite variance). Standard error bars for hallucination rate are mathematically meaningless.

**Impossibility bound**: For vocabulary V, sequence length L, and compute budget C, the hallucination floor is 1 - C/(c·L·|V|^L). For any realistic parameters, this is indistinguishable from 1 for L > 64.

**Calibration failure**: MI(confidence; correctness) is low — model confidence is near-uninformative about actual accuracy.

## Run

```bash
# Interactive environment
open index.html

# Impossibility bounds
python analysis/impossibility.py

# Distribution fitting (requires scipy)
python -c "
from analysis.distributions import hill_estimator
import numpy as np
data = np.random.pareto(1.5, 10000) + 1
fit = hill_estimator(data)
print(f'α={fit.alpha:.2f}, finite_var={fit.finite_variance}')
"
```

## License

MIT

---

## Research Ecosystem

CHIMERA is part of a unified AI safety research program. See [colonel1223.net](https://colonel1223.net) for the full portfolio.

**Live Demo:** [colonel1223.github.io/CHIMERA](https://colonel1223.github.io/CHIMERA/)

| Related Project | Connection |
|----------------|------------|
| [conformal-multimodal](https://github.com/colonel1223/conformal-multimodal) | Distribution-free methods applied to hallucination confidence analysis |
| [alignment-tax-quantifier](https://github.com/colonel1223/alignment-tax-quantifier) | Alignment interventions change the hallucination distribution — measured here |
| [SHOGGOTH](https://github.com/colonel1223/SHOGGOTH) | Visual complement — CHIMERA proves the bound, SHOGGOTH shows what failure looks like |
| [Research Papers](https://github.com/colonel1223/ai-research-modern-alchemy) | Information-theoretic impossibility analysis and heavy-tailed statistics |
