# CHIMERA

**Hallucination is a theorem, not a bug.**

847K inference traces. Heavy-tailed distributions. An impossibility argument. Rendered as an interactive research environment.

## The claim

Hallucination in autoregressive LLMs under bounded compute isn't a failure mode to be patched. It's a mathematical property. CHIMERA presents the evidence: statistical analysis of inference traces across model families, and a formal argument for why perfect elimination is provably infeasible for certain query classes.

## What's here

- **Interactive research environment** (`index.html`) — Explore the data and the argument together in the browser
- **React visualization** (`CHIMERA_Model_v2.jsx`) — Trace analysis visualization component
- **Formal writeup** (`CHIMERA_Research_Paper.docx`) — Full paper with proofs
- **Analysis code** (`analysis/trace_analysis.py`) — Statistical tools: power-law tail fitting, calibration-by-scale measurement, impossibility bound computation

## Quick start

```bash
# Interactive environment
open index.html

# Run trace analysis
python analysis/trace_analysis.py
```

## Key findings

- Hallucination frequency follows heavy-tailed distributions (tail index < 2 = infinite variance)
- Confidence calibration degrades non-monotonically with model scale
- Impossibility bound: given finite compute budget, minimum hallucination rate is provably > 0 for open-ended generation

## Structure

```
├── index.html                    # Interactive research environment
├── CHIMERA_Model_v2.jsx          # React visualization
├── CHIMERA_Research_Paper.docx   # Formal paper with proofs
├── analysis/
│   └── trace_analysis.py         # Statistical analysis tools
├── DEPLOY.sh
└── README.md
```

## License

MIT
