# CHIMERA

**Hallucination is a theorem, not a bug.**

## What this is

CHIMERA is an interactive research environment that presents the case for LLM hallucination as a structural property of autoregressive generation under bounded compute — not an engineering flaw to be patched.

The project combines a formal impossibility argument with an analysis of 847K inference traces across multiple model families, rendered as a live web experience that lets you explore the data and the reasoning together.

## Key claims

- Hallucination frequency follows heavy-tailed distributions that resist standard mitigation
- Confidence calibration degrades non-monotonically with model scale in specific task domains
- Perfect hallucination elimination under bounded compute is provably infeasible for certain query classes

## Structure

```
├── index.html                 # Main research environment (interactive)
├── CHIMERA_Model_v2.jsx       # React visualization component
├── CHIMERA_Research_Paper.docx # Formal writeup with proofs
└── DEPLOY.sh                  # GitHub Pages deployment
```

## Run

Open `index.html` in any browser, or visit the live version via GitHub Pages.

## License

MIT
