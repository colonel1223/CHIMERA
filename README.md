# CHIMERA

Statistical analysis of hallucination patterns across 847K LLM inference traces.

## Overview

CHIMERA explores whether hallucination in large language models is a reducible engineering problem or a fundamental limitation of autoregressive generation. The project analyzes inference traces across multiple model families and task types to characterize failure distributions.

Key findings:
- Hallucination frequency follows heavy-tailed distributions that resist standard mitigation
- Confidence calibration degrades non-monotonically with model scale in specific task domains
- Formal impossibility result: perfect hallucination elimination under bounded compute is provably infeasible for certain query classes

## Data

Analysis covers 847K inference traces spanning summarization, QA, code generation, and open-ended dialogue across 4 model families.

## Structure

```
├── analysis/       # Statistical analysis notebooks
├── data/           # Processed trace data and metadata
├── proofs/         # Formal impossibility theorem and supporting lemmas
└── figures/        # Generated plots and visualizations
```

## Usage

```bash
pip install -r requirements.txt
python run_analysis.py --traces data/traces.parquet
```

## Citation

If you use this analysis in your work, please cite this repository.

## License

MIT
