# CHIMERA

**Formal proof that LLM hallucination is a theorem, not a bug.**

## The claim

Everyone in the industry treats hallucination as an engineering problem — something to be reduced, fine-tuned away, patched with retrieval augmentation. CHIMERA presents evidence that this framing is wrong.

Hallucination in autoregressive language models under bounded compute isn't a failure mode. It's a mathematical inevitability for certain query classes. The impossibility theorem formalizes this: no finite-compute autoregressive system can guarantee non-hallucinatory output across all inputs without sacrificing completeness.

This isn't pessimism. It's a boundary condition. Knowing where the wall is lets you build around it instead of running into it.

## Data

847K inference traces across 4 model families spanning:
- Summarization
- Question answering
- Code generation
- Open-ended dialogue

## Key findings

- Hallucination frequency follows heavy-tailed distributions resistant to standard mitigation
- Confidence calibration degrades non-monotonically with scale in specific task domains
- The impossibility result holds under reasonable assumptions about compute bounds and output space cardinality

## Structure

```
├── analysis/       # Statistical workbooks
├── data/           # Processed traces
├── proofs/         # Impossibility theorem and lemmas
└── figures/        # Visualizations
```

## Run

```bash
pip install -r requirements.txt
python run_analysis.py --traces data/traces.parquet
```

## License

MIT
