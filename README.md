# Covariance shrinkage for minimum-variance portfolios

This is a learning project: I wanted to understand covariance shrinkage properly, by working
through the main papers and re-implementing the methods myself. It does not aim at new results.
So far it covers linear shrinkage (Ledoit and Wolf, 2004) and a test for comparing portfolio
variances (following Ledoit and Wolf, 2011), evaluated with the design of Ledoit and Wolf (2017)
on a different dataset. Nonlinear shrinkage comes next. I wrote the core implementation myself.
AI (Claude - Anthropic) served as a tutor, wrote the specifications, tests, data loaders and
experiment scripts, and drafted much of the write-up, which I revised; I take responsibility for its claims.

Full write-up: [PDF](build/WRITEUP.pdf), or [WRITEUP.md](WRITEUP.md).
