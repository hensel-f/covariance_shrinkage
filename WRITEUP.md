# Covariance shrinkage for minimum-variance portfolios

**Purpose.** This is a learning project: I wanted to understand covariance shrinkage properly,
by working through the main papers and re-implementing the methods myself. It does not aim at
new results. So far it covers linear shrinkage (Ledoit and Wolf, 2004) and a test for comparing
portfolio variances (following Ledoit and Wolf, 2011), evaluated with the design of Ledoit and
Wolf (2017) on a different dataset. Nonlinear shrinkage comes next (Section 7). I wrote the core
implementation myself. AI (Claude - Anthropic) served as a tutor, wrote the specifications,
tests, data loaders and experiment scripts, and drafted much of this write-up, which I revised; I take responsibility for its claims.

## 1. Question

A minimum-variance portfolio needs the inverse of a covariance matrix, and the sample
covariance matrix is a poor estimate of it when the number of assets $N$ is not small
against the length $T$ of the estimation window. Its smallest eigenvalues are biased
downwards, the optimiser loads on the directions they belong to, and the realised risk is
higher than the predicted risk (Section 3.3).

The question these papers ask, and this project works through:

> **Question.** Does shrinking the covariance matrix lower the out-of-sample risk of the
> global minimum-variance (GMV) portfolio? By how much, and how does the answer depend on
> the ratio $N/T$?

The benchmark is Ledoit and Wolf (2017), who ran this comparison on US stocks; this
project follows their evaluation design (Section 4).

## 2. Data

| Item     | Choice                                                         |
| -------- | -------------------------------------------------------------- |
| Universe | French's 100 portfolios on size and book-to-market (10 × 10)   |
| Weights  | value-weighted, daily returns                                  |
| Sample   | 2 Jan 1973 to 31 Aug 2026, 13,529 trading days                 |
| Returns  | simple returns, in excess of French's daily risk-free rate     |
| Missing  | a cell that held no stocks; whole formation years after 1970   |

**Notes.**

- **The risk-free rate** is the one-month T-bill rate, which French publishes per day,
  rounded to 0.01 %.
- **Missing values.** A missing value means that a cell of the size/book-to-market grid held
  no stocks. After 1970 every missing block is one or more whole formation years (July to
  June), with one exception: the largest, highest book-to-market cell is empty from 23 Feb
  2001, in the middle of a formation year, until June 2004.
- **Why the sample starts in 1973.** This is where Ledoit and Wolf's (2017) out-of-sample
  period begins. It is also the first year in which the portfolios include NASDAQ stocks.
  They are built from CRSP (the Center for Research in Security Prices, the standard
  database of US stock returns), whose NASDAQ data begin on 14 Dec 1972 (AMEX: 2 Jul 1962),
  so before 1973 the small-cap cells hold different stocks.

**Source.** Kenneth R. French Data Library; the two files are downloaded on the first run
and cached (`covariance_shrinkage/data.py`).

- Library: <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>
- Files, in <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/>:
  - `100_Portfolios_10x10_Daily_CSV.zip`, section
    `Average Value Weighted Returns -- Daily`;
  - `F-F_Research_Data_Factors_daily_CSV.zip`, column `RF`.

French updates them monthly, so a later download extends the sample.

### Why this dataset

1. **Free and reproducible.** Ledoit and Wolf (2017) use CRSP stocks, which need a
   subscription; this project runs on public files. The price: the comparison with their
   results is qualitative, not number for number.
2. **Survivorship-free.** Free stock-level data usually covers only firms that still exist,
   which drops the volatile firms that later failed and so understates risk. French's
   portfolios are re-formed each June from all CRSP stocks listed at the time, dead firms
   included, so nothing drops out after a crash or a delisting.
3. **The right ratio $N/T$.** With $N = 100$, windows of half a year to two years give
   $N/T$ from 0.8 to 0.2: the sample covariance matrix is invertible but noisy, the regime
   in which shrinkage should matter.
4. **Strongly correlated assets.** The average pairwise correlation is 0.73, so the
   covariance matrix has one large "market" eigenvalue and many small ones: its condition
   number is about 1,200 on the full sample. In 125-day windows estimation noise multiplies
   it by about 30 (median $3.8 \cdot 10^4$ over the 621 rebalances of the backtest).
5. **Long history.** Over fifty years of daily data, with several crises, give 13,029
   out-of-sample days for the tests.

## 3. Method

### 3.1 Notation

| Symbol                      | Meaning                                       |
| --------------------------- | --------------------------------------------- |
| $N$, $T$                    | number of assets; estimation window, in days  |
| $\Sigma$                    | true covariance matrix of daily returns       |
| $S$                         | sample covariance matrix of the window        |
| $S^*$                       | linear-shrinkage estimate (Section 3.4)       |
| $\hat\Sigma$                | the estimate the portfolio uses: $S$ or $S^*$ |
| $\lambda_i$, $u_i$          | eigenvalues, eigenvectors of $\Sigma$         |
| $\hat\lambda_i$, $\hat u_i$ | eigenvalues, eigenvectors of $S$              |
| $\mathbf 1$                 | the vector of $N$ ones                        |
| $w$                         | portfolio weights ($N$-vector)                |

Eigenvalues are in decreasing order: $\lambda_1 \ge \dots \ge \lambda_N$, and likewise
$\hat\lambda_i$. $\|v\|$ is the Euclidean norm of a vector; for matrices,
$\|A\|^2 = \operatorname{tr}(AA^\top)/N$, the normalised Frobenius norm of Ledoit and Wolf
(2004).

### 3.2 Portfolio rule

At each rebalance, the GMV portfolio is computed from $\hat\Sigma$, estimated on the last
$T$ days:

$$
\hat w = \frac{\hat\Sigma^{-1}\mathbf 1}{\mathbf 1^\top \hat\Sigma^{-1}\mathbf 1},
\qquad \text{the minimiser of } w^\top\hat\Sigma w \text{ subject to } \mathbf 1^\top w = 1 .
$$

- **Short positions are allowed.**
- **No explicit inverse.** The weights come from solving $\hat\Sigma x = \mathbf 1$ with a
  Cholesky factorisation. A matrix that is not positive definite raises an error rather
  than falling back to a pseudo-inverse.
- **Benchmark.** The 1/N portfolio needs no covariance estimate; it is the naive benchmark
  (DeMiguel, Garlappi and Uppal, 2009).

### 3.3 Why the sample covariance matrix fails

> **In short.** $S$ is right in expectation, but its eigenvalues are too spread out: the
> smallest are too small. GMV puts its weight exactly there, so its predicted risk is too
> low and its realised risk too high.

$S$ is unbiased, $E[S] = \Sigma$ (with the $1/T$ normalisation of Section 3.4, up to a
factor $(T-1)/T$ that plays no role here). Its eigenvalues are biased (points 1–3), and GMV
amplifies the bias (points 4–5).

**1. The extreme eigenvalues are biased outwards.** The largest eigenvalue is the largest
variance that $S$ assigns to any direction, so it picks the direction in which the noise
happens to add variance. Comparing with the fixed direction $u_1$:

$$
\hat\lambda_1 = \max_{\|v\| = 1} v^\top S v \;\ge\; u_1^\top S u_1
\quad\Rightarrow\quad
E[\hat\lambda_1] \;\ge\; u_1^\top \Sigma u_1 = \lambda_1 ,
$$

and in the same way $E[\hat\lambda_N] \le \lambda_N$.

**2. The whole spectrum is too dispersed.** With $\mu = \operatorname{tr}\Sigma / N$, the
mean of the true eigenvalues (Ledoit and Wolf, 2004, Lemma 2.1, p. 368, and eq. (12),
p. 372):

$$
E\Big[\frac1N \sum_{i=1}^N (\hat\lambda_i - \mu)^2\Big]
\;=\; \frac1N \sum_{i=1}^N (\lambda_i - \mu)^2 \;+\; E\|S - \Sigma\|^2 .
$$

The sample eigenvalues are centred correctly, but their dispersion exceeds the true one by
exactly the estimation error of $S$.

For a symmetric matrix $A$, $\|A\|^2$ is the mean of its squared eigenvalues, so the left
side is $E\|S - \mu I\|^2$ and the first term on the right is $\|\Sigma - \mu I\|^2$.
Splitting $S - \mu I = (S - \Sigma) + (\Sigma - \mu I)$ and expanding the square:

$$
\begin{aligned}
\|S - \mu I\|^2 &= \|S - \Sigma\|^2 + \|\Sigma - \mu I\|^2 +{}\\
&\quad \tfrac2N \operatorname{tr}\big((S - \Sigma)(\Sigma - \mu I)\big) .
\end{aligned}
$$

The cross term has expectation zero: $\Sigma - \mu I$ is fixed, so by linearity of the
trace the expectation falls on $S - \Sigma$ alone, which has mean zero. This is the matrix
form of $E(X - c)^2 = \operatorname{Var} X + (EX - c)^2$ for a random variable $X$ and a
constant $c$.

**3. The size of the effect is set by $N/T$.** Even when all true eigenvalues are equal,
$\Sigma = \sigma^2 I$, the sample eigenvalues spread over

$$
\Big[\sigma^2\big(1 - \sqrt{N/T}\big)^2,\; \sigma^2\big(1 + \sqrt{N/T}\big)^2\Big]
$$

as $N$ and $T$ grow (Marchenko and Pastur, 1967; Bun, Bouchaud and Potters, 2017, §2.2.3,
eq. (2.41)). At $N/T = 0.8$ the smallest is about 1 % of $\sigma^2$.

- **As $N/T \to 1$** the lower edge goes to 0, so GMV degrades before $S$ is singular; for
  $N \ge T$, $S$ is singular and calls some portfolios riskless.
- **At small $N/T$** the spread is small ($0.81\sigma^2$ to $1.21\sigma^2$ at 0.01), but only
  because the window is long, and old data may no longer describe today's covariance. The
  window trades noise against staleness.

**4. GMV loads on the bottom of the spectrum.** For $\hat\Sigma = S$, write $S$ in its
eigenbasis, $S = \sum_i \hat\lambda_i\, \hat u_i \hat u_i^\top$. Inverting replaces each
eigenvalue by its reciprocal and keeps the eigenvectors, so

$$
\begin{aligned}
\hat w \;\propto\; S^{-1}\mathbf 1
&= \sum_{i=1}^N \frac{1}{\hat\lambda_i}\, \hat u_i \hat u_i^\top \mathbf 1\\
&= \sum_{i=1}^N \frac{\hat u_i^\top \mathbf 1}{\hat\lambda_i}\,\hat u_i ,
\end{aligned}
$$

and each coefficient carries the factor $1/\hat\lambda_i$: the weights are most sensitive
to the smallest sample eigenvalues. These are also the least reliable: by points 1 and 3
they are biased down, to about 1 % of the truth at the lower edge for $N/T = 0.8$. Writing
$\hat\lambda_i = \lambda_i(1 + \varepsilon_i)$, the inverse is off by the factor
$1/(1 + \varepsilon_i)$, below 1 for an overestimate but exploding as $\varepsilon_i \to -1$,
so the bottom of the spectrum produces the largest errors in the weights.

The problem is statistical, not numerical: a stable solver computes these weights
accurately, but from a noisy $S$. A better solver does not help; a better estimate does.

The low-variance directions are spreads between similar assets, so loading on them means
large, offsetting long and short positions: the budget constraint fixes the net exposure
$\sum_i \hat w_i = 1$, but not the gross exposure $\sum_i |\hat w_i|$.

**5. Predicted risk is too low, realised risk too high.** Let $w^*$ be the true GMV
portfolio, computed from $\Sigma$ and therefore fixed, and $\hat w$ the one computed from
$S$:

$$
E\big[\hat w^\top S\, \hat w\big]
\;\le\; E\big[w^{*\top} S\, w^*\big]
\;=\; w^{*\top} \Sigma\, w^*
\;\le\; E\big[\hat w^\top \Sigma\, \hat w\big] .
$$

The two inequalities hold sample by sample: $\hat w$ minimises $w^\top S w$, and $w^*$
minimises $w^\top \Sigma w$. The equality uses $E[S] = \Sigma$, valid because $w^*$ does not
depend on the data. The optimiser fits the noise in $S$, as a model's training error falls
below its test error. **In-sample risk therefore cannot judge an estimator**; the
evaluation (Section 4) uses out-of-sample risk only.

### 3.4 Covariance estimators

**Sample covariance matrix.** Demeaned and divided by $T$ (scikit-learn's convention):

$$
S = \frac1T \sum_{t=1}^T (r_t - \bar r)(r_t - \bar r)^\top ,
$$

with $r_t$ the $N$-vector of returns on day $t$ of the window and $\bar r$ their mean. Its
rank is at most $T - 1$, so it is singular for $N \ge T$.

**Linear shrinkage** (Ledoit and Wolf, 2004). A convex combination of $S$ and a multiple of
the identity:

$$
S^* = \delta\, m I + (1 - \delta)\, S, \qquad m = \frac{\operatorname{tr} S}{N} .
$$

- **What it does.** It pulls every eigenvalue of $S$ towards their mean $m$, removing part
  of the excess dispersion of Section 3.3, and leaves the eigenvectors unchanged. For
  $\delta > 0$, $S^*$ is positive definite for any $N/T$, including $N \ge T$.
- **The intensity** $\delta \in [0, 1]$ is estimated from the data to minimise the expected
  loss $E\|S^* - \Sigma\|^2$.
- **Check.** The implementation matches scikit-learn's `LedoitWolf` to a relative
  tolerance of $10^{-10}$, including for $N > T$.

### 3.5 Backtest

| Setting         | Choice                                                  |
| --------------- | ------------------------------------------------------- |
| Windows         | $T$ = 125, 250, 500 days ($N/T$ = 0.8, 0.4, 0.2)        |
| Rebalancing     | every 21 trading days; buy-and-hold in between          |
| Out of sample   | 24 Dec 1974 to 31 Aug 2026: 13,029 days, 621 rebalances |
| Universe filter | drop a portfolio missing a return in window or holding  |

**Notes.**

- **Buy-and-hold.** Between rebalances the number of units held is fixed, so the weights
  drift with prices, as in Ledoit and Wolf (2017, footnote 15).
- **One out-of-sample period for all $T$.** Every window length is evaluated on the same
  days; the shorter windows take their estimation data from just before the first
  out-of-sample day. All return series are therefore paired.
- **The universe filter looks ahead** in its holding-period half, as in Ledoit and Wolf
  (2017, footnote 9). Lookahead biases a result only if the future information is correlated
  with the returns being measured, so its effect was measured. It removes a portfolio that
  is complete in the window at 9 of the 621 rebalances for $T = 125$ (6 for $T = 250$, 5 for
  $T = 500$), at most 3 portfolios at a time. All but one of these holding periods cross the
  end of June, so the filter only anticipates the re-sorting by a few days; the exception is
  the mid-year block of Section 2. Keeping these portfolios instead, with their value held
  constant once their data stops, changes no annualised SD in Section 5.1 by more than 0.01
  percentage points (`results/universe_filter.md`).

## 4. Evaluation

Each portfolio gives a series of daily out-of-sample excess returns, $n = 13{,}029$ days
long. To compare an estimator $a$ with a baseline $b$, write $a_t$ and $b_t$ for their
returns on day $t$, $\bar a$ and $\bar b$ for their means, and

$$
\hat\sigma_a^2 = \frac1n \sum_{t=1}^n (a_t - \bar a)^2
$$

(likewise $\hat\sigma_b^2$) for the sample variances; $\sigma_a^2$ and $\sigma_b^2$ are the
true ones.

### 4.1 Measures

As in Ledoit and Wolf (2017, p. 4368), annualised with 252 days:

| Measure  | Definition                        | Role                    |
| -------- | --------------------------------- | ----------------------- |
| AV       | mean daily excess return × 252    | reported                |
| SD       | standard deviation × $\sqrt{252}$ | **primary**, tested     |
| SR       | AV / SD                           | reported                |
| $\hat r$ | $\hat\sigma_a^2 / \hat\sigma_b^2$ | effect size, $a$ vs $b$ |

AV is the average (mean) excess return, SD the standard deviation of excess returns and SR
the Sharpe ratio, all annualised and in percent except SR.

**SD is the primary measure** because GMV minimises variance; AV and SR are reported but not
tested.

### 4.2 Test

**Null hypothesis:** equal out-of-sample variances, $\sigma_a^2 = \sigma_b^2$.

**The difference of variances is a mean.** With

$$
\hat d_t = (a_t - \bar a)^2 - (b_t - \bar b)^2, \qquad
\bar d = \frac1n \sum_{t=1}^n \hat d_t = \hat\sigma_a^2 - \hat\sigma_b^2 ,
$$

the test is a test on the mean of $\hat d_t$.

**The statistic** divides $\bar d$ by a Newey and West (1987) standard error:

$$
t = \frac{\bar d}{\sqrt{\hat\omega^2 / n}}, \qquad
\hat\omega^2 = \hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1 - \frac{k}{L+1}\Big)\hat\gamma_k ,
$$

where $\hat\gamma_k$ is the lag-$k$ sample autocovariance of $\hat d_t$, $L$ the lag length
(Section 4.3), and $\hat\omega^2$ the estimate of the long-run variance of $\hat d_t$. The
two-sided $p$-value is from the standard normal distribution.

**Why this standard error.** It handles the three features of daily portfolio returns:

- **Pairing.** $\hat d_t$ is formed day by day, so shocks common to both portfolios cancel.
- **Fat tails.** $\hat\gamma_0$ is the empirical variance of $\hat d_t$; normality is not
  assumed.
- **Volatility clustering.** The autocovariances $\hat\gamma_k$ account for it.

The classical $F$-test for equal variances assumes two independent samples of i.i.d. normal
observations, and each assumption fails here (Ledoit and Wolf, 2011): the two portfolios'
returns are highly correlated, up to 0.99; daily returns are fat-tailed, so the variance of
a sample variance is far larger than under normality; and volatility clusters, so the
observations are not independent over time.
Ledoit and Wolf (2011) test the log ratio with a prewhitened HAC estimator; with 13,029
days, the plain difference with Newey–West is the simpler choice.

### 4.3 Lag length

> **The default lag is too short.** $\lfloor 0.75\, n^{1/3} \rfloor = 17$ days ignores
> most of the autocorrelation of squared returns, which lasts for years.

$t$ roughly halves from $L = 17$ to $L = 252$ and keeps falling up to $L = 1000$. The tables
therefore report $t$ at $L = 17$ and at $L = 252$ (one year), and the conservative one is
relied on.

## 5. Results

### 5.1 First table: sample covariance, linear shrinkage, 1/N

| Estimation window $T$ (days)    |   125 |   250 |   500 |
| ------------------------------- | ----: | ----: | ----: |
| $N/T$                           |   0.8 |   0.4 |   0.2 |
| **Annualised SD (%)**           |       |       |       |
| 1/N (equal weights)             | 17.76 | 17.76 | 17.76 |
| GMV with $S$ (sample)           | 15.91 | 10.83 | 10.05 |
| GMV with $S^*$ (linear shrink.) |  9.96 |  9.82 |  9.77 |
| **$S^*$ against $S$**           |       |       |       |
| Variance ratio $\hat r$         | 0.392 | 0.822 | 0.946 |
| $t$, lag $L = 17$               | −18.6 |  −9.6 |  −5.9 |
| $t$, lag $L = 252$              |  −8.7 |  −4.0 |  −3.2 |

How to read it:

- **Columns:** the estimation window, $T$ trading days of returns before each rebalance
  (every 21 trading days), with $N = 100$ portfolios (fewer when the universe filter of
  Section 3.5 drops one). Every column is evaluated on the same 13,029 out-of-sample days,
  24 Dec 1974 to 31 Aug 2026.
- **Annualised SD (%):** the out-of-sample standard deviation of daily excess returns, times
  $\sqrt{252}$ (Section 4.1). Lower is better: it is what GMV tries to minimise.
  - *1/N* holds every portfolio with weight $1/N$. It estimates nothing, so it is the same
    in every column up to the universe filter, which depends on the window; it is the
    benchmark any estimator has to beat.
  - *GMV with $S$* is the minimum-variance portfolio of Section 3.2 computed with the sample
    covariance matrix $S$; *GMV with $S^*$*, with the linear-shrinkage estimate $S^*$
    (Section 3.4).
- **$S^*$ against $S$:** the two GMV portfolios compared directly.
  - *Variance ratio* $\hat r$: out-of-sample variance with $S^*$ over variance with $S$, the
    squared ratio of their SDs ($9.96^2 / 15.91^2 = 0.392$). Below 1: shrinkage is less
    risky; $0.392$ is a 61 % cut in variance.
  - *$t$*: the paired test of Section 4.2 of equal variances, with Newey–West lag $L$ = 17
    days (the default rule) or 252 days (one year, Section 4.3). Negative: $S^*$ has the
    lower variance. $|t| > 1.96$ is significant at 5 %; the $L = 252$ row is the one relied
    on.

The full table, with AV, SR and the tests of 1/N against the sample covariance, is in
`results/gmv_table.md` and `results/gmv_table_L252.md`.

Linear shrinkage gives the lowest out-of-sample volatility at every window length; its
advantage over the sample covariance is large when the sample covariance is noisy (variance
−61 % at N/T = 0.8), shrinks as the window grows (−5 % at N/T = 0.2), and stays significant
with a one-year Newey–West lag.

1/N is riskier than sample GMV at every window length, but only borderline significant at
$T = 125$: its variance is 25 % higher there ($\hat r = 1.25$), with $t = +2.0$ at a one-year
lag and $+1.6$ at $L = 1000$. At $T = 250$ and $500$ the gap is decisive (variance 2.7 and
3.1 times higher, $t = +5.2$ and $+5.4$ at $L = 252$).

Significance and size rank the comparisons differently. Linear shrinkage's 5 % gain over
the sample covariance at $T = 500$ has $t = -5.9$, the 25 % gap between 1/N and the sample
covariance at $T = 125$ only $t = +3.8$ (both at $L = 17$). Relative to the variance, the
second gap is 4.6 times larger, but its standard error is 7.2 times larger. Most of that
comes from the correlation: under normality the paired test's standard error scales with
$\sqrt{1 - \rho^2}$, which is 0.15 for the two GMV portfolios ($\rho = 0.99$; common shocks
cancel, so a small gap is measured precisely) and 0.97 for 1/N and sample GMV
($\rho = 0.25$), a factor 6.5. $t$ says whether a difference is distinguishable from zero,
not how large it is; the comparisons are ranked by the variance ratio.

**Shrinkage makes the window length almost irrelevant.** With linear shrinkage the SD is
about 10 % at every window (9.96 / 9.82 / 9.77); with the sample covariance it falls from
15.91 to 10.05 and needs a two-year window to get there. Shrinkage therefore needs far less
data for the same risk: with a 125-day window it reaches 9.96 %, below the sample
covariance's 10.05 % with 500 days (a difference not tested). At
$T = 500$ the gain itself is small: 0.28 pp of SD, or, for an investor who scales the
portfolio to a fixed volatility of 10.05 %, about 0.4 pp of return a year
($SR \times \Delta\sigma \approx 1.4 \times 0.28$; this needs leverage, and SR is not
tested). It is still worth taking: it costs nothing but a few lines of code, and it is
significant. Turnover is not measured; sample GMV's weights are more extreme and noisier, so
shrinkage would probably trade less, but that is not claimed here.

**AV and SR are descriptive only.** GMV has a higher SR than 1/N (0.63) with both
estimators at $T = 250$ and $500$ (sample 1.31 and 1.41, linear shrinkage 1.42 and 1.42)
and with linear shrinkage at $T = 125$ (1.20); sample GMV at $T = 125$ (0.73) is close to
1/N, about half a standard error apart (0.10 against a standard error of 0.19 for the
difference, under an i.i.d. normal approximation with the portfolios' correlation of 0.25).
Every GMV portfolio also has a higher AV than 1/N at the same window (11.59 to 14.20 %
against 11.18 to 11.24 %), with a lower SD. GMV uses no expected returns, so the higher AV
is not something the covariance estimator was built to deliver; it is consistent with the
low-volatility anomaly (Black, 1972; Frazzini and Pedersen, 2014), since a minimum-variance
portfolio tilts towards low-volatility assets (the weights are not analysed here). It would
then be a property of the minimum-variance rule, not of the estimator. Between the two
estimators at $T = 250$ and $500$, the sample covariance has the slightly higher AV
(14.19 against 13.90 %, 14.20 against 13.89 %); linear shrinkage's higher SR (1.42 against
1.31 and 1.41) comes only from its lower SD. Since AV and SR are not tested, no claim is
made that shrinkage, or GMV, improves returns or the Sharpe ratio.

## 6. Validation and limitations

### 6.1 How the code was checked

- Each estimator, the GMV weights, the backtest and the test were written against unit tests
  first: closed-form cases, simulated data with a known answer, edge cases (singular and
  indefinite matrices, a bankrupt portfolio) and a check for lookahead in the backtest.
- Linear shrinkage is pinned to scikit-learn's `LedoitWolf` on several shapes, including
  $N > T$. Matching an independent implementation is what catches a wrong normalisation in
  the intensity, which the clamp $\delta \le 1$ would otherwise hide.
- The variance test is checked against hand-computed standard errors at lags 0, 1, 2 and
  the largest lag, for its size under equal variances on simulated iid data, and for its
  correction of volatility clustering on simulated data.

### 6.2 Limitations

- **The Sharpe ratios overstate GMV's risk-adjusted performance.** At $T \ge 250$ GMV reaches an SR of
  1.31–1.42, against 0.63 for 1/N, for two reasons unrelated to the estimator. The
  universe is sorted on size and book-to-market, the dimensions that earned premia over
  this sample. And stale prices in the small-cap portfolios make their daily returns
  positively autocorrelated, so scaling the daily SD by $\sqrt{252}$ understates risk
  (Lo, 2002). GMV is drawn to whatever looks least risky, so it is more exposed to this
  than 1/N; its leverage (at $T = 125$, over the last 5,000 days, median gross exposure 18
  for the sample covariance, 7 for linear shrinkage) does not raise the SR by itself, but
  enlarges these positions and ignores the cost of shorting small stocks. Comparisons with
  1/N are therefore flattered. The variance ranking of the two estimators probably is not:
  the more leveraged sample-covariance GMV is plausibly the more exposed, which would bias
  the test against shrinkage. Not yet checked: the autocorrelation of each portfolio's
  daily returns, and the test on weekly or monthly returns.
- **Robustness to a few days is not checked.** Daily returns are heavy-tailed, and a few
  crisis days carry much of their fourth moment, on which the test's standard errors
  depend; whether the ranking survives without them, or in each sub-period, is not yet
  tested.
- **The standard errors depend on the lag.** $t$ still falls at $L = 1000$, and at that lag
  the 13,029 days contain only 13 non-overlapping 1000-day stretches, so the standard error
  is itself imprecise. The largest effects do not depend on this; the smallest ($T = 500$:
  variance ratio 0.946 in Section 5.1, $t = -3.2$ at $L = 252$) might.
- **One universe, of portfolios.** The 100 portfolios are already diversified and highly
  correlated, and $N/T \le 0.8$ keeps $S$ invertible. On stocks, with $N$ in the hundreds,
  $S$ is singular for these windows, so GMV with $S$ cannot be computed, and the size of
  the effect may differ. Ledoit and Wolf (2017) find that shrinkage lowers GMV volatility
  on stocks; this project does not test it (Section 7: a stock universe).
- **No costs, and a small lookahead.** Costs and turnover are not measured. They lower
  returns more than variance, and should hurt the more leveraged sample-covariance GMV
  most. The universe filter requires complete returns over each holding period, as in
  Ledoit and Wolf (2017). In French's data a missing value means an empty cell (Section 2),
  not a failed portfolio, so with one exception in 2001 the filter anticipates the June
  re-sorting, not returns (Section 3.5).

## 7. Next steps

- **Nonlinear shrinkage** (Ledoit and Wolf, 2020): validated first on simulated data with a
  known covariance matrix, reproducing their Table 2, then added to the table. From then on
  linear shrinkage is the baseline of the tests.
- **A stock universe** with $N > T$, where the sample covariance is singular. It needs
  survivorship-free stock data.
- **Uncertainty of the effect sizes:** confidence intervals for the variance ratios against
  linear shrinkage, from a paired block bootstrap.
- **Robustness of the test:** the variance test without the largest days, and by decade.
- **Hierarchical risk parity** (López de Prado, 2016) as a benchmark row: an allocation that
  never inverts the covariance matrix. HRP is long-only, so it is compared with a long-only
  GMV as well.
- **Diagnostics per rebalance:** the condition number and gross exposure of each portfolio,
  and its predicted against its realised variance.

## Reproducing

```
poetry install
poetry run pytest
poetry run python scripts/gmv_table.py              # results/gmv_table.{csv,md}
poetry run python scripts/gmv_table.py --lags 252   # results/gmv_table_L252.{csv,md}
poetry run python scripts/universe_filter_check.py  # results/universe_filter.md
make writeup                                        # build/WRITEUP.pdf
```

The French data are downloaded on the first run and cached in `data/`.

## References

- F. Black (1972), Capital market equilibrium with restricted borrowing, *Journal of
  Business* 45, 444–455.
- J. Bun, J.-P. Bouchaud, M. Potters (2017), Cleaning large correlation matrices: tools
  from random matrix theory, *Physics Reports* 666, 1–109; arXiv:1610.08104.
- V. DeMiguel, L. Garlappi, R. Uppal (2009), Optimal versus naive diversification: how
  inefficient is the 1/N portfolio strategy?, *Review of Financial Studies* 22, 1915–1953.
- A. Frazzini, L. H. Pedersen (2014), Betting against beta, *Journal of Financial
  Economics* 111, 1–25.
- K. R. French, Data Library: 100 portfolios formed on size and book-to-market, daily;
  Fama/French 3 factors, daily.
- O. Ledoit, M. Wolf (2004), A well-conditioned estimator for large-dimensional covariance
  matrices, *Journal of Multivariate Analysis* 88, 365–411.
- O. Ledoit, M. Wolf (2011), Robust performance hypothesis testing with the variance,
  *Wilmott*, September 2011, 86–89.
- O. Ledoit, M. Wolf (2017), Nonlinear shrinkage of the covariance matrix for portfolio
  selection: Markowitz meets Goldilocks, *Review of Financial Studies* 30, 4349–4388.
- O. Ledoit, M. Wolf (2020), Analytical nonlinear shrinkage of large-dimensional covariance
  matrices, *Annals of Statistics* 48, 3043–3065.
- A. W. Lo (2002), The statistics of Sharpe ratios, *Financial Analysts Journal* 58(4),
  36–52.
- M. López de Prado (2016), Building diversified portfolios that outperform out of sample,
  *Journal of Portfolio Management* 42(4), 59–69.
- V. A. Marchenko, L. A. Pastur (1967), Distribution of eigenvalues for some sets of random
  matrices, *Matematicheskii Sbornik* 114(4), 507–536.
- W. K. Newey, K. D. West (1987), A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix, *Econometrica* 55, 703–708.
