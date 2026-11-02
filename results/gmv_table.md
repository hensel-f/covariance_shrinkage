| T   | N/T | Estimator        | AV (%) | SD (%) | SR   | Var / var(sample) | t (SD vs sample, L = 17) |
| --- | --- | ---------------- | ------ | ------ | ---- | ----------------- | ------------------------ |
| 125 | 0.8 | 1/N              | 11.18  | 17.76  | 0.63 | 1.246             | +3.75                    |
| 125 | 0.8 | sample           | 11.59  | 15.91  | 0.73 |                   |                          |
| 125 | 0.8 | linear_shrinkage | 11.93  | 9.96   | 1.20 | 0.392             | -18.63                   |
| 250 | 0.4 | 1/N              | 11.20  | 17.76  | 0.63 | 2.688             | +11.10                   |
| 250 | 0.4 | sample           | 14.19  | 10.83  | 1.31 |                   |                          |
| 250 | 0.4 | linear_shrinkage | 13.90  | 9.82   | 1.42 | 0.822             | -9.62                    |
| 500 | 0.2 | 1/N              | 11.24  | 17.76  | 0.63 | 3.124             | +11.84                   |
| 500 | 0.2 | sample           | 14.20  | 10.05  | 1.41 |                   |                          |
| 500 | 0.2 | linear_shrinkage | 13.89  | 9.77   | 1.42 | 0.946             | -5.94                    |

Data 1973-01-02 to 2026-08-31, N = 100; out of sample 1974-12-24 to 2026-08-31 (13029 days), rebalanced every 21 days.
Daily returns in excess of the risk-free rate, annualised with 252 days. AV: mean x 252; SD: sd x sqrt(252); SR: AV / SD. Var / var(sample): out-of-sample variance relative to sample (effect size). t: paired variance-difference test against sample, Newey-West with lag L = 17; t < 0 means lower out-of-sample variance than sample.
