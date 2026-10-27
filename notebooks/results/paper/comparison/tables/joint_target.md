## Multi-objective control (5 seeds)

Target {'qed': 0.8, 'mw': 350.0, 'sa_score': 2.5}; mean |z-distance| over a pool of 512. Arms receive only the components their conditioning vector can express. Top-1 omitted: as a single extreme order statistic its seed-to-seed CV approaches 90% and it does not separate arms.

| model                | top10         | top50         |
|:---------------------|:--------------|:--------------|
| Stage 1 (QED-only)   | 0.137 ± 0.023 | 0.247 ± 0.012 |
| Stage 2 (9-property) | 0.072 ± 0.012 | 0.134 ± 0.009 |
| Stage 3 (RL)         | 0.079 ± 0.011 | 0.140 ± 0.005 |
| Stage 4 (filtered)   | 1.047 ± 0.074 | 1.304 ± 0.042 |
