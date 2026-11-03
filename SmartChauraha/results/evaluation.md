Average wait per vehicle in seconds (lower is better). 20 seeds x 60 min per time of day.

| Controller | Peak | Normal | Night | Worst single wait (peak) |
|---|---|---|---|---|
| Fixed-time (20 s round robin) | 168.2 | 46.5 | 34.6 | 399 |
| Original SmartChauraha rule | 55.5 | 26.8 | **18.6** | 310 |
| Weighted priority + clearance | 57.6 | 27.1 | 19.9 | 397 |
| RL - tabular Q-learning | 59.0 | 33.5 | 26.7 | 508 |
| RL - linear Q-learning | 47.8 | 27.6 | 21.6 | 189 |
| RL - Deep Q-Network | **44.8** | **25.6** | 18.9 | 244 |
