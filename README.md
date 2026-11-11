# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A forecast of Roblox's Up-and-Coming top 10 for make.fun's bounty. It learns from every pair of past chart reads where a game at a given rank, rank momentum, player growth and age ended up some hours later (a kernel-weighted nearest-neighbour model), and gives every game on the chart a probability at each of the 10 places, leaving each place's newcomer share unnamed.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`node main.ts --backtest` scores the model against past cutoffs of the history (each cutoff forecasts the next 72 hours), next to the house's persistence baseline; `--tune` grid-searches the kernel widths. On the practice history: model −38.77, baseline −46.02 (mean log score over 14 cutoffs).

Node 22, standard library only, no network, deterministic.
