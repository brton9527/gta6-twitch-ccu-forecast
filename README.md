# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/brton9527/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/brton9527/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), after the first. `main.py` (Python 3.13, standard library only) reads the case file and writes five quantiles (5%, 25%, 50%, 75%, 95%) for hours 2-24.

The model is built from the launch-day shapes of past big launches: per hour it fits a quadratic of the log viewer ratio to hour 1 on log10(hour-1 viewers) across the case's comparables, corrects for the game's own pre-launch hype (hour 1 over its pre-launch 48-hour median), and applies a per-hour bias fitted on the four practice launches (Wolverine, Red Dead Redemption 2, God of War Ragnarok, Spider-Man 2). On those practice launches it scores a pinball loss of 0.142 leave-one-case-out (house baseline 0.387), with a typical miss of 15-31% and 90% coverage of 0.78-1.00. See `main.py` for the full design and the local validation numbers.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=83pyrp7s&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
