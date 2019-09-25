# nba_predictions

A package for scraping the user-defined NFL season's schedule/results and simulating the upcoming week's games (`nfl_pickem`) and the unplayed games for the entire season (`nfl_season_simulation`) using the `game_predictions` algorithm.

To install, use: `pip install git+https://github.com/aaronengland/nba_predictions.git`

## nba_pickem

Arguments:
- `year`: season to simulate.
- `weighted_mean`: use of weighted mean in simulation (boolean; default=False; False is recommended for early in the season while True is recommended for later games).
- `n_simulations`: number of simulations for each game (default=1000).

Returns a data frame with predicted results for the upcoming month's games.

Example:

```
from nba_predictions import nba_pickem

# simulate upcoming month
upcoming_month_simulation = nfl_pickem(year=2019, weighted_mean=False, n_simulations=1000)

# view results
upcoming_month_simulation
```

---

## nba_season_simulation

Arguments:
- `year`: season to simulate.
- `weighted_mean`: use of weighted mean in simulation (boolean; default=False; False is recommended for early in the season while True is recommended for later games).
- `n_simulations`: number of simulations for each game (default=1000).

Attributes:
- `df_simulated_season`: data frame of all played and simulated games in season.
- `df_final_win_predictions_conf`: data frame of predicted wins.
- `df_east`: data frame of predicted wins (Eastern conference only).
- `df_west`: data frame of predicted wins (Western conference only).

Example:

```
from nba_predictions import nba_season_simulation

# simulate season
simulated_season = nba_season_simulation(year=2019, weighted_mean=False, n_simulations=1000)

# get simulated season
df_entire_season = simulated_season.df_simulated_season

# get final win predictions
df_standings = simulated_season.df_final_win_predictions_conf

# get eastern conference
df_east_standings = simulated_season.df_east

# get western conference
df_west_standings = simulated_season.df_west
```

---
