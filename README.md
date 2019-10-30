# nba_predictions

A package for scraping the user-defined NBA season's schedule/results from [here](https://www.basketball-reference.com/leagues/NBA_2020_games.html), tuning the model's hyperparameters on a user-defined proportion of the completed contests (`tune_hyperparameters`), simulating the entire season (`nba_season_simulation`), and generating postseason probabilities (`nba_postseason_probabilities`) using the [`game_predictions`](https://github.com/aaronengland/game_predictions/blob/master/README.md) algorithm.

To install, use: `pip install git+https://github.com/aaronengland/nba_predictions.git`

---

Example:

```
from nba_predictions import scrape_schedule, tune_hyperparameters, nba_season_simulation, nba_postseason_probabilities

# get simulations
n_simulations=1000

# get schedule and results
df = scrape_schedule(year=2020)

# tune hyperparameters
hyperparams_tuned = tune_hyperparameters(df=df, 
                                         list_outer_weighted_mean = ['all_games_weighted','none','time','opp_win_pct'],
                                         list_distributions = ['poisson','normal'],
                                         list_inner_weighted_mean = ['none','win_pct'],
                                         list_weight_home = [1,2,3,4,5,6,7,8,9,10],
                                         list_weight_away = [1,2,3,4,5,6,7,8,9,10],
                                         train_size = .9,
                                         n_simulations=n_simulations)

# get the best hyperparameters
dict_best_hyperparameters = hyperparams_tuned.get('dict_best_hyperparameters')

# simulate season
season_simulation = nba_season_simulation(df=df,
                                          dict_best_hyperparameters=dict_best_hyperparameters,
                                          n_simulations=n_simulations)

# get the final win totals
win_totals = season_simulation.get('final_win_predictions_w_conf')

# define function for postseason probabilities
postseason_prob = nba_postseason_probabilities(df=df, 
                                               dict_best_hyperparameters=dict_best_hyperparameters,
                                               n_simulations=n_simulations)
```
