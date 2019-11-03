# nba predictions
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
from game_predictions import game_predictions

# define function for scraping schedule and results
def scrape_nba_schedule(year):
    # create list of momths
    list_months = ['october','november','december','jauary','february','march','april','may','june']
    # instantiate lists
    list_month = []
    list_away_team = []
    list_away_score = []
    list_home_team = []
    list_home_score = []
    # get rows for each momth
    for month in list_months:
        # get url
        r = requests.get('https://www.basketball-reference.com/leagues/NBA_{0}_games-{1}.html'.format(year, month))
        # get content of page
        soup = BeautifulSoup(r.content, 'html.parser')
        # get the schedule table
        table = soup.find('table', id='schedule')
        if table:
            # get all table rows
            table_rows = table.find_all('tr')
            # get columns
            for row in table_rows:
                # get columns (i.e., td)
                columns = row.find_all('td')
                # if columns is not an empty list
                if columns:
                    # append month to list_month
                    list_month.append(month)
                    # get away team
                    away_team = columns[1].text
                    # append to list_away_team
                    list_away_team.append(away_team)
                    # get away_score
                    away_score = columns[2].text
                    # for empty strings
                    if away_score == '':
                        away_score = np.nan
                    else:
                        away_score = int(away_score)
                    # append to list_away_score
                    list_away_score.append(away_score)
                    # get home team
                    home_team = columns[3].text
                    # append to list_home_team
                    list_home_team.append(home_team)
                    # get home score
                    home_score = columns[4].text
                    # for empty strings
                    if home_score == '':
                        home_score = np.nan
                    else:
                        home_score = int(home_score)
                    # append to list_home_score
                    list_home_score.append(home_score)
            
    # put into df
    df = pd.DataFrame({'month': list_month,
                       'home_team': list_home_team,
                       'away_team': list_away_team,
                       'home_score': list_home_score,
                       'away_score': list_away_score})
    
    # define function for winning team
    def get_winner(home_team, away_team, home_score, away_score):
        if home_score > away_score:
            return home_team
        elif home_score < away_score:
            return away_team
        else:
            return np.nan
    
    # create winning team col
    df['winning_team'] = df.apply(lambda x: get_winner(home_team=x['home_team'], away_team=x['away_team'], home_score=x['home_score'], away_score=x['away_score']), axis=1)
    
    # return the df
    return df

# define function for tuning hyperparameters
def tune_nba_hyperparameters(df, list_outer_weighted_mean, list_distributions, list_inner_weighted_mean, list_weight_home, list_weight_away, train_size=.66, n_simulations=1000):
    # suppress the SettingWithCopyWarning
    pd.options.mode.chained_assignment = None
    
    # drop the unplayed games
    df_played = df.dropna(subset=['home_points'])
    
    # get winning team
    df_played['winning_team'] = df_played.apply(lambda x: x['home_team'] if x['home_points'] > x['away_points'] else x['away_team'], axis=1)
    
    # calculate spread
    df_played['spread'] = df_played['home_points'] - df_played['away_points']
    
    # get the number of rows in train size
    nrows = df_played.shape[0]
    # multiply by train_size
    n_train = round(train_size * nrows)
    
    # get the training data
    df_train = df_played[:n_train]
    # get the testing data
    df_test = df_played[n_train:]
    
    # time to tune model
    time_start = datetime.datetime.now()
    # instantiate empty list
    list_sum_correct = []
    list_sum_spread_error = []
    list_sum_error = []
    list_dict_outcomes = []
    for outer_weighted_mean in list_outer_weighted_mean:
        # if all_games_weighted
        if outer_weighted_mean == 'all_games_weighted':
            for distribution in list_distributions:
                for weight_home in list_weight_home:
                    for weight_away in list_weight_away:
                        # we only want equal weights when both equal 1
                        if (weight_home + weight_away == 2) or (weight_home != weight_away):
                            for inner_weighted_mean in list_inner_weighted_mean:
                                # predict every game in df_test
                                df_test['pred_outcome'] = df_test.apply(lambda x: game_predictions(home_team_array=df_train['home_team'], 
                                                                                                   home_score_array=df_train['home_points'], 
                                                                                                   away_team_array=df_train['away_team'], 
                                                                                                   away_score_array=df_train['away_points'], 
                                                                                                   home_team=x['home_team'], 
                                                                                                   away_team=x['away_team'], 
                                                                                                   outer_weighted_mean=outer_weighted_mean, 
                                                                                                   inner_weighted_mean=inner_weighted_mean, 
                                                                                                   weight_home=weight_home,
                                                                                                   weight_away=weight_away,
                                                                                                   n_simulations=n_simulations), axis=1)
                    
                                # get winning team
                                df_test['pred_winning_team'] = df_test.apply(lambda x: (x['pred_outcome']).get('winning_team'), axis=1)
                                # get number right
                                sum_correct = np.sum(df_test.apply(lambda x: 1 if x['winning_team'] == x['pred_winning_team'] else 0, axis=1))
                                # append to list
                                list_sum_correct.append(sum_correct)
                                    
                                # get the total spread difference so we can sort by that as well
                                # get predicted home points
                                df_test['pred_home_points'] = df_test.apply(lambda x: (x['pred_outcome']).get('mean_home_pts'), axis=1)
                                # get predicted away points
                                df_test['pred_away_points'] = df_test.apply(lambda x: (x['pred_outcome']).get('mean_away_pts'), axis=1)
                                # get predicted spread
                                df_test['pred_spread'] = df_test['pred_home_points'] - df_test['pred_away_points']
                                # get spread error
                                df_test['pred_spread_error'] = df_test.apply(lambda x: np.abs(x['spread'] - x['pred_spread']), axis=1)
                                # get the total spread error
                                sum_spread_error = np.sum(df_test['pred_spread_error'])
                                # append to list
                                list_sum_spread_error.append(sum_spread_error)
                                    
                                # get absolute difference between home_points and pred_home_points
                                df_test['pred_home_points_error'] = df_test.apply(lambda x: np.abs(x['home_points'] - x['pred_home_points']), axis=1)
                                # get absolute difference between away_points and pred_away_points
                                df_test['pred_away_points_error'] = df_test.apply(lambda x: np.abs(x['away_points'] - x['pred_away_points']), axis=1)
                                # sum pred_home_points_error and pred_away_points_error
                                sum_error = np.sum(df_test['pred_home_points_error']) + np.sum(df_test['pred_away_points_error'])
                                # append to list
                                list_sum_error.append(sum_error)
                                    
                                # create dictionary
                                dict_outcomes = {'outer_weighted_mean': outer_weighted_mean,
                                                 'distribution': distribution,
                                                 'weight_home': weight_home,
                                                 'weight_away': weight_away,
                                                 'inner_weighted_mean': inner_weighted_mean}
                                # append to list
                                list_dict_outcomes.append(dict_outcomes)
        # else (i.e., outer_weighted_mean != 'all_games_weighted')
        else:
            for distribution in list_distributions:
                # save weight home and weight away for the dictionary
                weight_home = None
                weight_away = None
                for inner_weighted_mean in list_inner_weighted_mean:
                    # predict every game in df_predictions
                    df_test['pred_outcome'] = df_test.apply(lambda x: game_predictions(home_team_array=df_train['home_team'], 
                                                                                       home_score_array=df_train['home_points'], 
                                                                                       away_team_array=df_train['away_team'], 
                                                                                       away_score_array=df_train['away_points'], 
                                                                                       home_team=x['home_team'], 
                                                                                       away_team=x['away_team'], 
                                                                                       outer_weighted_mean=outer_weighted_mean, 
                                                                                       inner_weighted_mean=inner_weighted_mean, 
                                                                                       n_simulations=n_simulations), axis=1)
                
                    # get winning team
                    df_test['pred_winning_team'] = df_test.apply(lambda x: (x['pred_outcome']).get('winning_team'), axis=1)
                    # get number right
                    sum_correct = np.sum(df_test.apply(lambda x: 1 if x['winning_team'] == x['pred_winning_team'] else 0, axis=1))
                    # append to list
                    list_sum_correct.append(sum_correct)
                        
                    # get the total spread difference so we can sort by that as well
                    # get predicted home points
                    df_test['pred_home_points'] = df_test.apply(lambda x: (x['pred_outcome']).get('mean_home_pts'), axis=1)
                    # get predicted away points
                    df_test['pred_away_points'] = df_test.apply(lambda x: (x['pred_outcome']).get('mean_away_pts'), axis=1)
                    # get predicted spread
                    df_test['pred_spread'] = df_test['pred_home_points'] - df_test['pred_away_points']
                    # get spread error
                    df_test['pred_spread_error'] = df_test.apply(lambda x: np.abs(x['spread'] - x['pred_spread']), axis=1)
                    # get the total spread error
                    sum_spread_error = np.sum(df_test['pred_spread_error'])
                    # append to list
                    list_sum_spread_error.append(sum_spread_error)
                    
                    # get absolute difference between home_points and pred_home_points
                    df_test['pred_home_points_error'] = df_test.apply(lambda x: np.abs(x['home_points'] - x['pred_home_points']), axis=1)
                    # get absolute difference between away_points and pred_away_points
                    df_test['pred_away_points_error'] = df_test.apply(lambda x: np.abs(x['away_points'] - x['pred_away_points']), axis=1)
                    # sum pred_home_points_error and pred_away_points_error
                    sum_error = np.sum(df_test['pred_home_points_error']) + np.sum(df_test['pred_away_points_error'])
                    # append to list
                    list_sum_error.append(sum_error)
                        
                    # create dictionary
                    dict_outcomes = {'outer_weighted_mean': outer_weighted_mean,
                                     'distribution': distribution,
                                     'weight_home': weight_home,
                                     'weight_away': weight_away,
                                     'inner_weighted_mean': inner_weighted_mean}
                    # append to list
                    list_dict_outcomes.append(dict_outcomes)
    # get elapsed time
    elapsed_time = (datetime.datetime.now() - time_start).seconds
    # print message
    print('Time to tune the model: {0} min'.format(elapsed_time/60))
        
    # put outcome lists into a df
    df_outcomes = pd.DataFrame({'hyperparameters': list_dict_outcomes,
                                'n_correct': list_sum_correct,
                                'spread_error': list_sum_spread_error,
                                'tot_pts_error': list_sum_error})
    
    # sort values descending
    df_outcomes_sorted = df_outcomes.sort_values(by=['n_correct','spread_error','tot_pts_error'], ascending=[False, True, True])
        
    # get the best set of hyperparameters
    dict_best_hyperparameters = df_outcomes_sorted['hyperparameters'].iloc[0]
        
    # make a dictionary with output
    dict_results = {'df_outcomes_sorted': df_outcomes_sorted,
                    'dict_best_hyperparameters': dict_best_hyperparameters}
    
    # return dict_results
    return dict_results

# simulate season
def simulate_nba_season(df, dict_best_hyperparameters, n_simulations=1000):
    # get winning team
    df['winning_team'] = df.apply(lambda x: x['home_team'] if x['home_points'] > x['away_points'] else x['away_team'], axis=1)  
        
    # drop the unplayed games
    df_played = df.dropna(subset=['home_points'])
    # get the unplayed games
    df_unplayed = df[pd.isnull(df['home_points'])]
    
    # simulate each game with best hyperparameters
    df_unplayed['pred_outcome'] = df_unplayed.apply(lambda x: game_predictions(home_team_array=df_played['home_team'], 
                                                                               home_score_array=df_played['home_points'], 
                                                                               away_team_array=df_played['away_team'], 
                                                                               away_score_array=df_played['away_points'], 
                                                                               home_team=x['home_team'], 
                                                                               away_team=x['away_team'], 
                                                                               outer_weighted_mean=dict_best_hyperparameters.get('outer_weighted_mean'), 
                                                                               distribution=dict_best_hyperparameters.get('distribution'),
                                                                               inner_weighted_mean=dict_best_hyperparameters.get('inner_weighted_mean'), 
                                                                               weight_home=dict_best_hyperparameters.get('weight_home'),
                                                                               weight_away=dict_best_hyperparameters.get('weight_away'),
                                                                               n_simulations=n_simulations), axis=1)
    
    # put into df_unplayed
    df_unplayed['home_points'] = df_unplayed.apply(lambda x: x['pred_outcome'].get('mean_home_pts'), axis=1)
    df_unplayed['away_points'] = df_unplayed.apply(lambda x: x['pred_outcome'].get('mean_away_pts'), axis=1)
    df_unplayed['winning_team'] = df_unplayed.apply(lambda x: x['pred_outcome'].get('winning_team'), axis=1)
    
    # drop pred_outcome
    df_unplayed.drop(['pred_outcome'], axis=1, inplace=True)
    
    # append df_played and df_unplayed
    df_final = pd.concat([df_played, df_unplayed],ignore_index=True)
    
    # get number of wins for each team
    df_unique_teams = pd.DataFrame({'team': pd.unique(df_final['home_team'])})
    
    # define function to get n_wins
    def n_wins(team):
        num_wins = 0
        for winning_team in df_final['winning_team']:
            if winning_team == team:
                num_wins += 1
        return num_wins
    
    # apply function for each team
    df_unique_teams['n_wins'] = df_unique_teams.apply(lambda x: n_wins(team=x['team']), axis=1)
    
    # sort by n_wins
    df_unique_teams = df_unique_teams.sort_values(by=['n_wins'], ascending=False)
    
    # get losses
    df_unique_teams['n_losses'] = 82 - df_unique_teams['n_wins']
    
    # get each teams conference
    df_nba_teams_conferences = pd.read_csv('https://raw.githubusercontent.com/aaronengland/data/master/nba_teams_conferences.csv')
    
    # join df_unique_teams and df_nba_teams_conferences
    df_unique_teams_w_conf = pd.merge(left=df_unique_teams, right=df_nba_teams_conferences,
                                      left_on='team', right_on='Team', how='left')
    # drop Team col
    df_unique_teams_w_conf.drop(['Team'], axis=1, inplace=True)
    
    # get west
    df_west = df_unique_teams_w_conf[df_unique_teams_w_conf['Conference'] == 'West'].reset_index(drop=True)
    # get east
    df_east = df_unique_teams_w_conf[df_unique_teams_w_conf['Conference'] == 'East'].reset_index(drop=True)
    
    # create dictionary for which to return
    results = {'simulated_season': df_final,
               'final_win_predictions_w_conf': df_unique_teams_w_conf,
               'west': df_west,
               'east': df_east}
    
    # return results
    return results

# postseason probabilities
def nba_postseason_probabilities(df, dict_best_hyperparameters, n_simulations=1000):
    # get df with all teams
    df_unique_teams = pd.DataFrame({'team': pd.unique(df['home_team'])})
    
    # loop through n_simulations
    for i in range(n_simulations):
        # simulate season
        season_simulation = nba_season_simulation(df=df,
                                                  dict_best_hyperparameters=dict_best_hyperparameters,
                                                  n_simulations=1)
        # get top 8 teams in western conference
        list_playoffs_west = season_simulation.get('west')[:8]['team'].to_list()
        # get top 8 teams in eastern conference
        list_playoffs_east = season_simulation.get('east')[:8]['team'].to_list()
        # combine both lists
        list_playoffs_nba = list_playoffs_west + list_playoffs_east
        
        # mark each row as 1 if in playoffs_nba
        df_unique_teams['sim_{}'.format(i)] = df_unique_teams.apply(lambda x: 1 if x['team'] in list_playoffs_nba else 0, axis=1)
    
    # get mean across rows
    df_postseason_prob = pd.DataFrame({'team': df_unique_teams['team'],
                                       'prob_postseason': df_unique_teams.mean(axis=1)})
    
    # sort descending
    df_final = df_postseason_prob.sort_values(by=['prob_postseason'], ascending=False)
    
    # return df_final
    return df_final




























    
