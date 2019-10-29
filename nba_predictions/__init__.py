# nba predictions
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from game_predictions import game_predictions

# define function for scraping schedule and results
def scrape_schedule(year):
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
                       'home_points': list_home_score,
                       'away_points': list_away_score})
    # return the df
    return df

# define function for tuning hyperparameters
def tune_hyperparameters(df, list_outer_weighted_mean, list_distributions, list_inner_weighted_mean, list_weight_home, list_weight_away, train_size=.66, n_simulations=1000):
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

























# define function for nba season simulation
def nba_season_simulation(year, weighted_mean=False, n_simulations=1000):
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
                    # append to list_away_score
                    list_away_score.append(away_score)
                    # get home team
                    home_team = columns[3].text
                    # append to list_home_team
                    list_home_team.append(home_team)
                    # get home score
                    home_score = columns[4].text
                    # append to list_home_score
                    list_home_score.append(home_score)
            
    # put into df
    df = pd.DataFrame({'month': list_month,
                       'home_team': list_home_team,
                       'away_team': list_away_team,
                       'home_points': list_home_score,
                       'away_points': list_away_score})
    
    # replace '' with nan
    for column in list(df.columns):
        df[column] = df.apply(lambda x: np.nan if x[column] == '' else x[column], axis=1)
    
    # convert score cols to integer
    for column in list(df.columns)[3:]:
        df[column] = df.apply(lambda x: x[column] if pd.isnull(x[column]) else int(x[column]), axis=1)

    # get the played games
    df_played_games = df.dropna(subset=['home_points'])    
        
    # get the unplayed games
    df_unplayed_games = df[df.isnull().any(axis=1)]
    
    # instantiate lists
    list_home_score = []
    list_away_score = []
    list_home_win_prob = []
    for i in range(df_unplayed_games.shape[0]):
        # get home_team
        home_team = df_unplayed_games['home_team'].iloc[i]
        # get away_team
        away_team = df_unplayed_games['away_team'].iloc[i]
    
        # check to make sure each team is in the respective lists
        if home_team in list(df_played_games['home_team']) and away_team in list(df_played_games['away_team']):
            # simulate game
            simulated_game = game_predictions(home_team_array=df_played_games['home_team'], 
                                              home_score_array=df_played_games['home_points'], 
                                              away_team_array=df_played_games['away_team'], 
                                              away_score_array=df_played_games['away_points'], 
                                              home_team=home_team, 
                                              away_team=away_team,
                                              n_simulations=n_simulations,
                                              weighted_mean=weighted_mean)
            # get the predicted home score
            home_score = simulated_game.mean_home_score
            # get the predicted away score
            away_score = simulated_game.mean_away_score
            # get the predicted win probability
            home_win_prob = simulated_game.prop_home_win
        else:
            home_score = 'NA'
            away_score = 'NA'
            home_win_prob = 'NA'
        # append to lists
        list_home_score.append(home_score)
        list_away_score.append(away_score)
        list_home_win_prob.append(home_win_prob)
    
    # put into df_unplayed_games
    df_unplayed_games['home_points'] = list_home_score
    df_unplayed_games['away_points'] = list_away_score
    df_unplayed_games['home_win_prob'] = list_home_win_prob

    # choose the winning team
    df_unplayed_games['winning_team'] = df_unplayed_games.apply(lambda x: x['home_team'] if x['home_points'] >= x['away_points'] else x['away_team'], axis=1)

    # make columns in df_played_games match with df_unplayed_games
    df_played_games['home_win_prob'] = df_played_games.apply(lambda x: 1.0 if x['home_points'] > x['away_points'] else 0.0, axis=1)
    df_played_games['winning_team'] = df_played_games.apply(lambda x: x['home_team'] if x['home_points'] >= x['away_points'] else x['away_team'], axis=1)

    # append df_unplayed_games to df_played_games
    df_simulated_season = df_played_games.append(df_unplayed_games)

    # get number of wins for each team

    # get unique teams
    df_unique_teams = pd.DataFrame(pd.unique(df['home_team']))

    # get the wins for each team
    list_unique_winning_teams = list(pd.value_counts(df_simulated_season['winning_team']).index)
    list_n_wins = list(pd.value_counts(df_simulated_season['winning_team']))

    # put into a df
    df_predicted_wins = pd.DataFrame({'team': list_unique_winning_teams,
                                      'wins': list_n_wins})

    # left join df_unique_teams and df_predicted_wins
    df_final_win_predictions = pd.merge(left=df_unique_teams, right=df_predicted_wins,
                                        left_on=df_unique_teams[0], right_on='team',
                                        how='left').fillna(0)
    
    # drop the col we dont want
    df_final_win_predictions.drop([0], axis=1, inplace=True)    

    # get predicted losses
    df_final_win_predictions['losses'] = 82 - df_final_win_predictions['wins']

    # sort by wins
    df_final_win_predictions = df_final_win_predictions.sort_values(by=['wins'], ascending=False)
    
    # get the conference for each team
    df_nba_teams_conferences = pd.read_csv('https://raw.githubusercontent.com/aaronengland/data/master/nba_teams_conferences.csv')
    
    # left join df_final_win_predictions and df_nba_teams_conferences on team name
    df_final_win_predictions_conf = pd.merge(left=df_final_win_predictions, right=df_nba_teams_conferences, 
                                             left_on='team', right_on='Team',
                                             how='left')

    # drop the cols we don't need
    df_final_win_predictions_conf.drop(['Unnamed: 0', 'ID', 'Team'], axis=1, inplace=True)

    # separate into east and west
    df_east = df_final_win_predictions_conf[df_final_win_predictions_conf['Conference'] == 'East']
    df_west = df_final_win_predictions_conf[df_final_win_predictions_conf['Conference'] == 'West']
    
    # define attributes class
    class attributes:
        def __init__(self, df_simulated_season, df_final_win_predictions_conf, df_east, df_west):
            self.df_simulated_season = df_simulated_season
            self.df_final_win_predictions_conf = df_final_win_predictions_conf
            self.df_east = df_east
            self.df_west = df_west
    # save returnable object
    x = attributes(df_simulated_season, df_final_win_predictions_conf, df_east, df_west)
    return x





























    
