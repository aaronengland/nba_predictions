# nba predictions

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from game_predictions import game_predictions

# define function for scraping nfl schedule/results
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


# define function for upcoming month predictions
def nba_pickem(year, weighted_mean=False, n_simulations=1000):
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
    
    # get games for upcoming month
    upcoming_month = np.min(df[df.isnull().any(axis=1)]['month'])
    
    # get the matchups for the upcoming month
    df_upcoming_month = df[df['month'] == upcoming_month]
        
    # drop rows with missing values
    df = df.dropna(subset=['home_points'])
    
    # instantiate lists
    list_home_score = []
    list_away_score = []
    list_home_win_prob = []
    for i in range(df_upcoming_month.shape[0]):
        # get home_team
        home_team = df_upcoming_month['home_team'].iloc[i]
        # get away_team
        away_team = df_upcoming_month['away_team'].iloc[i]
        # check to make sure each team is in the respective lists
        if home_team in list(df['home_team']) and away_team in list(df['away_team']):
            # simulate game
            simulated_game = game_predictions(home_team_array=df['home_team'], 
                                              home_score_array=df['home_points'], 
                                              away_team_array=df['away_team'], 
                                              away_score_array=df['away_points'], 
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
            
    # put into df
    df_upcoming_month['home_points'] = list_home_score
    df_upcoming_month['away_points'] = list_away_score
    df_upcoming_month['home_win_prob'] = list_home_win_prob
        
    # choose the winning team
    df_upcoming_month['winning_team'] = df_upcoming_month.apply(lambda x: x['home_team'] if x['home_points'] >= x['away_points'] else x['away_team'], axis=1)
        
    # return df_upcoming_month
    return df_upcoming_month


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





























    
