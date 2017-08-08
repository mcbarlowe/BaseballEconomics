#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:45:32 2017

@author: MattBarlowe
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
from scipy.stats.stats import pearsonr
#creating function to return dataframes with the last five years of the data
def get_years(dataframe):
    year_list = [2016, 2015, 2014, 2013, 2012]
    dataframe = dataframe[dataframe['yearID'].isin(year_list)]
    return dataframe

def team_cost_per_metric(dataframe, metrics):
    '''
    takes in dataframe and calculates the cost of each teams
    metric per dollar
    '''
    for x in metrics:
        if x != 'Bat_Avg':
            dataframe['$_per_%s' % str(x)]=round(dataframe['salary']/
                      dataframe[x],2)
        else:
            dataframe['$_per_%s' % str(x)]= round(dataframe['salary']
                .sum()/(dataframe[x]*1000), 2)
    
    return dataframe

def difference_from_league_mean_standardized(dataframe, league_avg_dict, stats):
    '''
    takes in the league averages and league values per statistic and
    standardizes them to graph later
    '''
    for x in stats:
        dataframe['%s_diff_from_league_avg' % (x)] = dataframe['$_per_%s' % (x)]-league_avg_dict[x]
    
    for x in stats:
        dataframe['%s_diff_from_league_avg' % str(x)]= dataframe['%s_diff_from_league_avg' \
                  % str(x)]/dataframe['%s_diff_from_league_avg' % str(x)].std()    
    
    return dataframe

def plot_player_metrics(dataframe, metrics):
    '''
    function to plot each player salary vs offensive production metric
    '''
    for x in metrics:
        sns.lmplot(x = 'salary', y=x, data=dataframe, hue = 'POS', scatter=True,
                  palette='bright', fit_reg=True)
        plot.set_title('Salary vs %s' % (x))
        plot.set_xlabel('Salary in $10 million units')
        
#sets position and offensive metric lists that will be commonly used throughout
#the program
fielders = ['OF', '1B', '2B', '3B', 'SS', 'C']
metrics = ['HR','R', 'H', 'SB','RBI', 'Bat_Avg']

#importing baseball stats csv files for salaries, batting
#fielding, and player names into dataframes
salary_df = pd.read_csv('/Users/MattBarlowe/baseballdatabank-2017.1/core/salaries.csv')
batting_df = pd.read_csv('/Users/MattBarlowe/baseballdatabank-2017.1/core/Batting.csv')
fielding_df = pd.read_csv('/Users/MattBarlowe/baseballdatabank-2017.1/core/Fielding.csv')
names_df = pd.read_csv('/Users/MattBarlowe/baseballdatabank-2017.1/core/Master.csv')
team_df = pd.read_csv('/Users/MattBarlowe/baseballdatabank-2017.1/core/Teams.csv')
#passing all dataframes to get_years function to create dataframes that only 
#have the last five years data except for names_df which does not have a yearID
salary_df = get_years(salary_df)
batting_df = get_years(batting_df)
fielding_df = get_years(fielding_df)
team_df = get_years(team_df)
#create a batting average statistic on the batting df because it is missing
batting_df['Bat_Avg'] = round((batting_df['H']/batting_df['AB']), 3)

#group the fielding csv by player and year and then check to see if each row
#is the max for that player and year.  This is to determine the position of 
#each player as the position the player played most games at in a year.  This
#is because many players can player many different positions 
#but are often evaluated at their preferred or main position
fielding_grouped = fielding_df.groupby(['playerID','yearID'])['G'].transform(max)\
        == fielding_df['G']
        
fielding_df = fielding_df[fielding_grouped]

#merging batting statistics with salary data did an inner merge that way any 
#players without salary data will be dropped from the dataframe
master_df = batting_df.merge(salary_df)

#mergine batting/salary dataframe with name_df to get players first and last names
master_df = master_df.merge(names_df[['playerID', 'nameLast', 'nameFirst']], 
                                on='playerID')

#rearranging columns to tidy up the data and put players names first in 
#dataframe
columns = master_df.columns.tolist()
columns = columns[-2:] + columns[:-2] 
master_df = master_df[columns]

#final merging of fielding statistics to the other stats to create the master
#df contiaing player, year, batting stats, position, and salary
master_df = master_df.merge(fielding_df[['playerID', 'yearID','POS']])

#removing pitchers from dataframe as they rarely or never hit and their numbers
#will skew the data
master_df = master_df[master_df['POS'].isin(fielders)]

#removing players who have less than 130 At Bats again to remove any non-regular
#hitters who might skew the data.  PIcked 130 ABs because that is the number
#considered to be the threshhold to count as rookie's first season
master_df = master_df[master_df['AB']>=130]



#determining the mean of cost per statistic for the whole league for standard
#deviation calculations later
league_values_per = {}
for x in ['HR', 'SB', 'RBI', 'H', 'Bat_Avg', 'R']:
    if x != 'Bat_Avg':
        league_values_per[x]=round((master_df['salary']
            .sum()/master_df[x].sum()), 2)
    else:
        league_values_per[x]= round(master_df['salary']
            .sum()/(master_df[x].sum()*1000), 2)
    
salary_hist = master_df['salary'].plot.hist(edgecolor='k')
salary_hist.set_title('Frequency of Player Salaries in units of $10 Million')
#plot position vs 
plot_player_metrics(master_df, metrics)
#create dataframe grouped by team
master_team_group = master_df.groupby('teamID')['HR','R', 'H', 'SB','RBI', 'Bat_Avg', 'salary'].sum()
master_team_group = team_cost_per_metric(master_team_group, metrics)
master_team_group = difference_from_league_mean_standardized(master_team_group, league_values_per, metrics)

#sums up team wins in year range and then adds a win column to the master dataframe
team_df = team_df.groupby('teamID')['W'].sum()

for x, y in zip(master_team_group.index, team_df):
    master_team_group.loc[x,'W'] = y
#averages all the teams std dev of metrics to plot against wins 
master_team_group['STD_metrics_avg_by_team']= master_team_group.iloc[:,13:18].mean(axis=1)



#creates the bar plot of each teams metrics 
plot = master_team_group[['HR_diff_from_league_avg','R_diff_from_league_avg', 
        'H_diff_from_league_avg', 'SB_diff_from_league_avg', 'RBI_diff_from_league_avg'
        ]].plot(kind='bar', title ="Cost Per Stat", figsize=(12, 5), 
        legend=True, fontsize=12, width=.7)




plot.set_title('Difference from league average Spending in Standard Deviations')

#plot.set_yticklabels([])
plot.set_xlabel('Teams')
legend = plt.legend()
legend.get_texts()[0].set_text('Home Runs')
legend.get_texts()[1].set_text('Runs')
legend.get_texts()[2].set_text('Hits')
legend.get_texts()[3].set_text('Stolen Bases')
legend.get_texts()[4].set_text('RBI')

pearson_r = pearsonr(master_team_group['STD_metrics_avg_by_team'], 
               master_team_group['W'])
pearson_list=[]
for x in pearson_r:
    pearson_list.append(x)
for x in pearson_list:
    print(round(x, 3))
#plot teams spending below or above average versus to wins to see if overspending
#makes a better team
plot2 = sns.lmplot(x='STD_metrics_avg_by_team', y='W', data=master_team_group)
plt.subplots_adjust(top=0.9)
plot2.fig.suptitle('Spending on Offensive Metrics in Standard Deviations versus\
 wins for years 2012-2016')
plot2.set_ylabels('Wins')
plot2.set_xlabels('Average Spending in Standard Deviations')