#import dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#load data - lebron, jordan, nba
lebron_url = 'https://www.basketball-reference.com/players/j/jamesle01.html#all_per_game'
lebron = pd.read_html(lebron_url, header=0)
lebron = lebron[0]
jordan_url = 'https://www.basketball-reference.com/players/j/jordami01.html#all_per_game'
jordan = pd.read_html(jordan_url, header=0)
jordan = jordan[0]
nba_url = 'https://www.basketball-reference.com/leagues/NBA_stats_per_game.html'
nba = pd.read_html(nba_url, header=1)
nba = nba[0]

#cleaning - drop NaN from DF
lebron = lebron.dropna()
jordan = jordan.dropna()
jordan = jordan[jordan.Lg.str.contains("NBA")]
nba = nba.dropna()
nba = nba[nba.Lg.str.contains("NBA")]

#add year number to df - lebron, jordan
lebron['Yr'] = np.arange(1,len(lebron)+1)
jordan['Yr'] = np.arange(1,len(jordan)+1)

#convert data types - lebron
lebron['Age'] = lebron.Age.astype(int)
lebron['Tm'] = lebron.Tm.astype(object)
lebron['Lg'] = lebron.Lg.astype(object)
lebron['Pos'] = lebron.Tm.astype(object)
lebron['G'] = lebron.G.astype(int)
lebron['GS'] = lebron.GS.astype(int)
lebron['MP'] = lebron.MP.astype(float)
lebron['FG'] = lebron.FG.astype(float)
lebron['FGA'] = lebron.FGA.astype(float)
lebron['FG%'] = lebron['FG%'].astype(float)
lebron['3P'] = lebron['3P'].astype(float)
lebron['3PA'] = lebron['3PA'].astype(float)
lebron['2P'] = lebron['2P'].astype(float)
lebron['2PA'] = lebron['2PA'].astype(float)
lebron['2P%'] = lebron['2P%'].astype(float)
lebron['eFG%'] = lebron['eFG%'].astype(float)
lebron['FT'] = lebron.FT.astype(float)
lebron['FTA'] = lebron.FTA.astype(float)
lebron['FT%'] = lebron['FT%'].astype(float)
lebron['ORB'] = lebron.ORB.astype(float)
lebron['DRB'] = lebron.DRB.astype(float)
lebron['TRB'] = lebron.TRB.astype(float)
lebron['AST'] = lebron.AST.astype(float)
lebron['STL'] = lebron.STL.astype(float)
lebron['BLK'] = lebron.BLK.astype(float)
lebron['TOV'] = lebron.TOV.astype(float)
lebron['PF'] = lebron.PF.astype(float)
lebron['PTS'] = lebron.PTS.astype(float)
#convert data types - jordan
jordan['Age'] = jordan.Age.astype(int)
jordan['Tm'] = jordan.Tm.astype(object)
jordan['Lg'] = jordan.Lg.astype(object)
jordan['Pos'] = jordan.Tm.astype(object)
jordan['G'] = jordan.G.astype(int)
jordan['GS'] = jordan.GS.astype(int)
jordan['MP'] = jordan.MP.astype(float)
jordan['FG'] = jordan.FG.astype(float)
jordan['FGA'] = jordan.FGA.astype(float)
jordan['FG%'] = jordan['FG%'].astype(float)
jordan['3P'] = jordan['3P'].astype(float)
jordan['3PA'] = jordan['3PA'].astype(float)
jordan['2P'] = jordan['2P'].astype(float)
jordan['2PA'] = jordan['2PA'].astype(float)
jordan['2P%'] = jordan['2P%'].astype(float)
jordan['eFG%'] = jordan['eFG%'].astype(float)
jordan['FT'] = jordan.FT.astype(float)
jordan['FTA'] = jordan.FTA.astype(float)
jordan['FT%'] = jordan['FT%'].astype(float)
jordan['ORB'] = jordan.ORB.astype(float)
jordan['DRB'] = jordan.DRB.astype(float)
jordan['TRB'] = jordan.TRB.astype(float)
jordan['AST'] = jordan.AST.astype(float)
jordan['STL'] = jordan.STL.astype(float)
jordan['BLK'] = jordan.BLK.astype(float)
jordan['TOV'] = jordan.TOV.astype(float)
jordan['PF'] = jordan.PF.astype(float)
jordan['PTS'] = jordan.PTS.astype(float)
#convert data types - nba
nba['Age'] = nba.Age.astype(float)
nba['Lg'] = nba.Lg.astype(object)
nba['G'] = nba.G.astype(int)
nba['MP'] = nba.MP.astype(float)
nba['FG'] = nba.FG.astype(float)
nba['FGA'] = nba.FGA.astype(float)
nba['FG%'] = nba['FG%'].astype(float)
nba['3P'] = nba['3P'].astype(float)
nba['3PA'] = nba['3PA'].astype(float)
nba['3P%'] = nba['3P%'].astype(float)
nba['FG%'] = nba['FG%'].astype(float)
nba['FT'] = nba.FT.astype(float)
nba['FTA'] = nba.FTA.astype(float)
nba['FT%'] = nba['FT%'].astype(float)
nba['ORB'] = nba.ORB.astype(float)
nba['DRB'] = nba.DRB.astype(float)
nba['TRB'] = nba.TRB.astype(float)
nba['AST'] = nba.AST.astype(float)
nba['STL'] = nba.STL.astype(float)
nba['BLK'] = nba.BLK.astype(float)
nba['TOV'] = nba.TOV.astype(float)
nba['PF'] = nba.PF.astype(float)
nba['PTS'] = nba.PTS.astype(float)
nba['Pace'] = nba.Pace.astype(float)

#Fix Season Column
lebron['Season'] = lebron['Season'].str.split('-').str[0]
jordan['Season'] = jordan['Season'].str.split('-').str[0]
nba['Season'] = nba['Season'].str.split('-').str[0]
lebron['Season'] = lebron.Season.astype(int)
jordan['Season'] = jordan.Season.astype(int)
nba['Season'] = nba.Season.astype(int)

plt.style.use('fivethirtyeight')
plt.plot(lebron.Yr, lebron.PTS, label ="Lebron")
plt.plot(jordan.Yr, jordan.PTS, label= "Jordan")
plt.title('PTS by Season')
plt.xlabel('Season')
plt.ylabel('PTS')
plt.legend()
plt.xticks(range(1, 18,2))
plt.yticks(range(20, 40,5))
plt.show()

plt.style.use('fivethirtyeight')
plt.plot(lebron.Yr, lebron.AST, label ="Lebron")
plt.plot(jordan.Yr, jordan.AST, label= "Jordan")
plt.title('AST by Season')
plt.xlabel('Season')
plt.ylabel('AST')
plt.legend()
plt.xticks(range(1, 18,2))
plt.yticks(range(2, 12,2))
plt.show()

plt.style.use('fivethirtyeight')
plt.plot(lebron.Yr, lebron.TRB, label ="Lebron")
plt.plot(jordan.Yr, jordan.TRB, label= "Jordan")
plt.title('TRB by Season')
plt.xlabel('Season')
plt.ylabel('TRB')
plt.legend()
plt.xticks(range(1, 18,2))
plt.yticks(range(2, 12,2))
plt.show()

plt.style.use('fivethirtyeight')
plt.plot(lebron.Yr, lebron.MP, label ="Lebron")
plt.plot(jordan.Yr, jordan.MP, label= "Jordan")
plt.title('MP by Season')
plt.xlabel('Season')
plt.ylabel('MP')
plt.legend()
plt.xticks(range(1, 18,2))
plt.yticks(range(20, 50,5))
plt.show()

plt.style.use('fivethirtyeight')
plt.plot(nba.Season, nba.Pace, label="Pace")
#plt.plot(nba.Season, nba.Age, label="Age")
plt.title('Pace by Season')
plt.xlabel('Season')
plt.ylabel('Pace')
plt.legend()
#plt.xticks(range(0, 50,5))
#plt.yticks(range(85, 110,5))
plt.show()

#relationship between Pace of NBA and Age
plt.style.use('fivethirtyeight')
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(nba.Season, nba.Pace, 'g-')
ax2.plot(nba.Season, nba.Age, 'b-')

ax1.set_xlabel('Season')
ax1.set_ylabel('Pace', color='g')
ax2.set_ylabel('Age', color='b')
plt.show()

# calculate the Pearson's correlation between two variables
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr, _ = pearsonr(nba.Pace, nba.Age)
print('Pearsons correlation: %.3f' % corr)

#relationship between Pace of NBA and Age
plt.style.use('fivethirtyeight')
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(nba.Season, nba['3PA'], 'g-')
ax2.plot(nba.Season, nba['3P%'], 'b-')

ax1.set_xlabel('Season')
ax1.set_ylabel('3PA', color='g')
ax2.set_ylabel('3P%', color='b')
plt.show()

#relationship between Pace of NBA and Age
plt.style.use('fivethirtyeight')
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(nba.Season, nba['3PA'], 'g-')
ax2.plot(nba.Season, nba['FTA'], 'b-')

ax1.set_xlabel('Season')
ax1.set_ylabel('3PA', color='g')
ax2.set_ylabel('FTA', color='b')
plt.show()

# calculate the Pearson's correlation between two variables
from scipy.stats import pearsonr
# calculate Pearson's correlation
corr, _ = pearsonr(nba['3PA'], nba['FTA'])
print('Pearsons correlation: %.3f' % corr)

plt.style.use('fivethirtyeight')
plt.plot(lebron.Season, lebron['FG%'].rolling(2).mean(), label ="Lebron")
plt.plot(jordan.Season, jordan['FG%'].rolling(2).mean(), label ="Jordan")
plt.plot(nba.Season, nba['FG%'], label= "NBA")
plt.title('FG% by Season')
plt.xlabel('Season')
plt.ylabel('FG%')
plt.legend()
#plt.xticks(range(1980, 2025,10))
#plt.yticks(range(20, 40,5))
plt.show()

plt.style.use('fivethirtyeight')
plt.plot(lebron.Yr, lebron['FG%'].rolling(2).mean(), label ="Lebron")
plt.plot(jordan.Yr, jordan['FG%'].rolling(2).mean(), label ="Jordan")
plt.title('FG% by Season')
plt.xlabel('Season')
plt.ylabel('FG%')
plt.legend()
#plt.xticks(range(1980, 2025,10))
#plt.yticks(range(20, 40,5))
plt.show()

#histogram
import seaborn as sns
sns.displot(lebron.PTS,kde=False)

#scatter plots - correlations
sns.pairplot(lebron[["PTS","AST","TRB"]])

#heat map - correlations
correlation = lebron[["PTS","AST","TRB"]].corr()
sns.heatmap(correlation, annot=True)
