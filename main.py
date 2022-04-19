import folium as folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastf1 as ff1
from fastf1 import plotting
import seaborn as sns
sns.set_style("darkgrid")
import folium
from folium.plugins import MarkerCluster
import webbrowser
from bokeh.palettes import Category20b
import matplotlib as mpl
from matplotlib.collections import LineCollection


def circuitsMap():
    circuits = pd.read_csv('data/circuits.csv')
    circuits_map = folium.Map(zoom_start=13)
    map_cluster = MarkerCluster().add_to(circuits_map)
    for idx, row in circuits.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            tooltip=row['name'],
            icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')
        ).add_to(map_cluster)

    circuits_map.save('mymap.html')
    webbrowser.open('C:/Users/helle/formulaOneStatistics/mymap.html')

def topHostCircuits():
    circuits = pd.read_csv('data/circuits.csv')
    races = pd.read_csv('data/races.csv')
    host_circuits = races.drop(['date', 'time', 'url', 'round'], axis = 1).rename(columns = {'name':'gp_name'}).merge(
        circuits.drop(['lat', 'lng', 'alt', 'url'], 1).rename(columns = {'name':'cir_name'}), how = 'left')
    top_hosts = host_circuits[['cir_name']].cir_name.value_counts().reset_index().rename(
        columns={'index': 'name', 'cir_name': 'counts'}).head(20)

    sns.barplot(top_hosts.counts, top_hosts.name,color='#FF1801')
    plt.title('Top 10 F1 host circuits')
    plt.xlabel('Number of races hosted')
    plt.ylabel('')
    plt.show()

def driverNationalities():
    drivers = pd.read_csv('data/drivers.csv')
    driversCountry = drivers[['driverId', 'nationality']].groupby('nationality').count().rename(
        columns = {'driverId' : 'counts'})
    driversCountry = driversCountry[driversCountry.counts > 22].sort_values('counts', ascending = False)
    driversCountry.loc['Others'] = [(len(drivers) - driversCountry.counts.sum())]

    plt.axis('equal')
    plt.title('Driver Nationalities')
    plt.pie(driversCountry.counts, labels=driversCountry.index, autopct='%1.1f%%',
            shadow=True, startangle=270)
    plt.show()

def constructorWorldTitles():
    constructorStand = pd.read_csv('data/constructorStandings.csv')
    constructors = pd.read_csv('data/constructors.csv')
    races = pd.read_csv('data/races.csv')

    racesHistory = races[races.date <= '2020-09-27']
    idx = racesHistory.groupby(['year'])['date'].transform(max) == racesHistory['date']
    totalRacesForSeason = racesHistory[idx].rename(columns = {'round' : 'tot_races'})

    standingSeasonEnd = totalRacesForSeason[['raceId', 'year', 'tot_races']].merge(constructorStand, on = 'raceId')
    champForEachYear = standingSeasonEnd[standingSeasonEnd['position'] == 1]
    champTotalPErTeam = champForEachYear[['constructorId', 'position']].groupby('constructorId').sum().reset_index().merge(
        constructors[['name', 'constructorId']]).rename(columns={'position':'titles'}).sort_values('titles', ascending = False) # Total titles per constructor

    sns.barplot(champTotalPErTeam.titles, champTotalPErTeam.name,color='#FF1801')
    plt.title('Constructor championship titles')
    plt.xlabel('Number of wins')
    plt.ylabel('')
    plt.show()

def racesPerTeam():
    results = pd.read_csv('data/results.csv')
    constructors = pd.read_csv('data/constructors.csv')

    groupTeamRace = results.groupby(['constructorId', 'raceId']).size()
    raceEnterForTeam = groupTeamRace.groupby('constructorId').count().reset_index().rename(columns = {0:'races_ent'})
    racesEntered = raceEnterForTeam.merge(constructors, on = 'constructorId')
    top30 = racesEntered.sort_values('races_ent',ascending = False).head(30)

    sns.barplot(top30.races_ent, top30.name,color='#FF1801')
    plt.title('Top 30 F1 constructor enteries')
    plt.xlabel('Number of races entered')
    plt.ylabel('')
    plt.show()

def racesPerSeason():
    races = pd.read_csv('data/races.csv').sort_values('date')
    idxLastRace = races.groupby(['year'])['date'].transform(max) == races['date']
    season_finale = races[idxLastRace].rename(columns = {'round' : 'tot_races'})

    plt.plot(season_finale.year, season_finale.tot_races, 's-b',color='#FF1801')
    plt.xlabel('Year')
    plt.ylabel('Number of races')
    plt.title("Races per seasons")
    plt.show()

def speedVisualization():
    year = input("Enter the year: ")
    year = int(year)
    wknd = input("Weekend number: ")
    wknd = int(wknd)
    driver = input("Which driver, use shortname(ex: VER): ")
    ses = 'R'

    colormap = mpl.cm.plasma

    session = ff1.get_session(year, wknd, ses)
    weekend = session.event
    session.load()
    lap = session.laps.pick_driver(driver).pick_fastest()

    x = lap.telemetry['X']
    y = lap.telemetry['Y']
    color = lap.telemetry['Speed']
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
    fig.suptitle(f'{weekend.name} {year}', size=24, y=0.97)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis('off')

    ax.plot(lap.telemetry['X'], lap.telemetry['Y'], color='black', linestyle='-', linewidth=16, zorder=0)

    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)

    lc.set_array(color)
    line = ax.add_collection(lc)

    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal")

    plt.show()

def fastestEachSeason():
    races = pd.read_csv('data/races.csv')
    laptimes = pd.read_csv('data/lapTimes.csv')
    drivers = pd.read_csv('data/drivers.csv')
    circuits = pd.read_csv('data/circuits.csv')

    fastest_data = pd.merge(laptimes, races, on='raceId', how='left')
    fastest_data = fastest_data[['raceId', 'driverId', 'time_x', 'milliseconds','year', 'round', 'circuitId', 'name', 'date']]
    fastest_data.rename(columns={'time_x':'lap_time', 'name':'circuit_name'}, inplace=True)
    fastest_data = pd.merge(fastest_data, drivers, on='driverId', how='left')
    fastest_data = pd.merge(fastest_data, circuits, on='circuitId', how='left')

    fastest_data = fastest_data[['raceId', 'driverId', 'lap_time', 'milliseconds', 'year', 'round',
                                 'circuitId', 'circuit_name', 'date', 'driverRef', 'number', 'code',
                                 'forename', 'surname', 'dob', 'nationality', 'circuitRef', 'location', 'country']]

    data = pd.merge(fastest_data.groupby(['circuit_name','date']).lap_time.min().to_frame().reset_index(), fastest_data[['circuit_name','date','lap_time', 'driverRef','code']], on=['circuit_name','date','lap_time'], how='left')
    data = data.sort_values(by='date', ascending = False)

    data['year'] = pd.DatetimeIndex(data.date).year
    data['counts'] = 1
    data = data.groupby(['year', 'code', 'driverRef']).counts.count().to_frame().reset_index().sort_values(by='year', ascending=False)

    fastest = pd.merge(data, data.groupby(['year'])['counts'].max().to_frame(name='max').reset_index(), on='year', how='left')
    fastest = fastest[fastest['counts'] == fastest['max']][['year','code','driverRef','counts']]
    fastest.driverRef = fastest.driverRef.str.capitalize()

    fastest = pd.merge(fastest, fastest_data.groupby('year')['round'].max().reset_index(), on='year', how='left')
    fastest['percent'] = np.array(fastest['counts'])/np.array(fastest['round'])*100
    fastest['year'] = fastest['year'].astype(str)

    fig, ax = plt.subplots(figsize=(12,16))
    fig.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    ax.hlines(fastest.year, xmin=0, xmax=fastest.percent, linestyle='dotted')

    groups = fastest[['year','percent','driverRef']].groupby('driverRef')
    colors=sns.color_palette("magma", len(fastest.code.unique()))

    for (name, group), color in zip(groups, colors):
        ax.plot(group.percent, group.year, marker='o', color=color, linestyle='', ms=12, label=name)
    ax.set_xlim(0,65)
    ax.legend()

    for x,y, label, count in zip(fastest.percent, fastest.year, fastest.code, fastest.counts):
        ax.annotate(label+'({} races)'.format(int(count)), xy=(x+0.8,y), textcoords='data')

    plt.xlabel('Percentage of Fastest Lap Wins(%)')
    plt.title('Who is the fastest driver in each season?', fontsize=18)

    plt.show()


def barQualiFinal():
    drivers = pd.read_csv("./data/drivers.csv")
    results = pd.read_csv("./data/results.csv")
    results.dropna(inplace=True)
    merge = pd.merge(results, drivers, how="inner", on="driverId")
    merge = merge.drop(['points','position', 'laps', 'url', 'nationality', 'dob', 'forename', 'code', 'number_y'], axis = 1)
    status = pd.read_csv("./data/status.csv")
    d1 = pd.merge(merge, status, how="inner", on='statusId')
    data = d1.drop(['driverId', 'statusId', 'driverRef'], axis = 1)
    races = pd.read_csv("./data/races.csv")
    races = races.drop(['round', 'circuitId', 'date', 'time', 'url'], axis = 1)
    cleanedData = pd.merge(data, races, how='inner', on='raceId')
    constructors = pd.read_csv("./data/constructors.csv")
    constructors = constructors.drop(['constructorRef', 'nationality', 'url', 'Unnamed: 5'], axis = 1)
    d2 = pd.merge(cleanedData, constructors, how='inner', on = 'constructorId')
    FinalData = d2.drop(['constructorId','raceId','number_x', 'positionText', 'positionText', 'milliseconds', 'time', 'rank'], axis = 1)

    First = []
    for i in range(1,11):
        First += [FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == i)].shape[0]]

    Second =[]
    for i in range(1,11):
        Second += [FinalData[(FinalData['grid'] == 2) & (FinalData['positionOrder'] == i)].shape[0]]

    Third = []
    for i in range(1,11):
        Third += [FinalData[(FinalData['grid'] == 3) & (FinalData['positionOrder'] == i)].shape[0]]

    Fourth =[]
    for i in range(1,11):
        Fourth += [FinalData[(FinalData['grid'] == 4) & (FinalData['positionOrder'] == i)].shape[0]]

    Fifth = []
    for i in range(1,11):
        Fifth += [FinalData[(FinalData['grid'] == 5) & (FinalData['positionOrder'] == i)].shape[0]]

    barPlot = pd.DataFrame({
        'First': First,
        'Second': Second,
        'Third':Third,
        'Fourth':Fourth,
        'Fifth': Fifth

    }, index = ['1','2','3','4','5','6','7','8','9','10'])

    barPlot.plot.bar(title='Driver Final Position Based on Qualifying', figsize=(15,7), grid = True)
    plt.xlabel('Final position')
    plt.ylabel('Number of Drivers')
    plt.show()

    #todo: dessa måste skrivas ut, vad är bäst?
    nonepoints = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] >= 10)].shape[0]/230) * 100
    points = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] <= 10)].shape[0]/230) * 100
    podium = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] <= 3)].shape[0]/230) * 100


def frequency():
    results = pd.read_csv("./data/results.csv")
    qualifying = pd.read_csv("./data/qualifying.csv")
    final = results.merge(qualifying,on=['raceId','driverId'],suffixes=['_result','_qual'])

    winprec = final[final.position_qual == 1].groupby('position_result',as_index=False).agg(count=pd.NamedAgg(column='resultId',aggfunc='nunique'))
    winprec['perc'] = winprec['count']/winprec.sum()['count']

    plt.figure(figsize=(15,5))
    graph = sns.barplot(
        data=winprec,
        y ='perc',
        x='position_result',
        color='#FF1801'
    ).set_title(
        'Race result from pole position',
        size=14
    )

    plt.xlabel('Final Race Result')
    plt.ylabel('Frequency(%)')

    plt.show()

def functionSwitch(choise):
    switcher={
        1: circuitsMap,
        2: topHostCircuits,
        3: driverNationalities,
        4: constructorWorldTitles,
        5: racesPerTeam,
        6: racesPerSeason,
        7: speedVisualization,
        8: fastestEachSeason,
        9: frequency,
        10: barQualiFinal
    }

    func = switcher.get(int(choise),lambda :"nothing")
    print(func())

if __name__ == "__main__":
    print("Welcome ti F1 statistics!")
    while True:
        print("1. World map with circuits")
        print("2. Top host circuits")
        print("3. Driver nationalities")
        print("4. Constructor world titles")
        print("5. Entered races per team")
        print("6. Number of races per season")
        print("7. Speed visualization")
        print("8. Fastest driver each season")
        print("9. Race result frequency from pole")
        print("10. Quali(1-5) vs race result")

        choise = input("Choose one topic you want to see (Q to quit):")
        if choise == "Q":
            break
        else:
            functionSwitch(choise)