import folium as folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastf1 as ff1
from fastf1 import plotting
import seaborn as sns
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
            icon=folium.Icon(color='cadetblue', prefix='fa', icon='flag-checkered',icon_color='white')
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

    year = 2021
    wknd = 14
    ses = 'R'
    driver = 'VER'
    colormap = mpl.cm.plasma

    session = ff1.get_session(year, wknd, ses)
    weekend = session.event
    session.load()
    lap = session.laps.pick_driver(driver).pick_fastest()

    # Get telemetry data
    x = lap.telemetry['X']              # values for x-axis
    y = lap.telemetry['Y']              # values for y-axis
    color = lap.telemetry['Speed']      # value to base color gradient on

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # We create a plot with title and adjust some setting to make it look good.
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
    fig.suptitle(f'{weekend.name} {year}', size=24, y=0.97)

    # Adjust margins and turn of axis
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
    ax.axis('off')

    # Create background track line
    ax.plot(lap.telemetry['X'], lap.telemetry['Y'], color='black', linestyle='-', linewidth=16, zorder=0)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)

    # Set the values used for colormapping
    lc.set_array(color)

    # Merge all line segments together
    line = ax.add_collection(lc)


    # Finally, we create a color bar as a legend.
    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal")


    # Show the plot
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

    # fastest = data.loc[data.groupby(['year'])['occ'].idxmax()]
    fastest = pd.merge(data, data.groupby(['year'])['counts'].max().to_frame(name='max').reset_index(), on='year', how='left')
    fastest = fastest[fastest['counts'] == fastest['max']][['year','code','driverRef','counts']]
    fastest.driverRef = fastest.driverRef.str.capitalize()

    # Calculate the percentage of fastest lap per season
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
        ax.annotate(label+'({} races)'.format(count), xy=(x+0.8,y), textcoords='data')
        #ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

    plt.xlabel('Percentage of Fastest Lap Wins(%)')
    plt.title('Who is the fastest driver in each season?', fontsize=18)

    plt.show()

if __name__ == "__main__":
    racesPerSeason()
