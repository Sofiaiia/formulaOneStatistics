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

    sns.barplot(top_hosts.counts, top_hosts.name)
    plt.title('Top 10 F1 host circuits')
    plt.xlabel('Number of races hosted')
    plt.ylabel('')
    plt.show()


if __name__ == "__main__":
    topHostCircuits()
