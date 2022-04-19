import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    #TODO: SKRIVA UT DESSA ELLER HUR VISA DOM???

    firstfirst = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 1)].shape[0]/230) * 100
    firstsecond = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 2)].shape[0]/230) * 100
    firstthird = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 3)].shape[0]/230) * 100
    firstfourth = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 4)].shape[0]/230) * 100
    firstfifth = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 5)].shape[0]/230) * 100
    firstsixth = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 6)].shape[0]/230) * 100
    firstseventh = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 7)].shape[0]/230) * 100
    firsteight = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 3)].shape[0]/230) * 100
    firstnine = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 3)].shape[0]/230) * 100
    firstten = (FinalData[(FinalData['grid'] == 1) & (FinalData['positionOrder'] == 3)].shape[0]/230) * 100

    #todo: dessa måste skrivas ut
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


if __name__ == "__main__":
    frequency()
