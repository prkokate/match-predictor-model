from flask import Flask, request 
import pandas as pd
import json 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS, cross_origin
import requests
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import re
from io import StringIO
from datetime import datetime, timedelta



# Setup flask server 
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



rf=RandomForestClassifier(n_estimators=30,min_samples_split=15,random_state=137)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton",
              "Manchester Utd": "Manchester United",
              "Newcastle Utd": "Newcastle United", 
              "Tottenham Hotspur": "Tottenham", 
              "West Ham United": "West Ham", 
              "Wolverhampton Wanderers": "Wolves",
              "Wolverhampton Wanderer": "Wolves",
              "Nott'ham Forest":"Nottingham",
              "Nottingham Fore":"Nottingham",
              "heffield United":"Sheffield United",
              "Sheffield Utd":"Sheffield United",
              "Chelse":"Chelsea",
              "Asron Vill":"Aston Villa"
             } 
mapping = MissingDict(**map_values)

# Setup url route which will calculate 
# total sum of array. 
def rolling_averages(group, cols, new_cols):
    group=group.sort_values("date")
    rolling_stats=group[cols].rolling(3,closed='left').mean()
    group[new_cols]=rolling_stats
    group=group.dropna(subset=new_cols)
    return group


def make_one_prediction(predictors,test_data):
    preds=rf.predict(test_data[predictors])
    # combined=pd.DataFrame(dict(actual=test_data["target"],predicted=preds), index=test.index)
    # precision=precision_score(test["target"],preds)
    return combined

def TrainModel(data):
    matches=pd.read_csv('matches.csv',index_col=0)
    
    matches=matches.groupby("team").get_group(data["team"])
    matches["target"]=matches["result"].astype("category").cat.codes
    new_row=pd.DataFrame(data,index=[0])
    matches = pd.concat([matches,new_row], ignore_index = True)
    
    matches["date"]=pd.to_datetime(matches["date"])
    matches["opp_code"]=matches["opponent"].astype("category").cat.codes
    matches["hour"]=matches["time"].str.replace(":.+","",regex=True).astype("int")
    matches["formation"].ffill(inplace=True)
    matches["formation_code"]=matches["formation"].str.replace(r'\D',"",regex=True).astype("int")
    matches["day_code"]=matches["date"].dt.dayofweek
    matches["venue_code"]=matches["venue"].astype("category").cat.codes
    matches["target"]=(matches["result"]=="W").astype("int")
    cols=["gf","ga","sh","sot","dist","fk","pk","pkatt"]
    
    predictors=["venue_code","opp_code","hour","day_code",'formation_code']
    
    new_cols=[f"{c}_rolling" for c in cols]
    
    
    
    matches_rolling=rolling_averages(matches,cols,new_cols)
    special=matches_rolling.groupby("opponent").get_group(data["opponent"])
    if(special.shape[0]>=2):
        matches_rolling=special
    matches_rolling.index=[i for i in range(matches_rolling.shape[0])]
    matches_rolling.sort_index(axis=0, ascending=True, inplace=False, kind='quicksort')
    last_index=matches_rolling.index.sort_values()[-1]
    # print("LATS INDEX=",matches_rolling.index[-1])
    
    prev_matches=matches_rolling.loc[:matches_rolling.shape[0]-2]
    
    train=prev_matches
    rf.fit(train[predictors+new_cols],train['target'])
    return rf, matches_rolling.loc[matches_rolling.shape[0]-1:]


@app.route('/', methods = ['GET']) 
@cross_origin()
def started_server():
    return "Server Started!" 

@app.route('/predict', methods = ['POST']) 
@cross_origin()
def predict_match(): 

        #Request Body Data:
        data = request.get_json()


        model,test_data=TrainModel(data)

        cols=["gf","ga","sh","sot","dist","fk","pk","pkatt"]
        new_cols=[f"{c}_rolling" for c in cols]
        predictors=["venue_code","opp_code","hour","day_code",'formation_code']


        result=model.predict(test_data[predictors+new_cols]).astype(int).tolist()[0]

        data2=data
        team=data["team"]
        data2["team"]=data["opponent"]
        data2["opponent"]=team
        data2["formation"]=np.nan
        if(data["venue"]=="Home"):
            data2["venue"]="Away"
        else:
            data2["venue"]="Home"


        model2,test_data2=TrainModel(data2)

        result2=model2.predict(test_data2[predictors+new_cols]).astype(int).tolist()[0]

        finalresult=result
        if(result==result2):
            finalresult=0
        elif(result==2.0 and result==1.0):
            finalresult=2
        elif(result==1.0 and result2==2.0):
            finalresult=1
        elif(result==0.0 and result2==1.0):
            finalresult=2
        elif(result==0.0 and result2==2.0):
            finalresult=1



        print("result=",finalresult)


        # Return data in json format 
        return json.dumps({"result":finalresult}) 

@app.route('/schedule', methods = ['GET']) 
@cross_origin()
def get_schedule():
    #requestBody = request.get_json()
    standings_url="https://fbref.com/en/comps/9/Premier-League-Stats"
    headers = {
    'authority': 'www.google.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    # Add more headers as needed
    }

    data=requests.get(standings_url, headers=headers)
    datastr=str(data)
    x=re.findall("[0-9][0-9][0-9]",datastr)
    if(x[0]!="200"):
        print("Failure!", x[0])
        return
    soup=BeautifulSoup(data.text,features="lxml")
    standings_table=soup.select('table.stats_table')[0]
    links=standings_table.find_all('a')
    links=[l.get("href") for l in links]
    links=[l for l in links if "/squads/" in l]
    team_urls=[f"https://fbref.com{l}" for l in links]
    today = pd.to_datetime(datetime.now())
    end_date=pd.to_datetime(datetime.today()+timedelta(days=30))
    # if(requestBody["start_date"]):
    #     today=pd.to_datetime(requestBody["start_date"])
    #     end_date=pd.to_datetime(today+timedelta(days=requestBody["interval"]))
    schmatches=[]
    for team_url in team_urls:
        data=requests.get(team_url)
        matches=pd.read_html(StringIO(data.text), match="Scores & Fixtures")
        scheduled=matches[0][matches[0]["Comp"]=="Premier League"]
        scheduled["Formation"]=scheduled["Formation"].ffill(axis=0)
        scheduled["Poss"]=scheduled["Poss"].ffill(axis=0)
        scheduled["Date"]=pd.to_datetime(scheduled["Date"])
        #ADD TEAM  NAME HERE
        team_name=team_url.split('/')[-1].strip("-Stats").replace("-"," ")
        scheduled["Team"]=team_name
        scheduled=scheduled[scheduled["Date"]>=today][scheduled["Date"]<=end_date]
        scheduled.columns=[c.lower() for c in scheduled.columns]
        scheduled.iloc[:,:-1]=scheduled.iloc[:,:-1].ffill(axis = 0)
        schmatches.append(scheduled)
    scheduled_matches=pd.concat(schmatches)
    scheduled_matches=scheduled_matches.drop(["gf","ga","xg","xga","attendance","captain","referee","match report","notes"],axis=1)
    scheduled_matches["team"]=scheduled_matches["team"].map(mapping)
    scheduled_matches["opponent"]=scheduled_matches["opponent"].map(mapping)
    scheduled_matches.index=[i for i in range(0,len(scheduled_matches.index))]
    response=scheduled_matches.to_json(index=False,date_format="iso",orient='records')
    return json.dumps(response)


# if __name__ == "__main__": 
# 	app.run(port=5000)
