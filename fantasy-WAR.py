# -*- coding: utf-8 -*-
"""
Created on Tue May 10 07:57:49 2022

@author: ryannewbury
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import pandas as pd
import scipy
pip install espn_api
pip install nfl_data_py
import nfl_data_py as nfl
import requests
import time
import json


class Settings:
    def __init__(self,season,scoring,teams,qb,rb,wr,te,flex,k,dst):
        self.season=season
        self.scoring=scoring
        self.teams=teams
        self.qb=qb
        self.rb=rb
        self.wr=wr
        self.te=te
        self.flex=flex
        self.k=k
        self.dst=dst
        if self.season >= 2021:
            self.weeks=range(1,17)
        else:
            self.weeks=range(1,16)
        self.league_id = 172
        self.ids = nfl.import_ids()

class WAR(Settings):
    def __init__(self, season, scoring, teams, qb, rb, wr, te, flex, k, dst):
        super().__init__(season, scoring, teams, qb, rb, wr, te, flex, k, dst)
        self.ros=nfl.import_rosters(range(self.season,self.season+1))
        self.ros.dropna(subset=['player_id'],inplace=True)
        self._fweek = []
        self._alldata =[]
        self._tproj = []
        self._rproj = []
        self._avetmscore = ()
        self._avetmstd = ()
        self.wp_ave=[]
        if self.scoring == 0:
            self._score='fantasy_points'
        elif self.scoring == .5:
            self._score='fantasy_points_half'
        elif self.scoring == 1:
            self._score='fantasy_points_ppr'
        
    def get_week(self):   
        fweek=nfl.import_weekly_data(range(self.season,self.season+1),['player_id','player_name',
            'recent_team','season','week','season_type','fantasy_points','fantasy_points_ppr'])
        if self.season >=2021:
            fweek=fweek.loc[fweek['week']<=16]
        else:
            fweek=fweek.loc[fweek['week']<=15]
    
        fweek=pd.merge(fweek,self.ros[['player_id','position','season','team'
                                  ]],on=['player_id','season'])
        fweek['fantasy_points_half']=((
            fweek['fantasy_points']+fweek['fantasy_points_ppr'])/2)
        fweek=fweek.loc[(fweek['position']=='QB')|(fweek['position']=='RB')|
                        (fweek['position']=='WR')|(fweek['position']=='TE')]
        self._fweek=fweek
        return fweek  

##########add outs so it doesnt go through process everytime   
    def get_espn_data(self):
        if isinstance(self._fweek,list):
            self.get_week()                     
        url = "https://fantasy.espn.com/apis/v3/games/ffl/seasons/" + \
            str(self.season) + "/segments/0/leagues/" + str(self.league_id) + \
            "?view=kona_player_info" 
        filters = {"players": {
            "filterStatus":{
                "value":[
                    "FREEAGENT",
                    "WAIVERS",
                    "ONTEAM"
                    ]
                },
            "limit":400,
            "sortPercOwned":{
                "sortPriority":1,
                "sortAsc":False
                },
            "sortDraftRanks":{
                "sortPriority":100,
                "sortAsc":True,
                "value":"STANDARD"
                }
            }
        }    

        data = []
        for week in self.weeks:
            d = requests.get(url, 
                             headers = {'x-fantasy-filter': json.dumps(filters)},
                             params={'scoringPeriodId': week}).json()    
            for pl in d['players']:
                name = pl['player']['fullName']
                espn_id = pl['player']['id']
                inj = 'NA'
                try:
                    inj = pl['player']['injuryStatus']
                except:
                    pass
                    # projected/actual points
                proj, act = None, None
                for stat in pl['player']['stats']:
                    if stat['scoringPeriodId'] != week:
                        continue
                    #actual points
                    if stat['statSourceId'] == 0:
                        act = stat['appliedTotal']
                    #projected points
                    elif stat['statSourceId'] == 1:
                        proj = stat['appliedTotal']
                data.append([self.season,
                    week, name, espn_id, inj, proj, act])                
        data = pd.DataFrame(data, 
                            columns=['season','week', 'player','espn_id', 
                                      'status', 'proj', 'actual'])       
        data = pd.merge(data,self.ids[['espn_id','gsis_id']],how = 'left', on='espn_id')
        data.rename(columns={'gsis_id':'player_id'},inplace=True)
        dst=data.loc[data['status'] == 'NA']
        data=data.loc[data['status'] != 'NA']
        data=pd.merge(data,self.ros[['player_id','position','season','team']],on=[
            'player_id','season'])
        data=pd.merge(data,self._fweek[['player_id','season','week',
            'fantasy_points','fantasy_points_ppr','fantasy_points_half']],on=[
                'player_id','season','week'], how ='left')                                                   
        data.fillna(0,inplace=True)
        dst['position']='DST'
        alldata=pd.concat([data,dst])
        for col in ['fantasy_points','fantasy_points_half','fantasy_points_ppr']:
            alldata.loc[(alldata['position']=='K')|
                    (alldata['position']=='DST'),col]=(
                        alldata['actual'].loc[
                    (alldata['position']=='K')|
                    (alldata['position']=='DST')])
        self._alldata=alldata
        return alldata
                    
    def get_proj(self,Top=True):
        if isinstance(self._alldata,list):
            self.get_espn_data()
        posnum={'QB':self.teams*self.qb,'RB':self.teams*self.rb,
                'WR':self.teams*self.wr,'TE':self.teams*self.te,
                'DST':self.teams*self.dst,'K':self.teams*self.k}   
        #normal positions
        for week in self.weeks:
            w=self._alldata.loc[(self._alldata['week']==week)&(
                self._alldata['season']==self.season)]
            for pos in posnum.keys():
                top=w.loc[w['position']==pos].sort_values(by='proj',
                                ascending=False).iloc[0:posnum[pos]]
                top['top']=1
                rep=w.loc[w['position']==pos].sort_values(by='proj',
                        ascending=False).iloc[posnum[pos]:(posnum[pos])*2]  
                rep['top']=0                                                                 
                if week == 1 and pos == 'QB':
                    tproj=top
                    rproj=rep
                else:    
                    tproj=pd.concat([tproj,top])
                    rproj=pd.concat([rproj,rep])         
        for week in self.weeks:
            w=self._alldata.loc[(self._alldata['week']==week)&(
                self._alldata['season']==self.season)]
            for pos in ['WR','TE','RB']:
                top=w.loc[w['position']==pos].sort_values(by='proj',
                        ascending=False).iloc[posnum[pos]:(posnum[pos])*2]                                                            
                if pos == 'WR':
                    flex=top
                else:
                    flex=pd.concat([flex,top])
            flex.sort_values(by='proj',ascending=False,inplace=True)
            flex['position']='FLEX'
            flext=flex.iloc[0:24]
            flext['top']=1
            flexr=flex.iloc[24:48]
            flexr['top']=0
            tproj=pd.concat([tproj,flext])
            rproj=pd.concat([rproj,flexr])
            self._tproj=tproj
            self._rproj=rproj
        if Top == True:
            return tproj
        elif Top == False:
            return rproj
    
    def get_team_ave(self):
        if isinstance(self._tproj,list) and isinstance(self._rproj,list):
            self.get_proj(True)
        tpos=self._tproj.groupby('position',as_index=False).agg({
            'proj':['mean','std'],
            self._score:['mean','std']})
        tpos.columns=tpos.columns.to_flat_index().str.join('_')
        tpos.columns=['position','proj_mean','proj_std','fp_mean','fp_std']        
        rpos=self._rproj.groupby('position',as_index=False).agg({
            'proj':['mean','std'],
            self._score:['mean','std']})
        rpos.columns=rpos.columns.to_flat_index().str.join('_')
        rpos.columns=['position','proj_mean','proj_std','fp_mean','fp_std']      
        pptm={'QB':self.qb,'RB':self.rb,'WR':self.wr,'TE':self.te,'FLEX':self.flex,
              'K':self.k,'DST':self.dst}
        for pos in pptm.keys():
            tpos.loc[tpos['position']==pos, 'pts_mean'] = tpos['fp_mean']*pptm[pos]
            tpos.loc[tpos['position']==pos, 'pts_std'] = (((tpos['fp_std'])**2)*pptm[pos])
        self._alldata=pd.merge(self._alldata,rpos[['position','fp_mean']], on='position',how='left')
        self._alldata.rename(columns={"fp_mean":"rep_mean"},inplace=True)
        self._alldata=pd.merge(self._alldata,tpos[['position','fp_mean']], on='position',how='left')
        self._alldata.rename(columns={"fp_mean":"top_mean"},inplace=True)  
        self._alldata = self._alldata.loc[
            (self._alldata['position'] == 'QB')|
            (self._alldata['position'] == 'RB')|
            (self._alldata['position'] == 'WR')|
            (self._alldata['position'] == 'TE')|
            (self._alldata['position'] == 'K')|
            (self._alldata['position'] == 'DST')|
            (self._alldata['position'] == 'FLEX')]
        avetmscore=tpos['pts_mean'].sum()
        avetmstd=(tpos['pts_std'].sum())**(0.5)
        self._avetmscore=avetmscore
        self._avetmstd=avetmstd
        return avetmscore, avetmstd

    def get_war(self):
        if isinstance(self._avetmscore,tuple) and isinstance(self._avetmstd,tuple):
            self.get_team_ave()
        self._alldata['above_rep']=(self._alldata[self._score]-self._alldata['top_mean'])
        
        self._alldata.loc[self._alldata['above_rep'] == -self._alldata['top_mean'], 'above_rep'] = (
            self._alldata['rep_mean']-self._alldata['top_mean'])
        
        self._alldata.loc[self._alldata['position']=='DST', 'above_rep']= (
            self._alldata['above_rep'].fillna(self._alldata['rep_mean']-self._alldata['top_mean']))
                                                                           
        self._alldata['wp']=norm(loc=self._avetmscore, scale = self._avetmstd
                            ).cdf(self._avetmscore+self._alldata['above_rep'])

            
        wp_ave=self._alldata.groupby(['player_id','player','position'],as_index=False).agg(
            {'team':'last','wp':'mean'})
        
        wp_dst=self._alldata.loc[self._alldata['position']=='DST'].groupby(
            ['player','position'],as_index=False).agg({'wp':'mean'})
        
        wp_ave=pd.concat([wp_ave,wp_dst])
        
        self._alldata.loc[self._alldata['above_rep']==(self._alldata['rep_mean']-self._alldata['top_mean']),
                    'wp_rep'] = self._alldata['wp']
      
        wp_rep=self._alldata.groupby('position',as_index=False).agg(
            {'wp_rep':'last'})
        
        wp_ave=pd.merge(wp_ave,wp_rep,on='position',how='left')
        
        wp_ave['wins_rep']=wp_ave['wp_rep']*max(self.weeks)
        
        wp_ave['wp_pl']=wp_ave['wp']*max(self.weeks)

        wp_ave['WAR']=wp_ave['wp_pl']-wp_ave['wins_rep']
        wp_ave.sort_values(by='WAR',ascending=False,inplace=True)
        wp_ave.reset_index(inplace=True,drop=True)
        self.wp_ave=wp_ave
        return wp_ave
    
    def get_war_plot(self,numplayers=50):
        if isinstance(self.wp_ave,list):
            self.get_war()
        y=self.wp_ave.WAR.head(numplayers)
        x=self.wp_ave.player.head(numplayers)
        fig,ax = plt.subplots(figsize=(20, 20))
        colors ={'QB':'blue',
                 'RB':'red',
                 'WR':'green',
                 'TE':'orange',
                 'K':'purple',
                 'DST':'yellow'}
        c = self.wp_ave['position'].apply(lambda x: colors[x])
        bars = ax.barh(x,y,height = 1, color = c,edgecolor='black')
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        ax.set_title(f'{self.season} Fantasy WAR',fontsize=36,fontweight='bold')
        ax.set_ylabel('Player Name', fontsize = 24,fontweight='bold')
        ax.set_xlabel('WAR', fontsize=24,fontweight='bold')
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
            
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
         
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 1)

        labels=list(colors.keys())

        handles=[plt.Rectangle((0,0),1,1, color = colors[label]) 
                 for label in labels]

        plt.legend(handles, labels, loc = 'center right',prop={'size':18})

        ax.invert_yaxis()
        
        ax.grid(b = True, color = 'grey',
                linestyle = '-.',linewidth = 0.5,
                alpha = 0.2)

        ax.bar_label(bars, labels=(round(y,2)), label_type='edge', 
                     fontsize = 16, color='black', padding =2.5)


        fig.text(0.8, 0.15, 'Data: nflfastR & ESPN', fontsize =16,
                 color = 'grey', ha = 'right', va = 'bottom',
                 alpha = 0.7)
        plt.tight_layout()
        plt.show()



