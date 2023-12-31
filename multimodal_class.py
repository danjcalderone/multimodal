import numpy as np
import numpy.linalg as mat
import scipy as sp
import scipy.linalg as smat
# import cvxpy as cp

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import peartree as pt #turns GTFS feed into a graph
import folium
import gtfs_functions as gtfs
import pickle

import matplotlib.pyplot as plt


from matplotlib.patches import FancyArrow
from itertools import product 
from random import sample

from shapely.geometry import Polygon, Point, LineString

import sklearn as sk
from sklearn import cluster as cluster
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn import metrics
from sklearn.cluster import DBSCAN

from scipy.spatial import ConvexHull, convex_hull_plot_2d

import time
import warnings
warnings.filterwarnings('ignore')

import random
import sys

import pygris
import os


os.chdir('/Users/dan/Documents/transit_webapp/')
from core import db, optimizer
import config
from common.helpers import _internalrequest


### NEWER STUFF ### NEWER STUFF ### NEWER STUFF ### NEWER STUFF ### NEWER STUFF 
### NEWER STUFF ### NEWER STUFF ### NEWER STUFF ### NEWER STUFF ### NEWER STUFF 
# from json import JSONDecodeError
import simplejson
from shapely.geometry import Polygon, Point
import shapely as shp
import plotly.express as px
import alphashape
from descartes import PolygonPatch
import time
import os
import pickle


os.chdir('/Users/dan/Documents/multimodal/')

########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 


############ BASIC FUNCTIONS ############
############ BASIC FUNCTIONS ############
############ BASIC FUNCTIONS ############
############ BASIC FUNCTIONS ############




def dictComb(VALUES0,coeffs): 
    """
    DESCRIPTIONS: takes linear combinations of dictionary objects...
    INPUTS:
    - VALUES0: list of dictionaries (all should have same fields)
    - coeffs: list of coefficients. (same length as VALUES0)
    OUTPUTS:
    - out: dictionary with same tags and linear combinations
    """ 
    out = {}
    VALUES = []
    for i,VALUE in enumerate(VALUES0):
        VALUES.append(VALUE.copy())

    for _,tag in enumerate(VALUES[0]):
        temp = 0
        for j,VALUE in enumerate(VALUES):
            temp = temp + coeffs[j]*VALUE[tag]
        out[tag] = temp
    return out 

def thresh(x,ths):
    """
    DESCRIPTION: implements sigmoid style threshold 
    (used for plotting purposes)
    """
    m = (ths[3]-ths[1])/(ths[2]-ths[0]);
    b = ths[1]-m*ths[0];
    return np.min([np.max([ths[1],m*x+b]),ths[3]])

def chopoff(x,den,mn,mx): return np.min([np.max([mn,x/den]),mx])



def invDriveCostFx(c,poly):   ##### HELPER FUNCTION 
    if len(poly)==2:
        alpha0 = poly[0]; alpha1 = poly[1];
        out = np.power((c - alpha0)/alpha1,1./1.)
    elif len(poly)==3:
        a0 = poly[0]; a1 = poly[1]; a2 = poly[2];
        out = np.sqrt(c - (a0/a2)+np.power(a1/(2*a2),2.)) - (a1/(2*a2))
    elif len(poly)==4:
        alpha0 = poly[0]; alpha1 = poly[3];
        out = np.power((c - alpha0)/alpha1,1./3.)
    else: 
        pwr = int(len(poly)-1);
        alpha0 = poly[0]; alpha1 = poly[pwr];
        out = np.power((c-alpha0)/alpha1,1./pwr)
    return out


def str2speed(str1):   #### HELPER FUNCTION 
    # print(str1)
    str2 = str1[:-4];
    if ' mph' in str1:
        str2 = str1[:str1.index(' mph')];
    else:
        str2 = str1;
    return int(str2)


###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 
###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE ###### GENERATE 

def ptsBoundary(corners,nums):
    out = [];
    for i,pt0 in enumerate(corners):
        if i==len(corners)-1:pt1 = corners[0];
        else: pt1 = corners[i+1]
        for k in range(nums[i]):
            alpha = k/nums[i];
            out.append((1-alpha)*pt0+alpha*pt1);
    out = np.array(out)
    return out

def filterODs(DF0,box,eps=1.0):
    minx = box[0]; maxx = box[1]; 
    miny = box[2]; maxy = box[3];
    DF = DF0.copy();
    tag = 'home_loc_lon'; val = minx
    DF[tag] = np.where(DF[tag] < val, val, DF[tag])
    tag = 'home_loc_lon'; val = maxx
    DF[tag] = np.where(DF[tag] > val, val, DF[tag])
    tag = 'home_loc_lat'; val = miny
    DF[tag] = np.where(DF[tag] < val, val, DF[tag])
    tag = 'home_loc_lat'; val = maxy
    DF[tag] = np.where(DF[tag] > val, val, DF[tag])
    
    OUT = pd.DataFrame({'pop':[],'hx':[],'hy':[],'wx':[],'wy':[]},index=[]);
    itr = 0;
    maxiter = 1000;
    while len(DF) > 0: # and (itr<maxiter):
        idx_tag = 'pop'+str(itr)
        hx = DF['home_loc_lon'].iloc[0]
        hy = DF['home_loc_lat'].iloc[0]
        wx = DF['work_loc_lon'].iloc[0]
        wy = DF['work_loc_lat'].iloc[0]
        maskhx = np.abs(DF['home_loc_lon']-hx)<eps
        maskhy = np.abs(DF['home_loc_lat']-hy)<eps
        maskwx = np.abs(DF['work_loc_lon']-wx)<eps
        maskwy = np.abs(DF['work_loc_lat']-wy)<eps
        mask = maskhx & maskhy & maskwx & maskwy
        pop_num = len(DF[mask])
        POP = pd.DataFrame({'pop':[pop_num],'hx':[hx],'hy':[hy],'wx':[wx],'wy':[wy]},index=[idx_tag]);
        OUT = pd.concat([OUT,POP])
        #OUT = OUT.append(POP);
        DF = DF[~mask]
        itr = itr + 1;
        if np.mod(itr,1000)==0:
            print(itr)
    return OUT



def kmeans_nodes(num,mode,GRAPHS,node_set = 'all',find_nodes=True):

    GRAPH = GRAPHS[mode];
    feed = GRAPH;
    if node_set == 'all':
        if mode == 'gtfs':
            nodes = feed.stops.index;
        else:
            nodes = GRAPH.nodes;
    else:
        nodes = node_set;

    MM = []
    for i,node in enumerate(nodes):
        if mode == 'gtfs':
            # feed = GRAPH;
            lat = feed.stops.stop_lat[node]
            lon = feed.stops.stop_lon[node]            
            temp = np.array([lon,lat])
            MM.append(temp)
        else:
            NODE = GRAPH.nodes[node]            
            lat = NODE['y']
            lon = NODE['x']
            temp = np.array([lon,lat])
            MM.append(temp)            
    MM = np.array(MM);
    
    kmeans = cluster.KMeans(n_clusters=num);
    kmeans_output = kmeans.fit(MM)
    centers = kmeans_output.cluster_centers_
    
    center_nodes = [];
    if find_nodes==True:
        for i,loc in enumerate(centers):
            if mode == 'gtfs':
                lon = loc[0];
                lat = loc[1];
                eps = 0.01;
                close = np.abs(feed.stops.stop_lat - lat) + np.abs(feed.stops.stop_lon - lon);
                close = close==np.min(close)
                found_stop = feed.stops.stop_id[close];
                found_stop = list(found_stop)[0]
                node = found_stop
                center_nodes.append(node);                
            else:
                node = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1])            
                center_nodes.append(node);
    return {'centers':centers,'nodes':center_nodes}


def SETUP_POPULATIONS_CHATTANOOGA(GRAPHS,cutoff_bnds = [],params={}):


    #### initial parameters...
    if 'pop_cutoff' in  params: pop_cutoff = params['pop_cutoff'];
    else: pop_cutoff = 30;

    if 'OD_version' in  params: OD_version = params['OD_version']
    else: OD_version = 'basic'

    SEG_TYPES = params['SEG_TYPES'];
    

    # for i,samp in enumerate(samps):
    #     tag = 'person'+str(i);
    zzgraph = GRAPHS['all']
    
    temp = [Point(n['x'],n['y']) for i,n in zzgraph.nodes(data=True)]
    temp2 = np.array([[n['x'],n['y']] for i,n in zzgraph.nodes(data=True)])
    use_box = True;
    if use_box:
        minz = np.min(temp2,0); maxz = np.max(temp2,0);
        dfz = maxz-minz; centerz = minz + 0.5*dfz;
        skz = 0.9;
        pts = 0.5*np.array([[dfz[0],dfz[1]],[-dfz[0],dfz[1]],[-dfz[0],-dfz[1]],[dfz[0],-dfz[1]]]) + centerz;
        points = [Point(zz[0],zz[1]) for i,zz in enumerate(pts)]
        temp = temp + points;
    graph_boundary = gpd.GeoSeries(temp).unary_union.convex_hull
    
    cornersz = np.array([[maxz[0],maxz[1]],[minz[0],maxz[1]],[minz[0],minz[1]],[maxz[0],minz[1]]]);
    #corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
    divx = 40; divy = int(divx*(dfz[1]/dfz[0]));
    ptBnds = ptsBoundary(cornersz,[divx,divy,divx,divy])
    # plt.plot(ptBnds[:,0],ptBnds[:,1],'o')
    # for i,samp in enumerate(samps):
    #     tag = 'person'+str(i);    
    ####### from pygris import tracts, block_groups
    import pygris
    ### Reading in the loads data set... (pandas dataframes)
    asdf0 = pd.read_parquet('data/pop/lodes_combinations_upd.parquet')
    # asdf0.head()
    
    ##### forget what this does.... 
    BGDEFS = pygris.block_groups(state = "TN", county="Hamilton", cb = True, cache=True)
    BGDEFS['pt']  = BGDEFS['geometry'].representative_point()
    BGDEFS['lon'] = BGDEFS['pt'].x;
    BGDEFS['lat'] = BGDEFS['pt'].y;



    #### Reading American Commuter Survey data set (pandas dataframes)
    ### information about vehicle ussage 
    VEHS = pd.read_csv('data/pop/ACSDT5Y2020.B992512-Data.csv')
    # BGDEFS['AFFGEOID']
    #VEHS = VEHS.rename(columns={'B992512_001E':'from_cbg','home_geo':'from_geo','w_geocode':'to_cbg','work_geo':'to_geo'}).drop(columns=['return_time'])[['from_cbg', 'to_cbg', 'total_jobs', 'go_time', 'from_geo', 'to_geo']]
    VEHS = VEHS.rename(columns={'GEO_ID':'AFFGEOID','B992512_001E':'workers','B992512_002E':'wout_cars','B992512_003E':'w_cars'}).drop(columns=['B992512_001EA','B992512_002EA','B992512_003EA','Unnamed: 8'])
    VEHS = VEHS.drop([0])
    
    print(len(VEHS))
    

    ### computing the percentage of workers with and without cars... (within pandas)
    VEHS['workers'] = pd.to_numeric(VEHS['workers'],errors='coerce')
    VEHS['wout_cars'] = pd.to_numeric(VEHS['wout_cars'],errors='coerce')
    VEHS['w_cars'] = pd.to_numeric(VEHS['w_cars'],errors='coerce')
    VEHS['percent_w_cars'] = VEHS['w_cars']/VEHS['workers'];
    VEHS['percent_wout_cars'] = VEHS['wout_cars']/VEHS['workers'];
    VEHS = VEHS.merge(BGDEFS,how='left',on='AFFGEOID')
    
    

    ### ADDING 


    ##### END OF LOADING IN DATA.... ##### END OF LOADING IN DATA....
    ##### END OF LOADING IN DATA.... ##### END OF LOADING IN DATA....



    # BGDEFS.explore()
    
    #VEHS.ilochead()
    # print(np.sum(list(VEHS['workers'])))
    # print(np.sum(list(VEHS['wout_cars'])))
    # print(np.sum(list(VEHS['w_cars'])))



    ### Filtering out population members outside a particular bounding box... 
    if len(cutoff_bnds)>0:
        #print('cutting off shit...')

        bot_bnd = cutoff_bnds[0];
        top_bnd = cutoff_bnds[1];

        mask1 = asdf0['home_loc_lon'] >= bot_bnd[0];
        mask1 = mask1 &  (asdf0['home_loc_lon'] <= top_bnd[0]);
        mask1 = mask1 &  (asdf0['home_loc_lat'] >= bot_bnd[1]);
        mask1 = mask1 &  (asdf0['home_loc_lat'] <= top_bnd[1]);

        mask1 = mask1 &  (asdf0['work_loc_lon'] >= bot_bnd[0]);
        mask1 = mask1 &  (asdf0['work_loc_lon'] <= top_bnd[0]);
        mask1 = mask1 &  (asdf0['work_loc_lat'] >= bot_bnd[1]);
        mask1 = mask1 &  (asdf0['work_loc_lat'] <= top_bnd[1]);
        asdf0 = asdf0[mask1]
    box = [minz[0],maxz[0],minz[1],maxz[1]];

    #### explain this later.... 
    asdf2 = filterODs(asdf0,box,eps=params['eps_filterODs']);

    ##################################################################
    ##################################################################
    ##################################################################


    ### initialization for the main loop.... 
    ### initialization for the main loop.... 

    mask1 = asdf2['pop']>pop_cutoff; #8;
    mask2 = asdf2['pop']<-1;
    asdf = asdf2[mask1 | mask2];
    # plt.plot(list(asdf['pop']))
    print('total pop is', np.sum(asdf['pop']),'out of',np.sum(asdf2['pop']))
    print('number of agents: ',len(asdf['pop']))
    
    #num_people = 40;
    #samps = sample(list(asdf.index),num_people);
    samps = list(asdf.index)
    num_people = len(samps)
    print(num_people)

    
    locs = [];
    for i,node in enumerate(GRAPHS['drive'].nodes):
        NODE = GRAPHS['drive'].nodes[node];
        lon = NODE['x']; lat = NODE['y']
        locs.append([lon,lat]);
    locs = np.array(locs);

    if len(cutoff_bnds)>0:
        minz = cutoff_bnds[0];
        maxz = cutoff_bnds[1];
    else:
        minz = np.min(locs,0)
        maxz = np.max(locs,0)
    
    NODES = {}
    
    NODES['orig'] = [];#sample(list(sample_graph.nodes()), num_people)
    NODES['dest'] = [];#sample(list(sample_graph.nodes()), num_targets)
    NODES['delivery1'] = []; #sample(list(sample_graph.nodes()), num_deliveries)
    NODES['delivery2'] = []; #sample(list(sample_graph.nodes()), num_deliveries)
    NODES['delivery1_transit'] = [];
    NODES['delivery2_transit'] = [];
    NODES['drive_transit'] = [];
    
    LOCS = {};
    LOCS['orig'] = [];
    LOCS['dest'] = [];
    LOCS['delivery1'] = []
    LOCS['delivery2'] = []
    SIZES = {};
    
    home_locs = [];
    work_locs = [];
    home_sizes = {};
    work_sizes = {};
    
    sample_graph = GRAPHS['drive'];
    
    PRE = {};
    compute_nodes = True;
    if compute_nodes:
        home_nodes = []
        work_nodes = []
    
    # for i,samp in enumerate(samps):
    i1 = 0;
    i2 = 0;
    i3 = 0;
    # for i,samp in enumerate(asdf.index):
    
    asdflist = list(asdf.index);  ## list of indices of asdf dataframe...
    people_tags = [];


    ## home = origin
    ## work = dest


    ##### MAIN LOOP ##### MAIN LOOP ##### MAIN LOOP ##### MAIN LOOP 
    ##### MAIN LOOP ##### MAIN LOOP ##### MAIN LOOP ##### MAIN LOOP 

    ############################################################
    ####  VERSION 1: uses the data sets loaded above.... 
    ############################################################
    if OD_version == 'basic':


        while i1<len(asdf.index):
            i = i2;
            samp = asdflist[i1];  # grabbing appropriate index of the dataframe... 
                
            
            ### pulling out information from the data frame... 
            hlon = asdf['hx'].loc[samp]  # home longitude
            hlat = asdf['hy'].loc[samp]  # home latitude...
            wlon = asdf['wx'].loc[samp]  # work longitude
            wlat = asdf['wy'].loc[samp]
            home_loc = np.array([hlon,hlat]); # locations...  
            work_loc = np.array([wlon,wlat]);


            #### figuring out what percentage of the population has cars or not based on ACS given home location
            ### VEHS has the info the ACS... driving information.... 
            #### for a population... what region are they in so which driving statistics apply... 
            VALS = np.abs(VEHS['lon']-hlon)+np.abs(VEHS['lat']-hlat);
            mask1 = VALS == np.min(VALS); ### find closest region in the VEHS data (so apply that driving statistic)
            perc_wcars = list(VEHS[mask1]['percent_w_cars'])[0]
            perc_wnocars = list(VEHS[mask1]['percent_wout_cars'])[0]



        
            home_size = asdf['pop'].loc[samp]
            work_size = asdf['pop'].loc[samp]
            
            ################################################################
            test1 = (maxz[0]>=home_loc[0]) and (maxz[1]>=home_loc[1]);
            test2 = (minz[0]<=home_loc[0]) and (minz[1]<=home_loc[1]);
            test3 = (maxz[0]>=work_loc[0]) and (maxz[1]>=work_loc[1]);
            test4 = (minz[0]<=work_loc[0]) and (minz[1]<=work_loc[1]);
            if True: #test1 and test2 and test3 and test4:
                
                #### VERSION 1 #### VERSION 1 #### VERSION 1 #### VERSION 1
                #### adding a population with cars... 
                if perc_wcars > 0.:

                    
                    if np.mod(i2,200)==0: print(i2) ### just shows how fast the loop is running... 


                    tag = 'person'+str(i2); ### creating a poulation tag... 
                    people_tags.append(tag)
                    PRE[tag] = {}; ## initializing 
                
                    ### adding location information.... 
                    PRE[tag]['orig_loc'] = home_loc; 
                    PRE[tag]['dest_loc'] = work_loc;
                    LOCS['orig'].append(home_loc);  
                    LOCS['dest'].append(work_loc);
                    

                    #### adding whether or not they have a car... 
                    PRE[tag]['take_car'] = 1.;
                    PRE[tag]['take_transit'] = 1.;
                    PRE[tag]['take_ondemand'] = 1.;        
                    PRE[tag]['take_walk'] = 1.;
            
                                      
                    if compute_nodes:
                        ### finding the nearest nodes within the driving network... 
                        home_node = ox.distance.nearest_nodes(GRAPHS['drive'],home_loc[0],home_loc[1]);
                        work_node = ox.distance.nearest_nodes(GRAPHS['drive'], work_loc[0],work_loc[1]);
                        

                        #### size of the population.... 
                        if home_node in home_sizes: home_sizes[home_node] = home_sizes[home_node] + home_size;
                        else: home_sizes[home_node] = home_size;
                        if work_node in work_sizes: work_sizes[work_node] = work_sizes[work_node] + work_size;
                        else: work_sizes[work_node] = work_size;
                        

                        ### adding nodes to different lists/objects... 
                        home_nodes.append(home_node);
                        work_nodes.append(work_node);
                        NODES['orig'].append(home_node);
                        NODES['dest'].append(work_node)
                        
                        PRE[tag]['home_node'] = home_node;        
                        PRE[tag]['work_node'] = work_node;
                        

                        #### adding the population size to the objects... 
                        PRE[tag]['pop'] = home_size*perc_wcars;

                    ##### adding specific trip types... 
                    # input from SEG_TYPES is generated using the function 
                    # generate_segtypes
                    samp = np.random.rand(1);  # extra sampling thing if we want to change the percentage... 
                    if (samp < 0.3):
                        seg_types = SEG_TYPES['car_opt']

                    else: 
                        seg_types = SEG_TYPES['car_only'] #[('drive',)]

                    PRE[tag]['seg_types'] = seg_types

                    ## seg_types: list of different travel modes... 
                    #### form...
                    # [('drive',),
                    #  ('ondemand',),
                    #  ('walk','gtfs','walk'),
                    #  ('walk','gtfs','ondemand'),
                    #  ('ondemand','gtfs','walk'),
                    #  ('ondemand','gtfs','ondemand')
                    #  ];
                    i2 = i2 + 1;

                    #### VERSION 2 #### VERSION 2 #### VERSION 2 #### VERSION 2 ####
        
                if perc_wnocars > 0.:
                    #### adding a population without cars... 
                    
                    if np.mod(i2,200)==0: print(i2)
                    tag = 'person'+str(i2);
                    people_tags.append(tag)
                    PRE[tag] = {};
                
                    PRE[tag]['orig_loc'] = home_loc
                    PRE[tag]['dest_loc'] = work_loc;
                    
                    LOCS['orig'].append(home_loc);
                    LOCS['dest'].append(work_loc);
                    
                    PRE[tag]['take_car'] = 0.;
                    PRE[tag]['take_transit'] = 1.;
                    PRE[tag]['take_ondemand'] = 1.;        
                    PRE[tag]['take_walk'] = 1.;
                        
                    if compute_nodes:
                        home_node = ox.distance.nearest_nodes(GRAPHS['drive'],home_loc[0],home_loc[1]);
                        work_node = ox.distance.nearest_nodes(GRAPHS['drive'], work_loc[0],work_loc[1]);
                        
                        if home_node in home_sizes: home_sizes[home_node] = home_sizes[home_node] + home_size;
                        else: home_sizes[home_node] = home_size;
                        if work_node in work_sizes: work_sizes[work_node] = work_sizes[work_node] + work_size;
                        else: work_sizes[work_node] = work_size;
                        
                        home_nodes.append(home_node);
                        work_nodes.append(work_node);
                        NODES['orig'].append(home_node);
                        NODES['dest'].append(work_node)
                        
                        PRE[tag]['home_node'] = home_node;        
                        PRE[tag]['work_node'] = work_node;
                        
                        PRE[tag]['pop'] = home_size*perc_wnocars;


                    seg_types = SEG_TYPES['car_no']
                    # [('ondemand',),
                    #              ('walk','gtfs','walk'),
                    #              ('walk','gtfs','ondemand'),
                    #              ('ondemand','gtfs','walk'),
                    #              ('ondemand','gtfs','ondemand')
                    #             ];              
                    PRE[tag]['seg_types'] = seg_types
                    i2 = i2 + 1;
                    
            i1 = i1 + 1;
        
        SIZES['home_sizes'] = home_sizes
        SIZES['work_sizes'] = work_sizes



    ###### ##############################################################################
    ###### VERSION 2: generates artifical data sampling gaussian distributions... 
    ###### ##############################################################################
    elif OD_version == 'gauss':

        orig_locs = np.array([0,2])
        dest_locs = np.array([0,2])

        ###### loading the gaussian distribution info 
        ### sampling from distributions
        for kk,stats in enumerate(params['gauss_stats']):
            stats
            num_pops = stats['num']
            orig_mean = stats['origs']['mean'] 
            dest_mean = stats['dests']['mean'] 
            orig_cov = stats['origs']['cov'] 
            dest_cov = stats['dests']['cov']
            pop = stats['pop']
            ### sampled origin and destination locations.... 
            orig_locs = np.vstack([orig_locs,np.random.multivariate_normal(orig_mean, orig_cov, size=num_pops)]);
            dest_locs = np.vstack([dest_locs,np.random.multivariate_normal(dest_mean, dest_cov, size=num_pops)]);


        # loop through each origin location... 
        for i,orig_loc in enumerate(orig_locs):
            dest_loc = dest_locs[i]
            home_loc = orig_loc; #np.array([hlon,hlat]);    
            work_loc = dest_loc; #np.array([wlon,wlat]);
        
            VALS = np.abs(VEHS['lon']-home_loc[0])+np.abs(VEHS['lat']-home_loc[1]);
            mask1 = VALS == np.min(VALS);

            perc_wcars = list(VEHS[mask1]['percent_w_cars'])[0]
            perc_wnocars = list(VEHS[mask1]['percent_wout_cars'])[0]
        
            home_size = pop; #['origs']['cov']; #1.;#asdf['pop'].loc[samp]
            work_size = pop; #origs']['cov']; #1.;#asdf['pop'].loc[samp]
            
            
            ################################################################
            test1 = (maxz[0]>=home_loc[0]) and (maxz[1]>=home_loc[1]);
            test2 = (minz[0]<=home_loc[0]) and (minz[1]<=home_loc[1]);
            test3 = (maxz[0]>=work_loc[0]) and (maxz[1]>=work_loc[1]);
            test4 = (minz[0]<=work_loc[0]) and (minz[1]<=work_loc[1]);
            if test1 and test2 and test3 and test4:
                

                ##### Adding populations.... 

                #### VERSION 1 #### VERSION 1 #### VERSION 1 #### VERSION 1
                #### if the population has a car....
                if perc_wcars > 0.:
                    if np.mod(i2,200)==0: print(i2)
                    tag = 'person'+str(i2);
                    people_tags.append(tag)
                    PRE[tag] = {};
                
                    PRE[tag]['orig_loc'] = home_loc
                    PRE[tag]['dest_loc'] = work_loc;
                    
                    LOCS['orig'].append(home_loc);
                    LOCS['dest'].append(work_loc);
                    
                    PRE[tag]['take_car'] = 1.;
                    PRE[tag]['take_transit'] = 1.;
                    PRE[tag]['take_ondemand'] = 1.;        
                    PRE[tag]['take_walk'] = 1.;
            
                    if compute_nodes:
                        home_node = ox.distance.nearest_nodes(GRAPHS['drive'],home_loc[0],home_loc[1]);
                        work_node = ox.distance.nearest_nodes(GRAPHS['drive'], work_loc[0],work_loc[1]);
                        
                        if home_node in home_sizes: home_sizes[home_node] = home_sizes[home_node] + home_size;
                        else: home_sizes[home_node] = home_size;
                        if work_node in work_sizes: work_sizes[work_node] = work_sizes[work_node] + work_size;
                        else: work_sizes[work_node] = work_size;
                        
                        home_nodes.append(home_node);
                        work_nodes.append(work_node);
                        NODES['orig'].append(home_node);
                        NODES['dest'].append(work_node)
                        
                        PRE[tag]['home_node'] = home_node;        
                        PRE[tag]['work_node'] = work_node;
                        
                        PRE[tag]['pop'] = home_size*perc_wcars;



                    samp = np.random.rand(1);
                    if (samp < 0.3):
                        seg_types = SEG_TYPES['car_opt']
                        # [('drive',),
                        #              ('ondemand',),
                        #              ('walk','gtfs','walk'),
                        #              ('walk','gtfs','ondemand'),
                        #              ('ondemand','gtfs','walk'),
                        #              ('ondemand','gtfs','ondemand')
                        #             ];

                    else: 
                        seg_types = SEG_TYPES['car_only'] #[('drive',)]

                    PRE[tag]['seg_types'] = seg_types
            
                    i2 = i2 + 1;




                #### VERSION 2 #### VERSION 2 #### VERSION 2 #### VERSION 2 ####
                ### if population doesn't have car 
                if perc_wnocars > 0.:
                    
                    if np.mod(i2,200)==0: print(i2)


                    #### creating a new population... 
                    tag = 'person'+str(i2);
                    people_tags.append(tag)
                    PRE[tag] = {};
                
                    PRE[tag]['orig_loc'] = home_loc
                    PRE[tag]['dest_loc'] = work_loc;
                    
                    LOCS['orig'].append(home_loc);
                    LOCS['dest'].append(work_loc);
                    
                    PRE[tag]['take_car'] = 0.;
                    PRE[tag]['take_transit'] = 1.;
                    PRE[tag]['take_ondemand'] = 1.;        
                    PRE[tag]['take_walk'] = 1.;
                        
                    if compute_nodes:
                        home_node = ox.distance.nearest_nodes(GRAPHS['drive'],home_loc[0],home_loc[1]);
                        work_node = ox.distance.nearest_nodes(GRAPHS['drive'], work_loc[0],work_loc[1]);
                        
                        if home_node in home_sizes: home_sizes[home_node] = home_sizes[home_node] + home_size;
                        else: home_sizes[home_node] = home_size;
                        if work_node in work_sizes: work_sizes[work_node] = work_sizes[work_node] + work_size;
                        else: work_sizes[work_node] = work_size;
                        
                        home_nodes.append(home_node);
                        work_nodes.append(work_node);
                        NODES['orig'].append(home_node);
                        NODES['dest'].append(work_node)
                        
                        PRE[tag]['home_node'] = home_node;        
                        PRE[tag]['work_node'] = work_node;
                        
                        PRE[tag]['pop'] = home_size*perc_wnocars;


                    seg_types = SEG_TYPES['car_no']
                    # [('ondemand',),
                    #              ('walk','gtfs','walk'),
                    #              ('walk','gtfs','ondemand'),
                    #              ('ondemand','gtfs','walk'),
                    #              ('ondemand','gtfs','ondemand')
                    #             ];              
                    PRE[tag]['seg_types'] = seg_types
                    i2 = i2 + 1;
                    
            i1 = i1 + 1;
        
        SIZES['home_sizes'] = home_sizes
        SIZES['work_sizes'] = work_sizes






    ####### SETTING UP ONDEMAND SERVICE
    ####### SETTING UP ONDEMAND SERVICE

    start_time = time.time()
    print('starting delivery1 sources...')
    for i,node in enumerate(NODES['delivery1']):
        if np.mod(i,200)==0: print(i)
        addNodeToDF(node,'drive',GRAPHS,NDF)
        
    print('starting delivery2 sources...')
    for i,node in enumerate(NODES['delivery2']):
        if np.mod(i,200)==0: print(i)
        addNodeToDF(node,'drive',GRAPHS,NDF)
            
    
            
    end_time = time.time()
    print('time to create nodes...: ',end_time-start_time)
        # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}    
    num_people = len(people_tags);
    num_targets = num_people;


    # if params['num_deliveries']

    if 'num_deliveries' in params:
        num_deliveries = params['num_deliveries']['delivery1'];
        num_deliveries2 = params['num_deliveries']['delivery2'];
    else:
        num_deliveries =  int(num_people/10);
        num_deliveries2 = int(num_people/10);


    ##### kmeans clustering of the population locations to see where the different ondemand vehicles should go
    ### CHANGE FOR PILOT....
    node_group = NODES['orig'] + NODES['dest']
    out = kmeans_nodes(num_deliveries,'ondemand',GRAPHS,node_group); 
    LOCS['delivery1'] = out['centers']
    out = kmeans_nodes(num_deliveries,'ondemand',GRAPHS,node_group); 
    LOCS['delivery2'] = out['centers']
    
    
    for i,loc in enumerate(LOCS['delivery1']):
        NODES['delivery1'].append(ox.distance.nearest_nodes(GRAPHS['ondemand'],loc[0],loc[1]));
    for i,loc in enumerate(LOCS['delivery2']):
        NODES['delivery2'].append(ox.distance.nearest_nodes(GRAPHS['ondemand'],loc[0],loc[1]));
        
            
    bus_graph = GRAPHS['gtfs'];
    # transit_start_nodes = sample(list(bus_graph.nodes()), num_sources)
    # transit_end_nodes = sample(list(bus_graph.nodes()), num_targets)
    delivery_transit_nodes = sample(list(bus_graph.nodes()), num_deliveries2)
    
    # indstoremove = [];
    # for i in range(len(LOCS['orig'])):
    # #     for j in range(len(LOCS['dest'])):
    # #         #add_OD_pair = True;
    #     try:
    #         orig = ox.distance.nearest_nodes(sample_graph, LOCS['orig'][i][0], LOCS['orig'][i][1]);
    #         dest = ox.distance.nearest_nodes(sample_graph, LOCS['dest'][i][0], LOCS['dest'][i][1]);
    #         path = nx.shortest_path(sample_graph, source=orig, target=dest, weight=None)
    #     except:
    #         if not(i in indstoremove):
    #             indstoremove.append(i)                
                    
    # print(indstoremove)
    # for i in indstoremove[::-1]:
    #     print('Origin ',i,' deleted...')
    #     LOCS['orig'].pop(i)
    #     LOCS['dest'].pop(i)    
    
    end_time = time.time();
    print('time to setup origins & dests: ',end_time - start_time)    


    ### TODO: COLLAPSE DOWN INTO ONE OBJECT...

    ### PRE is main object... 
    return {'PRE':PRE,'NODES':NODES,'LOCS':LOCS,'SIZES':SIZES,'VEHS':VEHS}




######## ================= GENERATE WORLD ================  ###################
######## ================= GENERATE WORLD ================  ###################
######## ================= GENERATE WORLD ================  ###################


class DASHBOARD:
    def __init__(self): pass
    def sortData(self): pass
    def generate(self): pass


class GRID:
    def __init__(self,params): pass


    def NOTEBOOKloadGraphs(self):  ### FROM NOTEBOOK

        szz = 1.; radius = szz*5000;
        time_window = [start,end]
        print('LOADING GRAPHS:')
        OUT = SETUP_GRAPHS_CHATTANOOGA(center_point,radius,time_window,bnds = bnds);
        GRAPHS = OUT['GRAPHS']; RGRAPHS = OUT['RGRAPHS']; feed = OUT['feed']

        #graph_bus = load_feed_as_graph(feed);
        mode = 'all';
        bgcolor = [0.8,0.8,0.8,1];
        # %time fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=bgcolor,node_size=1,figsize=(20,20),edge_color = [1,1,1],show=False,); #file_format='svg')
        # %time fig, ax = ox.plot_graph(GRAPHS[mode],bgcolor=bgcolor,node_size=1,figsize=(5,5),edge_color = [1,1,1],show=False); #file_format='svg')

    def NOTEBOOK(self):

        print('LOADING POPULATION DATA:')
        params = {'pop_cutoff':1}
        params['SEG_TYPES'] = generate_segtypes('reg8') # reg7 reg1,reg2,bg


        cent_pt = np.array(center_point)
        # dest_shift = np.array([0.001,-0.000]);
        dest_shift = np.array([0.003,-0.005]);
        orig_shift = np.array([0.022,-0.04]);
        orig_shift2 = np.array([-0.015,-0.042]);
        orig_shift3 = np.array([0.035,-0.01]);

        thd = 0.3; tho = -0.0; tho2 = -0.0; tho3 = 0.0;
        Rd = np.array([[np.cos(thd),-np.sin(thd)],[np.sin(thd),np.cos(thd)]]);
        Ro = np.array([[np.cos(tho),-np.sin(tho)],[np.sin(tho),np.cos(tho)]]);
        Ro2 = np.array([[np.cos(tho2),-np.sin(tho2)],[np.sin(tho2),np.cos(tho2)]]);
        Ro3 = np.array([[np.cos(tho3),-np.sin(tho3)],[np.sin(tho3),np.cos(tho3)]]);
        COVd  = np.diag([0.0001,0.00007]);
        # COVd3  = np.diag([0.00004,0.00004]);
        COVo  = np.diag([0.00005,0.0001]);
        COVo2 = np.diag([0.00003,0.0001]);
        COVo3 = np.diag([0.00008,0.00008]);
        # COVd  = np.diag([0.000002,0.000002]);
        # COVo  = np.diag([0.000002,0.000002]);
        # COVo2 = np.diag([0.000002,0.000002]);
        COVd = Rd@COVd@Rd.T
        COVo = Ro@COVo@Ro.T
        COVo2 = Ro2@COVo2@Ro2.T
        COVo3 = Ro3@COVo3@Ro3.T

        params['OD_version'] = 'gauss';
        params['gauss_stats'] = [{'num':70,'pop':1,
                                 'dests':{'mean':cent_pt+dest_shift,'cov':COVd},
                                 'origs':{'mean':cent_pt+orig_shift,'cov':COVo}},
                                 {'num':30,'pop':1,
                                 'dests':{'mean':cent_pt+dest_shift,'cov':COVd},
                                 'origs':{'mean':cent_pt+orig_shift2,'cov':COVo2}},
                                {'num':70,'pop':1,
                                 'dests':{'mean':cent_pt+dest_shift,'cov':COVd},
                                 'origs':{'mean':cent_pt+orig_shift3,'cov':COVo3}}]

        params['num_deliveries'] = {'delivery1':20,'delivery2':20}

        params['eps_filterODs'] = 0.001
        cutoff_bnds = bnds;
        # cutoff_bnds = [];
        OUT = SETUP_POPULATIONS_CHATTANOOGA(GRAPHS,cutoff_bnds = cutoff_bnds, params=params);
        PRE = OUT['PRE'];
        NODES = OUT['NODES']; LOCS = OUT['LOCS']; SIZES = OUT['SIZES']; 
        VEHS = OUT['VEHS']

        plotODs(GRAPHS,SIZES,NODES,scale=100.,figsize=(5,5))        

        # graph.graph ={'created_date': '2023-09-07 17:19:29','created_with': 'OSMnx 1.6.0','crs': 'epsg:4326','simplified': True}
        # FOR REFERENCE
        # {'osmid': 19496019, 'name': 'Benton Avenue',
        # 'highway': 'unclassified', 'oneway': False, 'reversed': True,
        # 'length': 377.384,
        # 'geometry': <LINESTRING (-85.21 35.083, -85.211 35.083, -85.211 35.083, -85.212 35.083, ...>}





    def find_close_node(node,graph,find_in_graph):
        """
        description -- takes node in one graph and finds the closest node in another graph
        inputs --
              node: node to find
              graph: initial graph node is given in
              find_in_graph: graph to find the closest node in
        returns --
              closest node in the find_in_graph
        """
        lon = graph.nodes[node]['x'];
        lat = graph.nodes[node]['y'];
        found_node = ox.distance.nearest_nodes(find_in_graph, lon,lat); #ORIG_LOC[i][0], ORIG_LOC[i][1]);
        xx = find_in_graph.nodes[found_node]['x'];
        yy = find_in_graph.nodes[found_node]['y'];
        if not(np.abs(xx-lon) + np.abs(yy-lat) <= 0.1):
            found_node = None;
        return found_node


    def find_close_node_gtfs_to_graph(stop,feed,graph):

        # print(list(feed.stops[feed.stops['stop_id']==stop].stop_lat))[0]
        # asdf
        # try: 

        # test1 = stop in list(feed.stops['stop_id'])
        # print(test1)
        # if test1==False:
        #     print(stop)

        lat = list(feed.stops[feed.stops['stop_id']==stop].stop_lat)
        lon = list(feed.stops[feed.stops['stop_id']==stop].stop_lon)
        # print(lat)
        # print(lon)
        lat = lat[0]
        lon = lon[0]
        # print(lat)
        # print(lon)
        # print(lat)
        # print(lon)
        found_node = ox.distance.nearest_nodes(graph, lon,lat)
            # print(found_node)
            #found_node = found_node; #ORIG_LOC[i][0], ORIG_LOC[i][1]);
        # except:
        #     found_node = 'XXXXXXXXXXXXX';

        # if isinstantce(found_node,list):
        #     found_node = found_node[0];
        return found_node

    def find_close_node_graph_to_gtfs(node,graph,feed):
        lon = graph.nodes[node]['x'];
        lat = graph.nodes[node]['y'];
        eps = 0.01;
        close = np.abs(feed.stops.stop_lat - lat) + np.abs(feed.stops.stop_lon - lon);
        close = close==np.min(close)
        found_stop = feed.stops.stop_id[close];
        found_stop = list(found_stop)[0]
        # if len(found_stop)==0:
        #     found_stop = None;
        #     #found_stop = found_stop[0];
        # if isinstance(found_stop,list):
        #     found_stop = found_stop[0];    
        return found_stop     

    # def find_close_node(node,graph,find_in_graph):
    #     lon = graph.nodes[node]['x'];
    #     lat = graph.nodes[node]['y'];
    #     found_node = ox.distance.nearest_nodes(find_in_graph, lon,lat); #ORIG_LOC[i][0], ORIG_LOC[i][1]);
    #     xx = find_in_graph.nodes[found_node]['x'];
    #     yy = find_in_graph.nodes[found_node]['y'];
    #     if not(np.abs(xx-lon) + np.abs(yy-lat) <= 0.1):
    #         found_node = None;
    #     return found_node





    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 
    ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD ######## LOAD 

    ########## ====================== LOADING DATA =================== ###################
    ########## ====================== LOADING DATA =================== ###################
    ########## ====================== LOADING DATA =================== ###################





def bndingBox():
    start = 8*60*60; end = 9*60*60;
    center_point = (-85.3094,35.0458)
    bnds = generate_bnds(center_point)
    return bnds


def generate_bnds(center_point):
    #center_point = (-85.3094,35.0458)
    cpt = np.array(center_point);
    dx_up = np.array([0.06,0.02]);
    dx_bot = np.array([-0.03,-0.1]);
    dxs = [dx_bot[0],dx_up[0]]; dys = [dx_bot[0],dx_up[1]];
    bnds = np.array([[dxs[1],dys[1]],[dxs[0],dys[1]],[dxs[0],dys[0]],[dxs[1],dys[0]]]);
    bnd_box = bnds + cpt; 
    top_bnd = cpt + dx_up; #bnd_box[0];
    bot_bnd = cpt + dx_bot; #bnd_box[2];
    bnds = [bot_bnd,top_bnd];
    return bnds





    ########## ====================== NODE CONVERSION =================== ###################
    ########## ====================== NODE CONVERSION =================== ###################
    ########## ====================== NODE CONVERSION =================== ###################

    
    # def findNode(self,node,from_type,to_type,NODES):
    #     out = None;
    #     if from_type == 'all':
    #         out = NODES['all'][to_type][node]
    #     if from_type == 'drive':
    #         out = NODES['drive'][to_type][node]
    #     if from_type == 'transit':
    #         out = NODES['transit'][to_type][node]
    #     if from_type == 'walk':
    #         out = NODES['walk'][to_type][node]
    #     if from_type == 'ondemand':
    #         out = NODES['ondemand'][to_type][node]
    #     return out        

class CONVERTER: 
    def __init__(self,params={}):

        self.modes = params['modes'];
        self.GRAPHS = params['GRAPHS']
        self.FEEDS = params['FEEDS']
        

        if 'NDF' in params: self.NDF = params['NDF']
        else: self.NDF = pd.DataFrame({mode:[] for mode in self.modes},index=[])
        if 'NINDS' in params: self.NINDS = params['NINDS'];
        else: self.NINDS = {}
        
        for mode in self.GRAPHS: self.NINDS[mode] = {};
        for mode in self.FEEDS: self.NINDS['feed_'+mode] = {};        


    def findClosestNode(self,node,from_mode,to_mode,from_type='graph',to_type='graph'):
        # node must be in GRAPHS[mode1]

        if from_type == 'graph': GRAPH1 = self.GRAPHS[from_mode];
        if to_type == 'graph': GRAPH2 = self.GRAPHS[to_mode];
        if from_type == 'feed': FEED1 = self.FEEDS[from_mode];
        if to_type == 'feed': FEED2 = self.FEEDS[to_mode];

        stop = node;
        if from_type == 'feed':#mode1 == 'gtfs':
            lat = list(FEED1.stops[FEED1.stops['stop_id']==stop].stop_lat)
            lon = list(FEED1.stops[FEED1.stops['stop_id']==stop].stop_lon)
            lat = lat[0]
            lon = lon[0]
        else:
            lon = GRAPH1.nodes[node]['x'];
            lat = GRAPH1.nodes[node]['y'];

        if to_type == 'feed':
            close = np.abs(FEED2.stops.stop_lat - lat) + np.abs(FEED2.stops.stop_lon - lon);
            close = close==np.min(close)
            found_stop = FEED2.stops.stop_id[close];
            found_node = list(found_stop)[0]
        else:
            found_node = ox.distance.nearest_nodes(GRAPH2, lon,lat); #ORIG_LOC[i][0], ORIG_LOC[i][1]);
            xx = GRAPH2.nodes[found_node]['x'];
            yy = GRAPH2.nodes[found_node]['y'];
            # if not(np.abs(xx-lon) + np.abs(yy-lat) <= 0.1):
            #     found_node = None;
        return found_node

    def addNodesByType(self,NODES,ndfs_to_rerun = ['gtfs','transit','delivery1','delivery2','source','target']):

        start_time = time.time()
        # if len(self.NDF) == 0:
        #     NDF = createEmptyNodesDF();

        if 'gtfs' in ndfs_to_rerun: 
            print('starting gtfs...')
            FEED = self.FEEDS['gtfs']
            for i,stop in enumerate(list(FEED.stops.stop_id)):
                if np.mod(i,100)==0: print(i)
                self.addNodeToConverter(stop,'gtfs',node_type='feed')

        if 'transit' in ndfs_to_rerun:
            print('starting transit nodes...')
            GRAPH = self.GRAPHS['gtfs'];
            for i,node in enumerate(list(GRAPH.nodes())):
                if np.mod(i,100)==0: print(i)
                self.addNodeToConverter(node,'gtfs',node_type='graph')
                # NDF = addNodeToConverter(node,'transit',GRAPHS,NDF)
        end_time = time.time()
        print('time to create nodes...: ',end_time-start_time)
                # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}        

        start_time = time.time()


        if 'delivery1' in ndfs_to_rerun:
            print('starting delivery1 sources...')
            for i,node in enumerate(NODES['delivery1']):
                if np.mod(i,200)==0: print(i)
                self.addNodeToConverter(node,'ondemand',node_type='graph')
                # NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)


        if 'delivery2' in ndfs_to_rerun:        
            print('starting delivery2 sources...')
            for i,node in enumerate(NODES['delivery2']):
                if np.mod(i,200)==0: print(i)
                self.addNodeToConverter(node,'ondemand',node_type='graph')
                # NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)
                

        if 'source' in ndfs_to_rerun:    
            print('starting source nodes...')
            for i,node in enumerate(NODES['orig']):
                if np.mod(i,200)==0: print(i)
                self.addNodeToConverter(node,'drive',node_type='graph')                    
                # NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)


        if 'target' in ndfs_to_rerun:        
            print('starting target nodes...')
            for i,node in enumerate(NODES['dest']):
                if np.mod(i,200)==0: print(i)
                self.addNodeToConverter(node,'drive',node_type='graph')                    
                # NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)
            
        end_time = time.time()
        print('time to create nodes...: ',end_time-start_time)
            # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}    
        # updateNodesDF(NDF);
        # return NDF    

    def addNodeToConverter(self,node,from_mode,node_type='graph'): #,GRAPHS,NODES):

        if not(node in self.NDF[from_mode]):
            node_index = 'node'+str(len(self.NDF));

            ########
            from_type = node_type;

            closest_nodes = {};
            #######
            if from_type == 'graph': self.NINDS[from_mode][node] = node_index;
            if from_type == 'feed':  self.NINDS['feed_'+from_mode][node] = node_index;

            for mode in self.GRAPHS:
                
                # closest_nodes[mode] = [self.findClosestNode(node,from_mode,mode,from_type = from_type,to_type='graph')];
                try: closest_nodes[mode] = [self.findClosestNode(node,from_mode,mode,from_type = from_type,to_type='graph')];
                except: pass
            for mode in self.FEEDS:
                # closest_nodes['feed_' + mode] = [self.findClosestNode(node,from_mode,mode,from_type=from_type,to_type='feed')];                
                try: closest_nodes['feed_' + mode] = [self.findClosestNode(node,from_mode,mode,from_type=from_type,to_type='feed')];
                except: pass
            new_nodes = pd.DataFrame(closest_nodes,index=[node_index])
            self.NDF = pd.concat([self.NDF,new_nodes]);


    def convertNode(self,node,from_mode,to_mode,from_type = 'graph',to_type = 'graph',verbose=False):
        """
        description -- converts nodes (at approx the same location) between two different graph types 
        inputs --
               node: node to convert
               from_type: initial mode node is given in
               to_type: desired node type
               NODES: node conversion dict - contains dataframes with conversion information
        returns --
               node in desired mode
        """
        #######
        from_mode2 = from_mode
        to_mode2 = to_mode;
        if from_type == 'feed': from_mode2 = 'feed_'+from_mode;
        if to_type == 'feed': to_mode2 = 'feed_'+to_mode
        # if not(from_mode2 in self.NINDS): self.addNodeToConverter(node,from_mode2,node_type=from_type):
        if not(node in self.NINDS[from_mode2]): self.addNodeToConverter(node,from_mode2,node_type=from_type)

        node_index = self.NINDS[from_mode2][node];
        to_node = self.NDF.loc[node_index][to_mode2]
        return to_node

    def setupNodeConversion(self):

        print('SETTING UP NODE CONVERSION:')

        # ndfs_to_rerun = ['gtfs','transit','delivery1','delivery2','source','target'];
        ndfs_to_rerun = ['delivery1','delivery2','source','target'];
        NDF = SETUP_NODESDF_CHATTANOOGA(GRAPHS,NODES,NDF=NDF,ndfs_to_rerun=ndfs_to_rerun)
        #BUS_STOP_NODES = INITIALIZING_BUSSTOPCONVERSION_CHATTANOOGA(GRAPHS);
        BUS_STOP_NODES = {};        

    #### ADDING NODES TO DATAFRAME
    def SETUP_NODESDF_CHATTANOOGA(self,GRAPHS,NODES,NDF=[],
        ndfs_to_rerun = ['gtfs','transit','delivery1','delivery2','source','target']):

        start_time = time.time()
        if len(NDF) == 0:
            NDF = createEmptyNodesDF();
        feed = GRAPHS['gtfs']

        if 'gtfs' in ndfs_to_rerun: 
            print('starting gtfs...')
            for i,stop in enumerate(list(feed.stops.stop_id)):
                if np.mod(i,100)==0: print(i)
                NDF = addNodeToConverter(stop,'gtfs',GRAPHS,NDF)

        if 'transit' in ndfs_to_rerun:
            print('starting transit nodes...')
            for i,node in enumerate(list(GRAPHS['transit'].nodes())):
                if np.mod(i,100)==0: print(i)
                NDF = addNodeToConverter(node,'transit',GRAPHS,NDF)
        end_time = time.time()
        print('time to create nodes...: ',end_time-start_time)
                # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}        

        start_time = time.time()


        if 'delivery1' in ndfs_to_rerun:
            print('starting delivery1 sources...')
            for i,node in enumerate(NODES['delivery1']):
                if np.mod(i,200)==0: print(i)
                NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)

        if 'delivery2' in ndfs_to_rerun:        
            print('starting delivery2 sources...')
            for i,node in enumerate(NODES['delivery2']):
                if np.mod(i,200)==0: print(i)
                NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)
                

        if 'source' in ndfs_to_rerun:    
            print('starting source nodes...')
            for i,node in enumerate(NODES['orig']):
                if np.mod(i,200)==0: print(i)
                NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)

        if 'target' in ndfs_to_rerun:        
            print('starting target nodes...')
            for i,node in enumerate(NODES['dest']):
                if np.mod(i,200)==0: print(i)
                NDF = addNodeToConverter(node,'drive',GRAPHS,NDF)
            
        end_time = time.time()
        print('time to create nodes...: ',end_time-start_time)
            # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}    
        updateNodesDF(NDF);
        return NDF    


def generate_segtypes(vers): # reg1,reg2,bg
    #### preloaded types of trips... 
    SEG_TYPES = {}
    if vers == 'reg1':
        SEG_TYPES['car_no'] = [('ondemand',),('walk','gtfs','walk'),('walk','gtfs','ondemand'),('ondemand','gtfs','walk'),('ondemand','gtfs','ondemand')];
        SEG_TYPES['car_opt'] = [('drive',),('ondemand',),('walk','gtfs','walk'),('walk','gtfs','ondemand'),('ondemand','gtfs','walk'),('ondemand','gtfs','ondemand')];
        SEG_TYPES['car_only'] = [('drive',)];
    elif vers == 'reg2':
        SEG_TYPES['car_no'] = [('ondemand',),('walk','gtfs','walk')];
        SEG_TYPES['car_opt'] = [('drive',),('ondemand',),('walk','gtfs','walk')];
        SEG_TYPES['car_only'] = [('drive',)];
    elif vers == 'reg3':
        SEG_TYPES['car_no'] = [('ondemand',),('walk','gtfs','walk')];
        SEG_TYPES['car_opt'] = [('ondemand',),('walk','gtfs','walk')];
        SEG_TYPES['car_only'] = [('ondemand',),('walk','gtfs','walk')];
    elif vers == 'reg4':
        temp = [('ondemand',),
                ('walk','gtfs','walk'),
                ('walk','gtfs','ondemand'),
                ('ondemand','gtfs','walk'),
                ('ondemand','gtfs','ondemand')];
        SEG_TYPES['car_no'] = temp 
        SEG_TYPES['car_opt'] = temp 
        SEG_TYPES['car_only'] = temp

    elif vers == 'reg5':
        temp = [('walk','gtfs','walk'),
                ('walk','gtfs','ondemand'),
                ('ondemand','gtfs','walk'),
                ('ondemand','gtfs','ondemand')];
        SEG_TYPES['car_no'] = temp 
        SEG_TYPES['car_opt'] = temp 
        SEG_TYPES['car_only'] = temp

    elif vers == 'reg6':
        temp = [('ondemand',),
                ('walk','gtfs','walk'),
                ('walk','gtfs','ondemand'),
                ('ondemand','gtfs','walk'),
                ('ondemand','gtfs','ondemand')];
        SEG_TYPES['car_no'] = temp 
        SEG_TYPES['car_opt'] = temp 
        SEG_TYPES['car_only'] = temp
    elif vers == 'reg7':
        temp = [('ondemand',),
                ('walk','gtfs','walk')];
        SEG_TYPES['car_no'] = temp 
        SEG_TYPES['car_opt'] = temp 
        SEG_TYPES['car_only'] = temp            

    elif vers == 'reg8':
        temp = [('ondemand',),
                ('walk','gtfs','walk'),
                ('ondemand','gtfs','walk')];
        SEG_TYPES['car_no'] = temp 
        SEG_TYPES['car_opt'] = temp 
        SEG_TYPES['car_only'] = temp            


    elif vers == 'bg':
        SEG_TYPES['car_no'] = [('walk','gtfs','walk')];
        SEG_TYPES['car_opt'] = [('drive',),('walk','gtfs','walk')];
        SEG_TYPES['car_only'] = [('drive',)];
    return SEG_TYPES


class WORLD:
    def __init__(self,params = {},full_setup = False,verbose=True,filename=None):





        if not(filename==None):
            file = open(filename, 'rb')
            DATA = pickle.load(file)
            DATA = pd.read_pickle(filename)
            self.LOADED = DATA
            file.close()

        if filename == None:
            self.verbose = verbose
            self.inputparams = params;
            self.main = {}
            self.main['iter'] = 1;
            self.main['alpha'] = 10./(self.main['iter']+1.);
            self.factors = ['time','money','conven','switches'];
            self.modes = params['modes'];
            self.bnds = params['bnds'];
        else: 
            self.verbose = self.LOADED['OTHER']['verbose']
            self.inputparams = self.LOADED['OTHER']['inputparams']
            self.main = self.LOADED['OTHER']['main']
            self.factors = self.LOADED['OTHER']['factors']
            self.modes = self.LOADED['OTHER']['modes']
            self.bnds = self.LOADED['OTHER']['bnds'];


        self.gtfs_feed_file = params['gtfs_feed_file'];
        self.gtfs_precomputed_file = params['gtfs_precomputed_file'];
        self.groups_regions_geojson = params['groups_regions_geojson']
        if 'background_congestion_file' in params: self.preloads_file = params['background_congestion_file'];

        ####### 
        self.NETWORKS = {}

        # for mode in self.modes:

        if full_setup == True: 
            # self.initGRAPHSnFEEDS2();
            self.initGRAPHSnFEEDS();
            self.initNETWORKS();
            self.initCONVERTER();
            self.initSTATS();
            self.initONDEMAND();
            self.initPEOPLE();
            self.initBACKGROUND();
            self.initUNCONGESTED()


    def initGRAPHSnFEEDS(self): #,verbose=True):
        ##### LOAD GRAPH/FEED OBJECTS ######
        self.GRAPHS = {};
        self.RGRAPHS = {}; ## REVERSE GRAPHS

        self.FEEDS = {};
        if 'gtfs' in self.modes:
            feed = gtfs.Feed(self.gtfs_feed_file, time_windows=[0, 6, 10, 12, 16, 19, 24]);
            self.FEEDS['gtfs'] = feed        

        for mode in self.modes:
            if self.verbose: print('loading graph/feed for',mode,'mode...')
            # self.NETWORKS[mode] = NETWORK(mode);
            if mode == 'gtfs':
                self.GRAPHS[mode] = self.graphFromFeed(self.FEEDS['gtfs']);#,start,end) ## NEW
            elif mode == 'ondemand':
                self.GRAPHS[mode] = ox.graph_from_place('chattanooga',network_type='drive'); #ox.graph_from_polygon(graph_boundary,network_type='drive')
            elif mode == 'drive' or mode == 'walk' or mode == 'bike':
                self.GRAPHS[mode] = ox.graph_from_place('chattanooga',network_type='drive'); #ox.graph_from_polygon(graph_boundary,network_type='drive')                

        if self.verbose: print('cutting graphs to boundaries...')
        if len(self.bnds) > 0:
            for mode in self.modes:
                self.GRAPHS[mode] = self.trimGraph(self.GRAPHS[mode],self.bnds)

        if self.verbose: print('composing graphs...')
        self.GRAPHS['all'] = nx.compose_all([self.GRAPHS[mode] for mode in self.modes]);
        if self.verbose: print('computing reverse graphs...')
        RGRAPHS = {};
        for i,mode in enumerate(self.GRAPHS):
            if self.verbose: print('...reversing',mode,'graph...')
            self.RGRAPHS[mode] = self.GRAPHS[mode].reverse();


    def initNETWORKS(self):
        for mode in self.modes:
            if self.verbose: print('constructing NETWORK ',mode,'mode...')
            params2 = {'graph':mode,'GRAPH':self.GRAPHS[mode]}

            if mode == 'gtfs':
                params2['gtfs_precomputed_file'] = self.gtfs_precomputed_file
            self.NETWORKS[mode] = NETWORK(mode,
                self.GRAPHS,
                self.FEEDS,
                params2);


    def initCONVERTER(self):
        params2 = {};
        if 'background' in self.inputparams: 
            data_filename = 'runs/data2524.obj';
            # data_filename = params['data_filename'];
            file = open(data_filename, 'rb')
            DATA = pickle.load(file)
            DATA = pd.read_pickle(filename)
            file.close()
            if reread_data:
                WORLD0 = DATA['WORLD'];
                params2['NDF'] = DATA['NDF']
                params2['NINDS'] = DATA['NINDS'];

        params2['modes'] = self.modes
        params2['GRAPHS'] = self.GRAPHS
        params2['FEEDS'] = self.FEEDS
        self.CONVERTER = CONVERTER(params = params2);


    def initSTATS(self):

        path2 = self.groups_regions_geojson
        group_polygons = generate_polygons('from_geojson',path = path2);
        self.group_polygons = group_polygons
        self.grpsDF = createGroupsDF(group_polygons);
        params3 = {'num_drivers':16,'am_capacity':8, 'wc_capacity':2,
                   'start_time' : 0,'end_time' : 3600*4.}
        self.driversDF = {group: createDriversDF(params3,WORLD) for group in list(self.grpsDF['group'])}
        self.DELIVERYDF = {'grps':self.grpsDF,'drivers':self.driversDF}              

    
        # def generatePopulation(GRAPHS,DELIVERY,WORLD,NODES,VEHS,LOCS,PRE,params,verbose=True):  ##### ADDED TO CLASS

        # modes = ['drive','transit','ondemand','walk','gtfs']
        # params = {}
        # params['modes'] = modes; params['graphs'] = graphs;
        # params['nodes'] = nodes; params['factors'] = factors; params['mass_scale'] = 1; # 4./3600.
        # asdf = generatePopulation(GRAPHS,DELIVERY,WORLD,NDF,VEHS,LOCS,PRE,params,verbose = True);

        # PEOPLE = asdf;
        # print('Number of agents:',num_people)
        # updateNodesDF(NDF);


        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 

        start = 8*60*60; end = 9*60*60;
        center_point = (-85.3094,35.0458)
        bnds = generate_bnds(center_point)

        print('LOADING POPULATION DATA:')
        params = {'pop_cutoff':1}
        params['SEG_TYPES'] = generate_segtypes('reg8') # reg7 reg1,reg2,bg


        cent_pt = np.array(center_point)
        # dest_shift = np.array([0.001,-0.000]);
        dest_shift = np.array([0.003,-0.005]);
        orig_shift = np.array([0.022,-0.04]);
        orig_shift2 = np.array([-0.015,-0.042]);
        orig_shift3 = np.array([0.035,-0.01]);

        thd = 0.3; tho = -0.0; tho2 = -0.0; tho3 = 0.0;
        Rd = np.array([[np.cos(thd),-np.sin(thd)],[np.sin(thd),np.cos(thd)]]);
        Ro = np.array([[np.cos(tho),-np.sin(tho)],[np.sin(tho),np.cos(tho)]]);
        Ro2 = np.array([[np.cos(tho2),-np.sin(tho2)],[np.sin(tho2),np.cos(tho2)]]);
        Ro3 = np.array([[np.cos(tho3),-np.sin(tho3)],[np.sin(tho3),np.cos(tho3)]]);
        COVd  = np.diag([0.0001,0.00007]);
        # COVd3  = np.diag([0.00004,0.00004]);
        COVo  = np.diag([0.00005,0.0001]);
        COVo2 = np.diag([0.00003,0.0001]);
        COVo3 = np.diag([0.00008,0.00008]);
        # COVd  = np.diag([0.000002,0.000002]);
        # COVo  = np.diag([0.000002,0.000002]);
        # COVo2 = np.diag([0.000002,0.000002]);
        COVd = Rd@COVd@Rd.T
        COVo = Ro@COVo@Ro.T
        COVo2 = Ro2@COVo2@Ro2.T
        COVo3 = Ro3@COVo3@Ro3.T

        params['OD_version'] = 'gauss';
        params['gauss_stats'] = [{'num':70,'pop':1,
                                 'dests':{'mean':cent_pt+dest_shift,'cov':COVd},
                                 'origs':{'mean':cent_pt+orig_shift,'cov':COVo}},
                                 {'num':30,'pop':1,
                                 'dests':{'mean':cent_pt+dest_shift,'cov':COVd},
                                 'origs':{'mean':cent_pt+orig_shift2,'cov':COVo2}},
                                {'num':70,'pop':1,
                                 'dests':{'mean':cent_pt+dest_shift,'cov':COVd},
                                 'origs':{'mean':cent_pt+orig_shift3,'cov':COVo3}}]

        params['num_deliveries'] = {'delivery1':20,'delivery2':20}

        params['eps_filterODs'] = 0.001
        cutoff_bnds = bnds;
        # cutoff_bnds = [];
        OUT = SETUP_POPULATIONS_CHATTANOOGA(self.GRAPHS,cutoff_bnds = cutoff_bnds, params=params);

        self.OUT = OUT;
        self.PRE = OUT['PRE'];
        self.NODES = OUT['NODES'];
        self.LOCS = OUT['LOCS'];
        self.SIZES = OUT['SIZES']; 
        self.VEHS = OUT['VEHS']

        # params2['modes'] = self.modes
        # params2['GRAPHS'] = self.GRAPHS
        # params2['FEEDS'] = self.FEEDS


    # def init1b(self):
  
        print('ADDING NODES BY TYPE...')
        self.CONVERTER.addNodesByType(self.NODES);


        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 



    def initONDEMAND(self):
        params2 = {};
        params2['LOCS']  = self.OUT['LOCS']
        params2['NODES'] = self.OUT['NODES']
        params2['GRAPHS'] = self.GRAPHS
        params2['FEEDS'] = self.FEEDS
        params2['CONVERTER'] = self.CONVERTER

        self.ONDEMAND = ONDEMAND(self.DELIVERYDF,params2);

        # PRE = self.PRE
        # NODES = self.NODES
        # LOCS = self.LOCS
        # SIZES = self.SIZES
        # VEHS = self.VEHS

    def initPEOPLE(self):

        people_tags = list(self.PRE); #params['people_tags']
        ORIG_LOC = self.LOCS['orig'] #params['ORIG_LOC'];
        DEST_LOC = self.LOCS['dest'] #params['DEST_LOC'];
        # modes = params['modes'];
        # graphs = params['graphs'];
        # nodes = params['nodes'];
        # factors = params['factors'];
        # mass_scale = params['mass_scale']


        print('GENERATING POPULATION OF',len(people_tags),'...')
        self.PEOPLE = {};
        for k,person in enumerate(self.PRE):#people_tags:
            if np.mod(k,10)==0: print('adding person',k,'...')
            self.PEOPLE[person] = PERSON(person,
                self.CONVERTER,
                self.GRAPHS,
                self.FEEDS,
                self.NETWORKS,
                self.ONDEMAND,
                self.PRE,{'modes':self.modes,'factors':self.factors})

    def initBACKGROUND(self):
        file = open(self.preloads_file, 'rb')
        DATA = pickle.load(file)
        file.close()
        self.PRELOADS = DATA.copy()
        self.add_base_edge_masses();


    def initUNCONGESTED(self):
        modes = ['drive','walk'];
        for mode in modes:
            NETWORK = self.NETWORKS[mode];
            NETWORK.computeUncongestedEdgeCosts();

        modes = ['drive','walk','gtfs','ondemand'];
        for mode in modes:
            NETWORK = self.NETWORKS[mode]
            NETWORK.computeUncongestedTripCosts();



    def plotPRELIMINARIES(self,include_demand_curves=False):
        fig,axs = plt.subplots(1,3,figsize=(12,4));
        plotODs(self.GRAPHS,self.SIZES,self.NODES,scale=100.,figsize=(5,5),ax=axs[0])
        plotShapesOnGraph(self.GRAPHS,self.group_polygons,figsize=(5,5),ax=axs[1]);
        if include_demand_curves:
            self.ONDEMAND.plotCongestionModels(axs[2])





    def fitModels(self,counts = {'num_counts':2,'num_per_count':1}):


        print('updating individual choices...')


        for i,person in enumerate(self.PEOPLE):
            PERSON = self.PEOPLE[person]
            if np.mod(i,200)==0: print(person,'...')
            PERSON.UPDATE(self.NETWORKS,takeall=True)

        NETWORK = self.NETWORKS['ondemand'];
        self.ONDEMAND.generateCongestionModels(NETWORK,counts = counts,verbose=self.verbose); 

    # def add_base_edge_masses(self,WORLD0): #GRAPHS,NETWORKS,WORLD0):
    #     modes = []
    #     for i,mode in enumerate(self.NETWORKS):
    #         if not(mode=='main' or mode=='transit' or mode=='gtfs'):
    #             self.NETWORKS[mode].base_edge_masses = {};
    #             for e,edge in enumerate(self.GRAPHS[mode].edges):
    #                 if edge in WORLD0[mode]['current_edge_masses']:
    #                     self.NETWORKS[mode]['base_edge_masses'][edge] = WORLD0[mode]['current_edge_masses'][edge];


    def add_base_edge_masses(self): #GRAPHS,NETWORKS,WORLD0):                        
        modes = ['drive'];
        for i,mode in enumerate(modes):
            self.NETWORKS[mode].base_edge_masses = {};
            for e,edge in enumerate(self.GRAPHS[mode].edges):
                if edge in self.PRELOADS[mode]['current_edge_masses']:
                    self.NETWORKS[mode].base_edge_masses[edge] = self.PRELOADS[mode]['current_edge_masses'][edge];





    #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS 
    #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS 
    #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS 
    #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS #### GRAPH LOADING FUNCTIONS 
                        
    def graphFromFeed(self,feed):  ###### ADDED TO CLASS 
        """
        DESCRIPTION:  takes gtfs feed and creates networkx.MultiDiGraph
        INPUTS:
        - feed: GTFS feed
        OUTPUTS:
        - graph: output networkx graph
        """

        graph = nx.MultiDiGraph() # creating empty graph structure
        stop_list = list(feed.stops['stop_id']); # getting list of transit stops (to become nodes)
        lon_list = list(feed.stops['stop_lon']); # getting lists of longitudes for stops, ie. 'x' values.
        lat_list = list(feed.stops['stop_lat']); # getting lists of latitudes for stops, ie. 'y' values.

        for i,stop in enumerate(stop_list): # looping through stops
            lat = lat_list[i]; lon = lon_list[i];
            graph.add_node(stop,x=lon,y=lat); #adding node with the correct x,y coordinates
        
        starts_list = list(feed.segments['start_stop_id']); # getting list of "from" stops for each transit segment 
        stops_list = list(feed.segments['end_stop_id']); # getting list of "to" stops for each transit segment
        geoms_list = list(feed.segments['geometry']); # getting geometries for each transit 
        
        for i,start in enumerate(starts_list): # looping through segements
            stop = stops_list[i];
            geom = geoms_list[i];
            graph.add_edge(start, stop,geometry=geom);  #adding edge for each segment with geometry indicated
        #######  graph.graph = {'created_date': '2023-09-07 17:19:29','created_with': 'OSMnx 1.6.0','crs': 'epsg:4326','simplified': True}
        graph.graph = {'crs': 'epsg:4326'} # not sure what this does but necessary 


        #### OLD... CHANGING THE BUS GRAPH SOME....
        print('connecting close bus stops...')
        graph_bus_wt = graph.copy();
        bus_nodes = list(graph_bus_wt.nodes);
        print('Original num of edges: ', len(graph_bus_wt.edges))
        for i in range(len(bus_nodes)):
            for j in range(len(bus_nodes)):
                node1 = bus_nodes[i]
                node2 = bus_nodes[j]
                x1 = graph_bus_wt.nodes[node1]['x']
                y1 = graph_bus_wt.nodes[node1]['y']        
                x2 = graph_bus_wt.nodes[node2]['x']
                y2 = graph_bus_wt.nodes[node2]['y']
                diff = np.array([x1-x2,y1-y2]);
                dist = mat.norm(diff)
                if (dist==0):
                    graph_bus_wt.add_edge(node1,node2)
                    graph_bus_wt.add_edge(node2,node1)
                #GRAPHS['bus'].add_edge(node1,node2) #['transfer'+str(i)+str(j)] = (node1,node2,0);
        print('Final num of edges: ', len(graph_bus_wt.edges))
        graph = graph_bus_wt.copy()
        return graph

    def trimGraph(self,graph,bnds): ### ADDED TO CLASS
        nodes1 = list(graph);
        nodes2 = [];
        bot_bnd = bnds[0]; top_bnd = bnds[1];
        for i,node in enumerate(nodes1):
            x = graph.nodes[node]['x'];
            y = graph.nodes[node]['y'];
            loc = np.array([x,y]);
            cond1 = (loc[0] <= top_bnd[0]) and (loc[1] <= top_bnd[1]);
            cond2 = (loc[0] >= bot_bnd[0]) and (loc[1] >= bot_bnd[1]);
            if cond1 & cond2: nodes2.append(node);
        # graph1 = graph.subgraph(nodes2)
        graph1 = nx.subgraph(graph,nodes2)
        return graph1


    def generateDashboard(self):
        pass
    def printDashboard(self):
        pass


    ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA 
    ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA 

    def RUN(self,num_iters = 1,restart=True):


        # self.add_base_edge_masses(self.GRAPHS,self.NETWORKS,WORLD0);

        mode = 'ondemand'
        # poly = np.array([-6120.8676711, 306.5130127])
        # poly = np.array([5047.38255623, -288.78570445,    6.31107635]); # 2nd order

        # poly = np.array([696.29355592, 10.31124288])
        # poly = np.array([406.35315058,  18.04891652]);
        # WORLD['ondemand']['poly'] = poly
        # poly = WORLD['ondemand']['fit']['poly']
        # pop_guess = 50.;
        # exp_cost = poly[0] + poly[1]*pop_guess; # + poly[2]*(pop_guess*pop_guess);

        for _,group in enumerate(self.ONDEMAND.groups):
            GROUP = self.ONDEMAND.groups[group]
            poly = GROUP.fit['poly']
            pop_guess = 25.;
            exp_cost = poly[0] + poly[1]*pop_guess; # + poly[2]*(pop_guess*pop_guess);
            GROUP.expected_cost = [exp_cost];
            GROUP.actual_average_cost = [0];
            GROUP.current_expected_cost = exp_cost;

        self.main['start_time'] = 0;
        self.main['end_time'] = 3600*4.;    

        if restart:
            self.main['iter'] = 0.;
            self.main['alpha'] = 10./(self.main['iter']+1.);
        else:
            # self.main['iter'] = 0.;
            self.main['alpha'] = 10./(self.main['iter']+1.);



        nk = num_iters;
        # nk = 5; 

        if restart: 
            print('------------ Planning initial trips... ------------')
            for i,person in enumerate(self.PEOPLE):
                PERSON = self.PEOPLE[person]
                if np.mod(i,200)==0: print(person,'...')
                PERSON.UPDATE(self.NETWORKS,takeall=True)


        # nk = 2
        # GRADIENT DESCENT...
        for k in range(nk):
            start_time = time.time();
            print('------------------ITERATION',int(self.main['iter']),'-----------')
            # alpha =1/(k+10.);

            clear_active=True;
            if k == nk-1: clear_active=False;

            params2 = {'iter':self.main['iter'],'alpha':self.main['alpha']};



            self.NETWORKS['gtfs'].UPDATE(params2,self.FEEDS['gtfs'],self.ONDEMAND,verbose=True,clear_active=clear_active); # WORLD,PEOPLE,GRAPHS,NDF,verbose=True,clear_active=clear_active);    
            self.NETWORKS['drive'].UPDATE(params2,self.FEEDS['gtfs'],self.ONDEMAND,verbose=True,clear_active=clear_active); #WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            self.NETWORKS['ondemand'].UPDATE(params2,self.FEEDS['gtfs'],self.ONDEMAND,verbose=True,clear_active=clear_active); #WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=True,show_delivs='all',clear_active=clear_active);
            self.NETWORKS['walk'].UPDATE(params2,self.FEEDS['gtfs'],self.ONDEMAND,verbose=True,clear_active=clear_active); #WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
                

            # world_of_gtfs(WORLD,PEOPLE,GRAPHS,NDF,verbose=True,clear_active=clear_active);    
            # world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            # world_of_ondemand(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=True,show_delivs='all',clear_active=clear_active);
            # world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            #world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=True);



            

            print('updating individual choices...')


            for i,person in enumerate(self.PEOPLE):
                PERSON = self.PEOPLE[person]
                if np.mod(i,200)==0: print(person,'...')
                PERSON.UPDATE(self.NETWORKS)
                # self.PEOPLE.UPDATE(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=False);
            # update_choices(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=False);


            end_time = time.time()
            
            print('iteration time: ',end_time-start_time)
            self.main['iter'] = self.main['iter'] + 1.;
            self.main['alpha'] = 10./(self.main['iter']+1.);
            # WORLD['main']['alpha'] = 10.;



#### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK 
#### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK 
#### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK 
#### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK 
#### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK 
#### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK #### RUN FROM NOTEBOOK 


# # INITIALIZE SIM...  runs one step of main loop.   -- ADDED TO CLASS
# # INITIALIZE SIM...  runs one step of main loop.   -- ADDED TO CLASS
# # INITIALIZE SIM...  runs one step of main loop.   -- ADDED TO CLASS
# # INITIALIZE SIM...  runs one step of main loop.   -- ADDED TO CLASS

# %load_ext autoreload
# %autoreload 2
# from multimodal_functions import * 

# poly = np.array([406.35315058,  18.04891652]);
# WORLD['ondemand']['poly'] = poly


# nk = 1; takeall = True;
# # GRADIENT DESCENT...
# for k in range(nk):
#     start_time = time.time();
#     print('------------------ITERATION',int(WORLD['main']['iter']),'-----------')
#     # alpha =1/(k+10.);

#     clear_active=True;
#     if k == nk-1: clear_active=False;
        
#     world_of_gtfs(WORLD,PEOPLE,GRAPHS,NDF,verbose=True,clear_active=clear_active);    
#     world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
#     world_of_ondemand(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=True,show_delivs='all',clear_active=clear_active);
#     world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
#     #world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=True);
    
#     print('updating individual choices...')
#     update_choices(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=takeall);
#     end_time = time.time()
    
#     print('iteration time: ',end_time-start_time)
#     WORLD['main']['iter'] = WORLD['main']['iter'] + 1.;
#     WORLD['main']['alpha'] = 10./(WORLD['main']['iter']+1.);
#     # WORLD['main']['alpha'] = 10.;


# ## INITIALIZE: --- ADDED TO CLASS 
# ## INITIALIZE: --- ADDED TO CLASS 
# ## INITIALIZE: --- ADDED TO CLASS 
# ## INITIALIZE: --- ADDED TO CLASS 


# add_base_edge_masses(GRAPHS,WORLD,WORLD0);

# mode = 'ondemand'
# # poly = np.array([-6120.8676711, 306.5130127])
# # poly = np.array([5047.38255623, -288.78570445,    6.31107635]); # 2nd order

# # poly = np.array([696.29355592, 10.31124288])
# # poly = np.array([406.35315058,  18.04891652]);
# # WORLD['ondemand']['poly'] = poly
# # poly = WORLD['ondemand']['fit']['poly']
# # pop_guess = 50.;
# # exp_cost = poly[0] + poly[1]*pop_guess; # + poly[2]*(pop_guess*pop_guess);

# for _,group in enumerate(DELIVERY['groups']):
#     poly = DELIVERY['groups'][group]['fit']['poly']
#     pop_guess = 25.;
#     exp_cost = poly[0] + poly[1]*pop_guess; # + poly[2]*(pop_guess*pop_guess);
#     DELIVERY['groups'][group]['expected_cost'] = [exp_cost];
#     DELIVERY['groups'][group]['actual_average_cost'] = [0];
#     DELIVERY['groups'][group]['current_expected_cost'] = exp_cost;

# WORLD['main']['iter'] = 0.;
# WORLD['main']['alpha'] = 10./(WORLD['main']['iter']+1.);

# WORLD['main']['start_time'] = 0;
# WORLD['main']['end_time'] = 3600*4.;    

# nk = 5; 



# ## RUN  --- ADDED TO CLASS 
# ## RUN  --- ADDED TO CLASS 
# ## RUN  --- ADDED TO CLASS 
# ## RUN  --- ADDED TO CLASS 

# # nk = 2
# # GRADIENT DESCENT...
# for k in range(nk):
#     start_time = time.time();
#     print('------------------ITERATION',int(WORLD['main']['iter']),'-----------')
#     # alpha =1/(k+10.);

#     clear_active=True;
#     if k == nk-1: clear_active=False;
        
#     world_of_gtfs(WORLD,PEOPLE,GRAPHS,NDF,verbose=True,clear_active=clear_active);    
#     world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
#     world_of_ondemand(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=True,show_delivs='all',clear_active=clear_active);
#     world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
#     #world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=True);
    

#     print('updating individual choices...')
#     update_choices(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=False);
#     end_time = time.time()
    
#     print('iteration time: ',end_time-start_time)
#     WORLD['main']['iter'] = WORLD['main']['iter'] + 1.;
#     WORLD['main']['alpha'] = 10./(WORLD['main']['iter']+1.);
#     # WORLD['main']['alpha'] = 10.;


    def generateOutputs(self):

        colheaders = ['people']
        colheaders = colheaders + ['time','money','switches','conven'];
        colheaders = colheaders + ['time0','money0','switches0','conven0'];
        colheaders = colheaders + ['start_node','end_node'];
        colheaders = colheaders + ['active']
        
        colheaders2 = colheaders + ['runid','group','num_passengers']
        COLHEADERS = {tag:[] for tag in colheaders};
        COLHEADERS2 = {tag:[] for tag in colheaders2};

        self.OUTPUTS2 = {}; #'people':{},'drive':{},'walk':{},'ondemand':{},'gtfs':{}}
        for _,mode in enumerate(self.modes):
            if 'ondemand' == mode: self.OUTPUTS2[mode] = {'DF': pd.DataFrame(COLHEADERS2,index=[])}
            else: self.OUTPUTS2[mode] = {'DF': pd.DataFrame(COLHEADERS2,index=[])}

        for _,mode in enumerate(self.modes):
            DF = self.OUTPUTS2[mode]['DF'];
            active_segs = self.NETWORKS[mode].active_segs;
            all_segs = list(self.NETWORKS[mode].segs);
            SEGS = self.NETWORKS[mode].segs

            for seg in SEGS:
                SEG = SEGS[seg];
                
                NEWDATA = {};
                if hasattr(SEG,'people'): NEWDATA['people'] = SEG['people']
                else: NEWDATA['people'] = [None];
                    
                NEWDATA['start_node'] = [seg[0]];
                NEWDATA['end_node'] = [seg[1]];

                if seg in active_segs: NEWDATA['active'] = True;
                else: NEWDATA['active'] = False;

                NEWDATA['time'] = [SEG.costs['current_time']]
                NEWDATA['money'] = [SEG.costs['current_money']]
                NEWDATA['switches'] = [SEG.costs['current_switches']]
                NEWDATA['conven'] = [SEG.costs['current_conven']]
                if hasattr(SEG,'uncongested'):
                    if 'costs' in SEG.uncongested:
                        NEWDATA['time0'] = [SEG.uncongested['costs']['time']]
                    if 'costs' in SEG.uncongested:
                        NEWDATA['money0'] = [SEG.uncongested['costs']['money']]
                else: 
                    NEWDATA['time0'] = NEWDATA['time']
                    NEWDATA['money0'] = NEWDATA['money']
                
                if mode == 'ondemand':
                    if hasattr(SEG,'group'): NEWDATA['group'] = [SEG.group];
                    if hasattr(SEG,'run_id'): NEWDATA['runid'] = [SEG.run_id];
                    if hasattr(SEG,'num_passengers'): NEWDATA['num_passengers'] = [SEG.num_passengers];

                NEWDF = pd.DataFrame(NEWDATA)
                DF = pd.concat([DF,NEWDF], ignore_index = True)            
            self.OUTPUTS2[mode]['DF'] = DF;

        ####################################################################################################
        ####################################################################################################

        self.OUTPUTS = {}; #'PEOPLE':None,'ONDEMAND':None};

        colheaders = ['start_node','end_node','people','active']
        colheaders = colheaders + ['mode','distance','time','money','switches'];
        colheaders = colheaders + ['uncongested_distance','uncongested_time']
        colheaders = colheaders + ['group','run_id','num_passengers']
        colheaders = colheaders + ['pickup_time','dropoff_time','num_other_passengers']
        COLHEADERS = {tag:[] for tag in colheaders};

        for _,mode in enumerate(self.modes):
            DF = pd.DataFrame(COLHEADERS,index=[])
            active_segs = self.NETWORKS[mode].active_segs;
            all_segs = list(self.NETWORKS[mode].segs);
            SEGS = self.NETWORKS[mode].segs
            for seg in SEGS:
                SEG = SEGS[seg];

                NEWDATA = {};
                if hasattr(SEG,'people'): NEWDATA['people'] = SEG['people']
                else: NEWDATA['people'] = [None];
                NEWDATA['start_node'] = [seg[0]];
                NEWDATA['end_node'] = [seg[1]];
                if seg in active_segs: NEWDATA['active'] = True;
                else: NEWDATA['active'] = False;

                NEWDATA['mode'] = [mode]
                NEWDATA['time'] = [SEG.costs['current_time']]
                NEWDATA['money'] = [SEG.costs['current_money']]
                NEWDATA['switches'] = [SEG.costs['current_switches']]
                # NEWDATA['conven'] = [SEG.costs['current_conven']]
                if hasattr(SEG,'uncongested'):
                    if 'costs' in SEG.uncongested:
                        NEWDATA['uncongested_time'] = [SEG.uncongested['costs']['time'][-1]]
                    # NEWDATA['uncongested_money'] = [SEG.uncongested['costs']['money']]
                else: 
                    NEWDATA['uncongested_time'] = NEWDATA['time']
                    # NEWDATA['uncongested_money'] = NEWDATA['money']
                if mode == 'ondemand':
                    if hasattr(SEG,'group'): NEWDATA['group'] = [SEG.group];
                    if hasattr(SEG,'run_id'): NEWDATA['runid'] = [SEG.run_id];
                    if hasattr(SEG,'num_passengers'): NEWDATA['num_passengers'] = [SEG.num_passengers];
                    if hasattr(SEG,'pickup_time_window_start'): NEWDATA['pickup_time'] = [SEG.pickup_time_window_start]
                    if hasattr(SEG,'dropoff_time_window_start'): NEWDATA['dropoff_time'] = [SEG.dropoff_time_window_start]
                    NEWDATA['num_other_passengers'] = 0; 



                NEWDF = pd.DataFrame(NEWDATA)
                DF = pd.concat([DF,NEWDF], ignore_index = True)            
        self.OUTPUTS['PEOPLE'] = DF.copy();


        ####################################################################################################
        ####################################################################################################

        # colheaders = ['driver_run_id','group'];
        # colheaders = colheaders + ['distance','time']
        # colheaders = colheaders + ['total_passengers','time_wpassengers','distance_wpassengers'];
        # COLHEADERS = {tag:[] for tag in colheaders};

        # for _,group in enumerate(self.ONDEMAND.groups):
        #     GROUP = 
        #     for _,run_id in enumerate(self.)
        #  #mode in enumerate(self.modes):
        #     DF = pd.DataFrame(COLHEADERS,index=[])
        #     active_trips = self.NETWORKS[mode]['active_segs'];
        #     all_trips = list(self.NETWORKS[mode]['segs']);
        #     SEGS = self.NETWORKS[mode].segs
        #     for seg in SEGS:
        #         SEG = SEGS[seg];

        #         NEWDATA = {};
        #         if 'people' in SEG: NEWDATA['people'] = SEG['people']
        #         else: NEWDATA['people'] = [None];
        #         NEWDATA['start_node'] = [seg[0]];
        #         NEWDATA['end_node'] = [seg[1]];
        #         if seg in active_segs: NEWDATA['active'] = True;
        #         else: NEWDATA['active'] = False;

        #         NEWDATA['mode'] = [mode]
        #         NEWDATA['time'] = [SEG.costs['current_time']]
        #         NEWDATA['money'] = [SEG.costs['current_money']]
        #         NEWDATA['switches'] = [SEG.costs['current_switches']]
        #         # NEWDATA['conven'] = [SEG.costs['current_conven']]
        #         if 'uncongested' in SEG:
        #             NEWDATA['uncongested_time'] = [SEG.uncongested['costs']['time']]
        #             # NEWDATA['uncongested_money'] = [SEG.uncongested['costs']['money']]
        #         else: 
        #             NEWDATA['uncongested_time'] = NEWDATA['time']
        #             # NEWDATA['uncongested_money'] = NEWDATA['money']
        #         if mode == 'ondemand':
        #             if 'group' in SEG: NEWDATA['group'] = [SEG.group];
        #             if 'run_id' in SEG: NEWDATA['runid'] = [SEG.run_id];
        #             if 'num_passengers' in SEG: NEWDATA['num_passengers'] = [SEG.num_passengers];
        #             if 'pickup_time_window_start' in SEG: NEWDATA['pickup_time'] = self.pickup_time_window_start
        #             if 'dropoff_time_window_start' in SEG: NEWDATA['dropoff_time'] = self.dropoff_time_window_start
        #             NEWDATA['num_other_passengers'] = 0; 



        #         NEWDF = pd.DataFrame(NEWDATA)
        #         DF = pd.concat([DF,NEWDF], ignore_index = True)                    



    def SAVE(self,filename):

        fileObj = open(filename, 'wb')                    
        DATA = {};
        # if hasattr(self,'CONVERTER'): DATA['CONVERTER'] = self.CONVERTER
        if hasattr(self,'NODES'): DATA['NODES'] = self.NODES
        if hasattr(self,'LOCS'): DATA['LOCS'] = self.LOCS
        if hasattr(self,'PRE'): DATA['PRE'] = self.PRE
        if hasattr(self,'SIZES'): DATA['SIZES'] = self.SIZES
        if hasattr(self,'OUTPUTS'): DATA['OUTPUTS'] = self.OUTPUTS


        if hasattr(self,'PEOPLE'): DATA['PEOPLE'] = self.PEOPLE   
        # if hasattr(self,'GRAPHS'): DATA['GRAPHS'] = self.GRAPHS
        # if hasattr(self,'NETWORKS'): DATA['NETWORKS'] = self.NETWORKS
        # if hasattr(self,'ONDEMAND'): DATA['ONDEMAND'] = self.ONDEMAND
                
        pickle.dump(DATA,fileObj)
        fileObj.close()



    def LOAD(self):

        reload_data = True;
        #filename = 'data/data1176.obj'
        # filename = 'data/data1073.obj'
        # filename = 'data/data353.obj'
        # filename = 'data/data103.obj'
        # filename = 'data/small_data287.obj'
        # filename = 'data/small_data228_select.obj'
        # filename = 'runs/small_data233_select.obj'
        # filename = 'runs/small_data306_blank.obj'
        # filename = 'runs/small_data233_select.obj'
        # filename = 'runs/small_data153_blank.obj'
        # filename = 'runs/small_data93_blank.obj'
        # filename = 'runs/small_data51_blank.obj'
        # filename = 'runs/small_data51_full.obj'
        # filename = 'runs/small_data154_full.obj'
        # group_version = 'tiny1';
        filename = 'runs/'+group_version+'b.obj'; 

        # import pandas as pd

        # df = pd.read_pickle("file.pkl")

        if reload_data:
            #feed = pt.get_representative_feed('carta_gtfs.zip') #loading gtfs from chattanooga
            #feed = gtfs.Feed('carta_gtfs.zip', time_windows=[0, 6, 10, 12, 16, 19, 24])

            feed = gtfs.Feed('data/gtfs/carta_gtfs.zip',time_windows=[0, 6, 10, 12, 16, 19, 24])
            feed_details = {'routes': feed.routes,'trips': feed.trips,'stops': feed.stops,'stop_times': feed.stop_times,'shapes':feed.shapes}
            
            file = open(filename, 'rb')
            DATA = pickle.load(file)
            DATA = pd.read_pickle(filename)
            file.close()
            
        reread_data = True;
        if reread_data:
            asdf = DATA['PEOPLE']
            WORLD = DATA['WORLD']
            DELIVERY = DATA['DELIVERY']
            NDF = DATA['NDF']
            #GRAPHS = DATA['GRAPHS']
            PRE = DATA['PRE'];
            BUS_STOP_NODES = DATA['BUS_STOP_NODES']
            NODES = DATA['NODES']
            LOCS = DATA['LOCS']    
            SIZES = DATA['SIZES']
        GRAPHS['gtfs'] = feed;
        # GRAPHS['gtfs_details'] = feed_details;

        # for i,tag in enumerate(PEOPLE):
        #     # PEOPLE[tag]['mass_total'] = PEOPLE[tag]['mass']
        #     # PEOPLE[tag]['mass'] = 4*PEOPLE[tag]['mass_total']/(3600);
        #     print(PEOPLE[tag]['mass'])
        assign_people = True;
        if assign_people: 
            PEOPLE = asdf;























    ########## ====================== GRAPH BASIC =================== ###################
    ########## ====================== GRAPH BASIC =================== ###################
    ########## ====================== GRAPH BASIC =================== ###################

    def nodesFromTrips(trips): ### ADDED TO CLASS
        nodes = [];
        for i,trip in enumerate(trips):
            node0 = trip[0]; node1 = trip[1];
            if not(node0 in nodes):
                nodes.append(node0);
            if not(node1 in nodes):
                nodes.append(node1);
        return nodes

    def edgesFromPath(path): ### ADDED TO CLASS
        """ computes edges in a path from a list of nodes """
        edges = []; # initializing list
        for i,node in enumerate(path): # loops through nodes in path
            if i<len(path)-1: # if not the last node
                node1 = path[i]; node2 = path[i+1]; 
                edges.append((node1,node2)) # add an edge tag defined as (node1,node2) which is the standard networkx structure
        return edges

    def locsFromNodes(nodes,GRAPH): ### ADDED TO CLASS
        """ returns a list of locations from a list of nodes in a graph"""
        out = []
        for i,node in enumerate(nodes):
            out.append([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]);
        out = np.array(out)
        return out




    def nearest_nodes(mode,GRAPHS,NODES,x,y): ### ADDED TO CLASS
        if mode == 'gtfs':
            node = ox.distance.nearest_nodes(GRAPHS['transit'], x,y);
            out = WORLD.CONVERTER.convertNode(node,'transit','gtfs')
            # print(out)
        else:

            out = ox.distance.nearest_nodes(GRAPHS[mode], x,y);
        return out




           



def generatePolygons(self):

    # group_polygons = generate_polygons('reg2',center_point);
    # group_version = 'huge1';
    # group_version = 'large1';
    group_version = 'regions2'
    # group_version = 'medium1';
    # group_version = 'small1';
    # group_version = 'regions11';
    path2 = './DAN/group_sections/'+group_version+'.geojson'; #'/map.geojson'
    group_polygons = generate_polygons('from_geojson',path = path2);


    plotShapesOnGraph(GRAPHS,group_polygons,figsize=(10,10));    


    # WORLD['ondemand']['people'] = people_tags.copy();
    #  #people_tags.copy();
    # # WORLD['ondemand+transit']['people'] = people_tags.copy();
    # WORLD['ondemand']['trips'] = {};



# def nearest_applicable_gtfs_node(mode,gtfs_target,GRAPHS,WORLD,NDF,x,y,radius=0.25/69.):
def nearest_applicable_gtfs_node(mode,GRAPHS,NETWORK,CONVERTER,x1,y1,x2,y2):#,rad1=0.25/69.,rad2=0.25/69.):  ### ADDED TO CLASS
    ### ONLY WORKS IF 'gtfs' and 'transit' NODES ARE THE SAME...
    rad_miles = 2.;
    rad1=rad_miles/69.; rad2=rad_miles/69.;

    #(69 mi/ 1deg)*(1 hr/ 3 mi)*(3600 s/1 hr)
    walk_speed = (69./1.)*(1./3.)*(3600./1.)
    GRAPH = GRAPHS['gtfs'];
    # PRECOMPUTE = NETWORK.precompute.
    REACHED = NETWORK.precompute['reached'];
    # print(REACHED)
    close_start_nodes = [];
    start_dists = []
    close_end_nodes = [];
    end_dists = [];
    gtfs_costs = [];
    for i,node in enumerate(GRAPH.nodes):
        NODE = GRAPH.nodes[node];
        dist = mat.norm([x1 - NODE['x'],y1 - NODE['y']])
        if dist < rad1:
            close_start_nodes.append(node);
            start_dists.append(dist.copy());
        dist = mat.norm([x2 - NODE['x'],y2 - NODE['y']])
        if dist < rad2:
            close_end_nodes.append(node);
            end_dists.append(dist.copy());


    num_starts = len(close_start_nodes);
    num_ends = len(close_end_nodes);
    DISTS = np.zeros([num_starts,num_ends]);
    for i,start_node in enumerate(close_start_nodes):
        for j,end_node in enumerate(close_end_nodes):
            try:
                DISTS[i,j] = REACHED[start_node][-1][end_node];
                DISTS[i,j] = DISTS[i,j] + walk_speed*(start_dists[i] + end_dists[j])
            except:
                DISTS[i,j] = 1000000000000.;
    inds = np.where(DISTS == np.min(DISTS.flatten()))
    i = inds[0][0]; j = inds[1][0];



    start_node = close_start_nodes[i]
    end_node = close_end_nodes[j]

    # print('start node is',start_node,'and end node is ',end_node)
    return start_node,end_node




def generate_polygons(vers,center_point=[],path=''):
    if vers == 'reg1':

        x0 = -2.7; x1 = 0.5; x1b = 2.; x2 = 6.;
        y0 = -6.2; y1 = -2.; y2 = 1.5;
        pts1 = 0.01*np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])+center_point;
        pts2 = 0.01*np.array([[x1,y0],[x2,y0],[x2,y1],[x1,y1]])+center_point;
        pts3 = 0.01*np.array([[x0,y1],[x1b,y1],[x1b,y2],[x0,y2]])+center_point;
        pts4 = 0.01*np.array([[x1b,y1],[x2,y1],[x2,y2],[x1b,y2]])+center_point;
        polygons=[pts1,pts2,pts3,pts4]

    if vers == 'reg2':
        x0 = -2.7; x1 = 0.5; x1b = 2.; x2 = 6.;
        y0 = -6.2; y1 = -2.; y2 = 1.5;
        pts1 = 0.01*np.array([[x0,y0],[x2,y0],[x2,y1],[x0,y1]])+center_point;
        pts2 = 0.01*np.array([[x0,y1],[x2,y1],[x2,y2],[x0,y2]])+center_point;        
        polygons=[pts1,pts2]

    if vers == 'from_geojson':
        polygons = [];
        # path2 = './DAN/group_sections/small1/map.geojson'
        dataframe = gpd.read_file(path);
        for i in range(len(dataframe)):
            geoms = dataframe.iloc[i]['geometry'].exterior.coords;
            polygons.append(np.array([np.array(geom) for geom in geoms]))
    return polygons



class ONDEMAND:

    # def addFits(self,poly):
    #     self.fit = {'poly':poly}
    #     for i,group in enumerate(self.groups):
    #         GROUP = self.groups[group]
    #         GROUP.addFit(poly)

    def __init__(self,DELIVERYDF,params):


        # poly = np.array([-6120.8676711, 306.5130127])
        # poly = np.array([5047.38255623, -288.78570445,    6.31107635]); # 2nd order
        # poly = np.array([696.29355592, 10.31124288])
        poly = np.array([406.35315058,  18.04891652]);
        self.fit = {'poly':np.array([406.35315058,  18.04891652])};


        if 'driver_info' in params: driver_info = params['driver_info'];
        if 'center_point' in params: center_point = params['center_point'];
        else: center_point: center_point = (-85.3094,35.0458)


        LOCS = params['LOCS']
        NODES = params['NODES']


        self.booking_ids = [];


        self.GRAPHS = params['GRAPHS']
        self.FEEDS = params['FEEDS']
        CONVERTER = params['CONVERTER']

        self.groups = {}
        self.grpsDF = DELIVERYDF['grps']
        self.driversDF = DELIVERYDF['drivers']
        self.grpsDF['num_drivers'] = list(np.zeros(len(self.grpsDF)))

        self.num_groups = len(self.grpsDF)


        for k in range(self.num_groups):
            group = 'group' + str(k);
            params2 = {};
            params2['loc'] = self.grpsDF.iloc[k]['depot_loc']
            params2['GRAPHS'] = self.GRAPHS
            params2['FEEDS'] = self.FEEDS
            params2['group_ind'] = k;
            params2['default_poly'] = self.fit['poly']
            self.groups[group] = GROUP(self.grpsDF,self.driversDF[group],params2);



        # num_drivers = 8; 
        # driver_start_time = WORLD['main']['start_time'];
        # driver_end_time = WORLD['main']['end_time'];
        # am_capacity = 8
        # wc_capacity = 2
        # self.driver_runs = [];
        # self.time_matrix = np.zeros([1,1]);

        # self.date = '2023-07-31'
        # loc = center_point
        # self.depot = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
        # self.depot_node = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

        # for i in range(num_drivers):
        #     DELIVERY['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
        #         'am_capacity': am_capacity,'wc_capacity': wc_capacity})


        ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 
        ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 
        ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 
        ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 
        ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 

        # print('DELIVERY SETUP...')

        # params = {}
        # params['direct_locs'] = LOCS['delivery1'];
        # params['shuttle_locs'] = LOCS['delivery2'];
        # params['NODES'] = NODES;
        # params['num_groups'] = 2;
        # # params['BUS_STOP_NODES'] = BUS_STOP_NODES;



        direct_node_lists = {}
        direct_node_lists['source'] = params['NODES']['delivery1'];
        direct_node_lists['transit'] = params['NODES']['delivery1_transit'];
        shuttle_node_lists = {}
        shuttle_node_lists['source'] = params['NODES']['delivery2'];
        shuttle_node_lists['transit'] = params['NODES']['delivery2_transit'];
        
        #BUS_STOP_NODES = params['BUS_STOP_NODES']
                    
        DELIVERY = {};
        DELIVERY['direct'] = {};
        DELIVERY['shuttle'] = {};        



        shuttle_locs = LOCS['delivery2'];

        # nopts = len(opts);
        # people_tags = [];
        delivery1_tags = [];
        delivery2_tags = [];

        delivery_nodes = [];
        # for i,loc in enumerate(direct_locs):
        #     tag = 'delivery1_' + str(i);
        #     delivery1_tags.append(tag);
        #     DELIVERY['direct'][tag] = {};
        #     DELIVERY['direct'][tag]['active_trips'] = [];
        #     DELIVERY['direct'][tag]['active_trip_history'] = [];    
        #     DELIVERY['direct'][tag]['loc'] = loc; #DELIVERY1_LOC[i]
        #     DELIVERY['direct'][tag]['current_path'] = []    

        #     DELIVERY['direct'][tag]['nodes'] = {};
        #     node = ox.distance.nearest_nodes(GRAPHS['ondemand'], loc[0], loc[1]);
        #     DELIVERY['direct'][tag]['nodes']['source'] = node
        #     direct_node_lists['source'].append(node)


        #     # node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
        #     # node = BUS_STOP_NODES['ondemand'][node0];
        #     node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
        #     node = int(convertNode(node0,'transit','ondemand',NDF))

        #     DELIVERY['direct'][tag]['nodes']['transit'] = node
        #     DELIVERY['direct'][tag]['nodes']['transit2'] = node0
        #     direct_node_lists['transit'].append(node)    

        #     DELIVERY['direct'][tag]['people'] = [];
        #     DELIVERY['direct'][tag]['sources'] = [];
        #     DELIVERY['direct'][tag]['targets'] = [];


        self.DELIVERY = {}
        self.DELIVERY['shuttle'] = {};

        delivery2_nodes = [];
        for i,loc in enumerate(shuttle_locs):
            tag = 'delivery2_' + str(i);2
            delivery2_tags.append(tag);
            self.DELIVERY['shuttle'][tag] = {};
            self.DELIVERY['shuttle'][tag]['active_trips'] = [];    
            self.DELIVERY['shuttle'][tag]['active_trip_history'] = [];    
            self.DELIVERY['shuttle'][tag]['loc'] = loc;
            self.DELIVERY['shuttle'][tag]['current_path'] = []

            self.DELIVERY['shuttle'][tag]['nodes'] = {};
            node = ox.distance.nearest_nodes(self.GRAPHS['ondemand'], loc[0], loc[1]);
            self.DELIVERY['shuttle'][tag]['nodes']['source'] = node
            shuttle_node_lists['source'].append(node)    

            # node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
            # node = BUS_STOP_NODES['ondemand'][node0];
            node0 = ox.distance.nearest_nodes(self.GRAPHS['gtfs'], loc[0], loc[1]);
            node = int(CONVERTER.convertNode(node0,'gtfs','ondemand'));#from_type='feed'))


            self.DELIVERY['shuttle'][tag]['nodes']['transit'] = node
            self.DELIVERY['shuttle'][tag]['nodes']['transit2'] = node0
            shuttle_node_lists['transit'].append(node)    

            self.DELIVERY['shuttle'][tag]['people'] = [];
            self.DELIVERY['shuttle'][tag]['sources'] = [];
            self.DELIVERY['shuttle'][tag]['targets'] = [];


    def selectDeliveryGroup(self,trip,GRAPHS,typ='direct'):


        GRAPH = GRAPHS['drive'];
        node1 = trip[0]; node2 = trip[1];
        loc1 = np.array([GRAPH.nodes[node1]['x'],GRAPH.nodes[node1]['y']]);
        loc2 = np.array([GRAPH.nodes[node2]['x'],GRAPH.nodes[node2]['y']]);
        pt1_series = gpd.GeoSeries([Point(loc1)])
        pt2_series = gpd.GeoSeries([Point(loc2)])

        ###### types are in priority order 
        possible_group_inds = [];
        riders_to_drivers = [];
        for i in range(len(self.grpsDF)):
            ROW = self.grpsDF.iloc[i]
            types = ROW['type']
            if not(isinstance(types,list)): types = [types]        

            polygon = ROW['polygon']
            polygon_series = gpd.GeoSeries([Polygon(polygon)])
            intersect1 = polygon_series.intersection(pt1_series)
            intersect2 = polygon_series.intersection(pt2_series)
            pt1_inside = not(intersect1.is_empty[0])
            pt2_inside = not(intersect2.is_empty[0])



            if (typ in types) and pt1_inside and pt2_inside:
                possible_group_inds.append(i)
                riders_to_drivers.append(ROW['num_possible_trips']/ROW['num_drivers'])

        if len(riders_to_drivers)>0:
            ind = np.argmin(riders_to_drivers);
            group_ind = possible_group_inds[ind];
            group = self.grpsDF.iloc[group_ind]['group']
        else:
            group = []; group_ind = None
        return group,group_ind


    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 


    def generateCongestionModels(self,NETWORK,counts = {'num_counts':2,'num_per_count':1},verbose=False):
    # def generateOndemandCongestion(WORLD,PEOPLE,DELIVERIES,GRAPHS,num_counts=3,num_per_count=1,verbose=True,show_delivs='all',clear_active=True):

    # %load_ext autoreload
    # %autoreload 2
    # from multimodal_functions import * 

    # num_counts = 5; num_per_count = 1;
    # pts = generateOndemandCongestion(WORLD,PEOPLE,DELIVERY,GRAPHS,num_counts = num_counts,num_per_count=num_per_count)



        # def NOTEBOOK(self):
        # num_counts = 5; num_per_count = 1;
        # pts = generateOndemandCongestion(WORLD,PEOPLE,DELIVERY,GRAPHS,num_counts = num_counts,num_per_count=num_per_count)
        # for _,group in enumerate(DELIVERY['groups']):
        #     counts = DELIVERY['groups'][group]['fit']['counts']
        #     pts = DELIVERY['groups'][group]['fit']['pts']
        #     shp = np.shape(pts);
        #     xx = np.reshape(counts,shp[0]*shp[1])
        #     coeffs = DELIVERY['groups'][group]['fit']['poly']
        #     Yh = DELIVERY['groups'][group]['fit']['Yh'];
        #     for j in range(np.shape(pts)[1]):
        #         plt.plot(counts,pts[:,j],'o',color='blue')        
        #     plt.plot(xx,Yh,'--',color='red')

        print('DEMAND CURVE - generate (takes 10 MIN...)')
        print('...runs VRP solver (num_pts*num_per_count) times...')
        print('### ~ approx. 1 run/15 seconds...')




        if hasattr(self,'groups'):
            mode = 'ondemand'
            all_possible_trips = list(NETWORK.segs);
            if verbose: print('starting on-demand curve computations for a total of',len(all_possible_trips),'trips...')
            trips_by_group = {group:[] for _,group in enumerate(self.groups)}
            for _,seg in enumerate(all_possible_trips):
                group = NETWORK.segs[seg].group;
                trips_by_group[group].append(seg)

            # for k, group in enumerate(self.groups):
            for k, group in enumerate(trips_by_group):
                possible_trips = trips_by_group[group]

                GROUP = self.groups[group]
                GROUP.generateCongestionModel(possible_trips,NETWORK,self,counts,verbose=verbose)


    def plotCongestionModels(self,ax):  ## NOT FIXED YET....
        for _,group in enumerate(self.groups):
            counts = self.groups[group].fit['counts']
            pts = self.groups[group].fit['pts']
            shp = np.shape(pts);
            xx = np.reshape(counts,shp[0]*shp[1])
            coeffs = self.groups[group].fit['poly']
            Yh = self.groups[group].fit['Yh'];
            for j in range(np.shape(pts)[1]):
                ax.plot(counts,pts[:,j],'o',color='blue')
            ax.plot(xx,Yh,'--',color='red')        



        # else: 
        #     DELIVERY = DELIVERIES;
        #     if verbose: print('starting on-demand curve computation...')    
        #     mode = 'ondemand'
        #     GRAPH = GRAPHS['ondemand'];
        #     ONDEMAND = WORLD['ondemand'];
        #     possible_trips = list(WORLD[mode]['trips']) ;

        #     print('...for a total number of trips of',len(possible_trips))
        #     print('counts to compute: ',counts)

        #     pts = np.zeros([len(counts),num_per_count]);

        #     ptsx = {};
        #     for i,count in enumerate(counts):
        #         print('...computing averages for',count,'active ondemand trips...')
        #         for j in range(num_per_count):
        #             trips_to_plan = sample(possible_trips,count)

        #             nodes_to_update = [DELIVERY['depot_node']];
        #             nodeids_to_update = [0];
        #             for _,trip in enumerate(trips_to_plan):
        #                 TRIP = WORLD[mode]['trips'][trip]
        #                 nodes_to_update.append(trip[0]);
        #                 nodes_to_update.append(trip[1]);
        #                 nodeids_to_update.append(TRIP['pickup_node_id'])
        #                 nodeids_to_update.append(TRIP['dropoff_node_id'])

        #             updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
        #             payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

        #             manifests = optimizer.offline_solver(payload) 
        #             PDF = payloadDF(payload,GRAPHS)
        #             MDF = manifestDF(manifests,PDF)
        #             average_time,times_to_average = assignCostsFromManifest(trips_to_plan,self.segs,nodes,0)

        #             pts[i,j] = average_time
        #             # ptsx[(i,j)] = times_to_average
        #     out = pts






# def createGroupsDF(self):
def createGroupsDF(polygons,types0 = []):  ### ADDED TO CLASS
    ngrps = len(polygons);
    groups = ['group'+str(i) for i in range(len(polygons))];
    depotlocs = [];
    for polygon in polygons:
        depotlocs.append((1./np.shape(polygon)[0])*np.sum(polygon,0));    
    if len(types0)==0: types = [['direct','shuttle'] for i in range(ngrps)]
    else: types = types0;
    GROUPS = pd.DataFrame({'group':groups,
    'depot_loc':depotlocs,'type':types,'polygon':polygons,
    'num_possible_trips':np.zeros(ngrps),
    'num_drivers':np.zeros(ngrps) 
       }); #,index=list(range(ngrps))
    # new_nodes = pd.DataFrame(node_tags,index=[index_node])
    # NODES[mode] = pd.concat([NODES[mode],new_nodes]);
    return GROUPS

# def createDriversDF(self): 
def createDriversDF(params,WORLD):  ### ADDED TO CLASS
    num_drivers = params['num_drivers']
    if not('start_time' in params): start_times = [WORLD['main']['start_time'] for _ in range(num_drivers)]
    elif not(isinstance(params['start_time'],list)): start_times = [params['start_time'] for _ in range(num_drivers)]
    if not('end_time' in params): end_times = [WORLD['main']['end_time'] for _ in range(num_drivers)]
    elif not(isinstance(params['end_time'],list)): end_times = [params['end_time'] for _ in range(num_drivers)]
    if not('am_capacity' in params): am_capacities = [8 for _ in range(num_drivers)];
    elif not(isinstance(params['am_capacity'],list)): am_capacities = [params['am_capacity'] for _ in range(num_drivers)]        
    if not('wc_capacity' in params): wc_capacities = [2 for _ in range(num_drivers)];
    elif not(isinstance(params['wc_capacity'],list)): wc_capacities = [params['wc_capacity'] for _ in range(num_drivers)]
    OUT = pd.DataFrame({'start_time':start_times, 'end_time':end_times,'am_capacity':am_capacities, 'wc_capacity':wc_capacities})
    return OUT
        

def fakeManifest(self): #payload): ### ADDED TO CLASS
# def fakeManifest(payload): ### ADDED TO CLASS
    num_drivers = len(payload['driver_runs']);
    num_requests = len(payload['requests']);
    requests_per_driver = int(num_requests/(num_drivers-1));

    run_id = 0; 
    scheduled_time = 10;
    manifest = [];
    requests_served = 0;

    for i,request in enumerate(payload['requests']):

        scheduled_time = scheduled_time + 1000;
        manifest.append({'run_id':run_id,
                         'booking_id': request['booking_id'],
                         'action': 'pickup',
                         'scheduled_time':scheduled_time,                
                         'node_id':request['pickup_node_id']});

        scheduled_time = scheduled_time + 1000;
        manifest.append({'run_id':run_id,
                         'booking_id': request['booking_id'],
                         'action': 'dropoff',
                         'scheduled_time':scheduled_time,                
                         'node_id':request['dropoff_node_id']});

        requests_served = requests_served + 1;
        if requests_served > requests_per_driver:
            run_id = run_id + 1
            requests_served = 0;
    return manifest


class GROUP: 
    def __init__(self,grpsDF,driversDF,params):


        self.GRAPHS = params['GRAPHS'];
        self.FEEDS = params['FEEDS'];
        self.group_ind = params['group_ind']
        self.group = 'group'+str(self.group_ind);

        self.fit = {'poly':params['default_poly']}#np.array([406.35315058,  18.04891652])};

        self.booking_ids = [];
        self.time_matrix = np.zeros([1,1]);
        self.expected_cost = [0.];
        self.current_expected_cost = 0.;
        self.actual_average_cost = [];
        self.driver_runs = [];
        self.date = '2023-07-31'
        # num_drivers = 8; 
        # loc = ROW['depot_loc']        
        self.loc = params['loc']
        self.depot = {'pt': {'lat': self.loc[1], 'lon': self.loc[0]}, 'node_id': 0}
        self.depot_node = ox.distance.nearest_nodes(self.GRAPHS['drive'], self.loc[0], self.loc[1]);
        

        self.expected_cost = [0.];
        self.current_expected_cost = 0.;
        self.actual_average_cost = [];

        self.driversDF = driversDF

        # for k in range(self.num_groups):
        #     ROW = self.grpsDF.iloc[k];
        #     group = ROW['group']
        if len(self.driversDF) > 0:
            num_drivers = len(self.driversDF);
            for i in range(num_drivers):
                ROW2 = self.driversDF.iloc[i];
                driver_start_time = ROW2['start_time'];
                driver_end_time = ROW2['end_time']; 
                am_capacity = ROW2['am_capacity'];
                wc_capacity = ROW2['wc_capacity'];
                self.driver_runs.append({'run_id': i,
                    'start_time': driver_start_time,'end_time': driver_end_time,
                    'am_capacity': am_capacity,'wc_capacity': wc_capacity})
        else: 
            for i in range(num_drivers):
                driver_start_time = WORLD['main']['start_time'];
                driver_end_time = WORLD['main']['end_time'];
                am_capacity = 8
                wc_capacity = 2
                self.driver_runs.append({'run_id': i,
                    'start_time': driver_start_time,'end_time': driver_end_time,
                    'am_capacity': am_capacity,'wc_capacity': wc_capacity})

        grpsDF['num_drivers'].iloc[self.group_ind] = num_drivers;
        grpsDF['num_possible_trips'].iloc[self.group_ind] = 0;

    # def addFit(self,poly):
    #     self.fit = {'poly':poly}

    def generateCongestionModel(self,possible_trips,NETWORK,ONDEMAND,counts = {'num_counts':2,'num_per_count':1},verbose=True):

        num_counts = counts['num_counts'];
        num_per_count = counts['num_per_count'];
    # def generateOndemandCongestion(WORLD,PEOPLE,DELIVERIES,GRAPHS,num_counts=3,num_per_count=1,verbose=True,show_delivs='all',clear_active=True):

        # def NOTEBOOK(self):
        # num_counts = 5; num_per_count = 1;
        # pts = generateOndemandCongestion(WORLD,PEOPLE,DELIVERY,GRAPHS,num_counts = num_counts,num_per_count=num_per_count)
        # for _,group in enumerate(DELIVERY['groups']):
        #     counts = DELIVERY['groups'][group]['fit']['counts']
        #     pts = DELIVERY['groups'][group]['fit']['pts']
        #     shp = np.shape(pts);
        #     xx = np.reshape(counts,shp[0]*shp[1])
        #     coeffs = DELIVERY['groups'][group]['fit']['poly']
        #     Yh = DELIVERY['groups'][group]['fit']['Yh'];
        #     for j in range(np.shape(pts)[1]):
        #         plt.plot(counts,pts[:,j],'o',color='blue')        
        #     plt.plot(xx,Yh,'--',color='red')
        
        if len(possible_trips)>0:

            # counts = [20,30,40,50,60,70,80];
            num_pts = num_counts; num_per_count = num_per_count; 

            counts = np.linspace(1,len(possible_trips)-1,num_pts)
            counts = [int(count) for _,count in enumerate(counts)];
                    
            if verbose: print('starting on-demand curve computation for group NUMBER ',self.group_ind) #'with',len(possible_trips),'...')
                    
            GRAPH = self.GRAPHS['ondemand'];
                    
            # ONDEMAND = self;
            print('...for a total number of trips of',len(possible_trips))
            print('counts to compute: ',counts)

            pts = np.zeros([len(counts),num_per_count]);

            ptsx = {};
            for i,count in enumerate(counts):
                print('...computing averages for',count,'active ondemand trips in group',self.group_ind,'...')
                for j in range(num_per_count):

                    # try: 
                    trips_to_plan = sample(possible_trips,count)
                    nodes_to_update = [self.depot_node];
                    nodeids_to_update = [0];
                    for _,seg in enumerate(possible_trips):
                        SEG = NETWORK.segs[seg]
                        nodes_to_update.append(seg[0]);
                        nodes_to_update.append(seg[1]);
                        nodeids_to_update.append(SEG.pickup_node_id)
                        nodeids_to_update.append(SEG.dropoff_node_id)



                    self.updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,ONDEMAND); #GRAPHS['ondemand']);
                    payload,nodes = self.constructPayload(trips_to_plan,ONDEMAND,NETWORK,self.GRAPHS);
                    # self.updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,self.GRAPHS['ondemand'],DELIVERY,WORLD);
                    # payload,nodes = self.constructPayload(trips_to_plan,self,WORLD,self.GRAPHS);

                    manifests = optimizer.offline_solver(payload) 

                    PDF = self.payloadDF(payload,self.GRAPHS)
                    MDF = self.manifestDF(manifests,PDF)
                    average_time,times_to_average = self.assignCostsFromManifest(trips_to_plan,NETWORK.segs,nodes,MDF,NETWORK,0)
                    pts[i,j] = average_time
                    # except:
                    #     pass
                    # ptsx[(i,j)] = times_to_average
        else:
            counts = np.linspace(1,100,num_counts)
            pts = 1000.*np.ones([num_counts,num_per_count]);


        self.fit = {}
        self.fit['counts'] = np.outer(counts,np.ones(num_per_count))
        self.fit['pts'] = pts.copy();

        shp = np.shape(pts);
        xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
        yy = np.reshape(pts,shp[0]*shp[1])
        order = 1;
        coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
        Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
        self.fit['poly'] = coeffs[::-1]
        self.fit['Yh'] = Yh.copy();

    def planGroup(self):
        pass

    def updateTravelTimeMatrix(self,nodes,ids,ONDEMAND,group=[0]): #nodes,ids,GRAPH,DELIVERY,WORLD,group=[0]):  ### ADDED TO CLASS
    # def updateTravelTimeMatrix(nodes,ids,GRAPH,DELIVERY,WORLD,group=[0]):  ### ADDED TO CLASS
        #MAT = travel_time_matrix(nodes,GRAPH)
        GRAPH = self.GRAPHS['drive'];
        MAT = np.zeros([len(nodes),len(nodes)]);
        for i,node1 in enumerate(nodes):
            # distances, paths = nx.multi_source_dijkstra(GRAPH, nodes, target=node1, weight='c');
            # paths = nx.shortest_path(GRAPH, target=node1, weight='c');
            distances = nx.shortest_path_length(GRAPH, target=node1, weight='c');
            for j,node2 in enumerate(nodes):
                try:
                    MAT[i,j] = distances[node2];
                except:
                    MAT[i,j] = 10000000.0

        for i,id1 in enumerate(ids):
            for j,id2 in enumerate(ids):
                if len(self.time_matrix)>0:
                    self.time_matrix[id1][id2] = MAT[i,j];
                else:
                    ONDEMAND.time_matrix[id1][id2] = MAT[i,j];



    def getBookingIds(self):    ### ADDED TO CLASS
    # def getBookingIds(PAY):    ### ADDED TO CLASS
        booking_ids = [];
        for i,request in enumerate(PAY['requests']):
            booking_ids.append(request['booking_id'])
        return booking_ids

    def filterPayloads(self):
    # def filterPayloads(PAY,ids_to_keep):  ### ADDED TO CLASS
        OUT = {};
        OUT['date'] = PAY['date']
        OUT['depot'] = PAY['depot']
        OUT['driver_runs'] = PAY['driver_runs']
        OUT['time_matrix'] = PAY['time_matrix']

        OUT['requests'] = [];
        for i,request in enumerate(PAY['requests']):
            if request['booking_id'] in ids_to_keep: 
                OUT['requests'] = request
        return OUT;



    def constructPayload(self,active_segs,ONDEMAND,NETWORK,GRAPHS): #active_trips,DELIVERY,WORLD,GRAPHS,group=[0]): #PAY,ids_to_keep):   ### ADDED TO CLASS
    # def constructPayload(active_trips,DELIVERY,WORLD,GRAPHS,group=[0]): #PAY,ids_to_keep):   ### ADDED TO CLASS

        payload = {};
        payload['date'] = self.date
        payload['depot'] = self.depot
        payload['driver_runs'] = self.driver_runs

        if len(self.time_matrix)>0:
            MAT = self.time_matrix;
        else:
            MAT = ONDEMAND.time_matrix;

        GRAPH = GRAPHS['ondemand']
        
        nodes = {};
        inds = [0];
        inds2 = [0];
        curr_node_id = 1;

        mode = 'ondemand'
        requests = [];
        for i,seg in enumerate(active_segs):
            requests.append({})
            node1 = seg[0];
            node2 = seg[1];

            SEG = NETWORK.segs[seg];

            requests[i]['booking_id'] = SEG.booking_id
            requests[i]['pickup_node_id'] = curr_node_id; curr_node_id = curr_node_id + 1; #TRIP['pickup_node_id']
            requests[i]['dropoff_node_id'] = curr_node_id; curr_node_id = curr_node_id + 1; #TRIP['dropoff_node_id']

            requests[i]['am'] = int(SEG.am)
            requests[i]['wc'] = int(SEG.wc)
            requests[i]['pickup_time_window_start'] = SEG.pickup_time_window_start
            requests[i]['pickup_time_window_end'] = SEG.pickup_time_window_end
            requests[i]['dropoff_time_window_start'] = SEG.dropoff_time_window_start
            requests[i]['dropoff_time_window_end'] = SEG.dropoff_time_window_end

            requests[i]['pickup_pt'] = {'lat': float(GRAPH.nodes[node1]['y']), 'lon': float(GRAPH.nodes[node1]['x'])}
            requests[i]['dropoff_pt'] = {'lat': float(GRAPH.nodes[node2]['y']), 'lon': float(GRAPH.nodes[node2]['x'])}

            inds.append(requests[i]['pickup_node_id'])
            inds.append(requests[i]['dropoff_node_id'])

            inds2.append(SEG.pickup_node_id);
            inds2.append(SEG.dropoff_node_id);

            nodes[node1] = {'main':SEG.pickup_node_id,'curr':requests[i]['pickup_node_id']}
            nodes[node2] = {'main':SEG.dropoff_node_id,'curr':requests[i]['dropoff_node_id']}

            # 'pickup_pt': {'lat': 35.0296296, 'lon': -85.2301767},
            # 'dropoff_pt': {'lat': 35.0734152, 'lon': -85.1315328},
            # 'booking_id': 39851211,
            # 'pickup_node_id': 1,
            # 'dropoff_node_id': 2},


        inds = np.array(inds);
        inds2 = np.array(inds2);

        MAT = np.array(MAT)
        MAT = MAT[np.ix_(inds2,inds2)]
        MAT2 = []
        for i,row in enumerate(MAT):
            MAT2.append(list(row.astype('int')))
        payload['time_matrix'] = MAT2
        payload['requests'] = requests
        return payload,nodes;


    def payloadDF(self,PAY,GRAPHS,include_drive_nodes=False): #PAY,GRAPHS,include_drive_nodes = False):  ### ADDED TO CLASS
    # def payloadDF(PAY,GRAPHS,include_drive_nodes = False):  ### ADDED TO CLASS
        PDF = pd.DataFrame({'booking_id':[],
                            'pickup_pt_lat':[],'pickup_pt_lon':[],
                            'dropoff_pt_lat':[],'dropoff_pt_lon':[],
                            'pickup_drive_node':[],'dropoff_drive_node':[],
                            'pickup_node_id':[],
                            'dropoff_node_id':[]},index=[])

        zz = PAY['requests']
        for i,request in enumerate(PAY['requests']):

            GRAPH = GRAPHS['drive'] 
            book_id = request['booking_id'];


            plat = request['pickup_pt']['lat']
            plon = request['pickup_pt']['lon']
            dlat = request['dropoff_pt']['lat']
            dlon = request['dropoff_pt']['lon']

            if include_drive_nodes:
                pickup_drive = ox.distance.nearest_nodes(GRAPH, plon,plat);
                dropoff_drive = ox.distance.nearest_nodes(GRAPH, dlon,dlat);
            else:
                pickup_drive = None;
                dropoff_drive = None;
            
            INDEX = book_id;
            DETAILS = {'booking_id':[book_id],
                       'pickup_pt_lat':[plat],'pickup_pt_lon':[plon],
                       'dropoff_pt_lat':[dlat],'dropoff_pt_lon':[dlon],
                       'pickup_drive_node':pickup_drive,
                       'dropoff_drive_node':dropoff_drive,
                       'pickup_node_id':request['pickup_node_id'],
                       'dropoff_node_id':request['dropoff_node_id']}
            
            NEW = pd.DataFrame(DETAILS,index=[INDEX])

            PDF = pd.concat([PDF,NEW]);
        return PDF



    ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS 
    ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS 
    ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS ###### MANIFEST FUNCTIONS 



    def manifestDF(self,MAN,PDF):
    # def manifestDF(MAN,PDF):   ### ADDED TO CLASS
        MDF = pd.DataFrame({'run_id':[],'booking_id':[],'action':[],'scheduled_time':[],'node_id':[]},index=[])
        for i,leg in enumerate(MAN):
            run_id = leg['run_id']
            book_id = leg['booking_id'];
            action = leg['action']
            scheduled_time = leg['scheduled_time'];
            node_id = leg['node_id']

            if action == 'pickup': drive_node = PDF['pickup_drive_node'][book_id]
            if action == 'dropoff': drive_node = PDF['dropoff_drive_node'][book_id]
                
            INDEX = drive_node;
            DETAILS = {'booking_id':[book_id],'run_id':[run_id],'action':[action],'scheduled_time':[scheduled_time],
                        'node_id':[node_id],'drive_node':[drive_node]}
            
            NEW = pd.DataFrame(DETAILS,index=[INDEX])
            MDF = pd.concat([MDF,NEW]);
        return MDF


    def routeFromStops(self): #nodes,GRAPH): ### ADDED TO CLASS        
    # def routeFromStops(nodes,GRAPH): ### ADDED TO CLASS
        PATH_NODES = [];
        for i,node in enumerate(nodes):
            if i<len(nodes)-1:

                try:
                    start_node = node; end_node = nodes[i+1]
                    path = nx.shortest_path(GRAPH, source=start_node, target=end_node,weight = 'c'); #, weight=None);
                    PATH_NODES = PATH_NODES + path
                except:
                    PATH_NODES = PATH_NODES + [];

        PATH_EDGES = [];
        for i,node in enumerate(PATH_NODES):
            if i<len(PATH_NODES)-1:
                start_node = node; end_node = PATH_NODES[i+1]
                PATH_EDGES.append((start_node,end_node))

        out = {'nodes':PATH_NODES,'edges':PATH_EDGES}
        return out


    def routesFromManifest(self): #MDF,GRAPHS): ### ADDED TO CLASS
    # def routesFromManifest(MDF,GRAPHS): ### ADDED TO CLASS
        max_id_num = int(np.max(list(MDF['run_id'])))+1
        PLANS = {};
        for i in range(max_id_num):
            ZZ = MDF[MDF['run_id']==i]
            ZZ = list(ZZ.sort_values(['scheduled_time'])['drive_node'])
            out = routeFromStops(ZZ,GRAPHS['drive']);
            PLANS[i] = out.copy();
        return PLANS

    def singleRoutesFromManifest(self): #MDF,GRAPHS): ### ADDED TO CLASS
    # def singleRoutesFromManifest(MDF,GRAPHS): ### ADDED TO CLASS
        max_id_num = int(np.max(list(MDF['run_id'])))+1
        PLANS = {};
        for i in range(max_id_num):
            ZZ = MDF[MDF['run_id']==i]
            ZZ = ZZ.sort_values(['scheduled_time'])
            booking_ids = ZZ['booking_id'].unique();
            PLANS[i] = {};
            for idd in booking_ids:
                inds = np.where(ZZ['booking_id']==idd)[0]
                ZZZ = ZZ.iloc[inds[0]:inds[1]+1];
                zz = list(ZZZ['drive_node'])
                out = routeFromStops(zz,GRAPHS['drive']);
                node1 = int(zz[0]);
                node2 = int(zz[-1]);
                PLANS[i][(node1,node2)] = {}
                PLANS[i][(node1,node2)]['booking_id'] = idd;
                PLANS[i][(node1,node2)]['nodes'] = out['nodes']
                PLANS[i][(node1,node2)]['edges'] = out['edges'];
        return PLANS        

    def assignCostsFromManifest(self,active_segs,all_segs,nodes,MDF,NETWORK,ave_cost,track=True): #active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS
    # def assignCostsFromManifest(active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS
        mode = 'ondemand';

        times_to_average = [0.];

        actual_costs = {};
        runs_info = {};
        run_ids = {};
        for i,seg in enumerate(active_segs):
            node1 = seg[0]
            node2 = seg[1]
            booking_id = NETWORK.segs[seg].booking_id
            main_node_id1 = nodes[seg[0]]['main']
            main_node_id2 = nodes[seg[1]]['main']
            curr_node_id1 = nodes[seg[0]]['curr']
            curr_node_id2 = nodes[seg[1]]['curr']


            try: 

                # mask1 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
                # mask2 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');
                mask1 = (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
                mask2 = (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');

                time1 = MDF[mask1]['scheduled_time'][0];
                time2 = MDF[mask2]['scheduled_time'][0];

                run_id = int(MDF[mask1]['run_id'][0]);


                time_diff = np.abs(time2-time1)

                actual_costs[seg] = time_diff;
                run_ids[seg] = run_id;

                runs_info[seg] = {}
                runs_info[seg]['runid'] = run_id;
                
                MDF2 = MDF[MDF['run_id'] == run_id]
                ind1 = np.where(MDF2['node_id'] == curr_node_id1)[0][0];
                ind2 = np.where(MDF2['node_id'] == curr_node_id2)[0][0];
                MDF2 = MDF2.iloc[ind1:ind2+1]
                runs_info[seg]['num_passengers'] = MDF2['booking_id'].nunique();

                if time_diff < 7200:
                    times_to_average.append(time_diff);

            except: 
                actual_costs[seg] = 1000000000.;
                # pass; #time_diff = 10000000000.

            # print(time_diff)

            # if False:
            #     WORLD[mode]['trips'][trip]['costs']['time'].append(time_diff);
            #     WORLD[mode]['trips'][trip]['costs']['current_time'] = time_diff;
            #     factors = ['time','money','conven','switches']
            #     for j,factor in enumerate(factors):#WORLD[mode]['trips'][trip]['costs']):
            #         if not(factor == 'time'): 
            #             # WORLD[mode]['trips']['costs'][factor] = 0. 
            #             WORLD[mode]['trips'][trip]['costs'][factor].append(0.)
            #             WORLD[mode]['trips'][trip]['costs']['current_'+factor] = 0. 

        average_time = np.mean(np.array(times_to_average))

        print('...average manifest trip time: ',average_time)
        # est_ave_time = average_time
        est_ave_time = ave_cost



        for i,seg in enumerate(active_segs):#active_trips):
            #if not(trip in active_trips):

            NETWORK.segs[seg].costs['current_time'] = est_ave_time;
            if seg in active_segs:
                NETWORK.segs[seg].costs['time'].append(actual_costs[seg]);#est_ave_time);
                NETWORK.segs[seg].costs['current_time'] = actual_costs[seg];#est_ave_time);
                try:
                    NETWORK.segs[seg]['run_id'] = run_ids[seg] 
                    NETWORK.segs[seg]['num_passengers'] = runs_info[seg]['num_passengers'];
                except:
                    pass
            else:
                if len(NETWORK.segs[seg].costs['time'])==0: prev_time = est_ave_time;
                else: prev_time = NETWORK.segs[seg].costs['time'][-1]
                NETWORK.segs[seg].costs['time'].append(prev_time);
                NETWORK.segs[seg].costs['current_time'] = prev_time;


            factors = ['time','money','conven','switches']
            for j,factor in enumerate(factors):#WORLD[mode]['trips'][trip]['costs']):
                if not(factor == 'time'): 
                    # WORLD[mode]['trips']['costs'][factor] = 0. 
                    NETWORK.segs[seg].costs[factor].append(0.);
                    NETWORK.segs[seg].costs['current_'+factor] = 0. 

        return average_time,times_to_average



class TRIP:
    def __init__(self,modes,nodes,CONVERTER,delivery_grps=[],deliveries=[]): #,node_types,NODES,deliveries = []):
    # def makeTrip(modes,nodes,NODES,delivery_grps=[],deliveries=[]): #,node_types,NODES,deliveries = []):
        trip = {};
        legs = [];

        deliv_counter = 0;
        for i,mode in enumerate(modes):
            legs.append({});
            legs[-1]['mode'] = mode;
            legs[-1]['start_nodes'] = []; #nodes[i]
            legs[-1]['end_nodes'] = []; #nodes[i+1]
            legs[-1]['path'] = [];
            
            if len(delivery_grps)>0:
                if mode == 'ondemand': legs[-1]['delivery_group'] = delivery_grps[i];
                else: legs[-1]['delivery_group'] = None;

            for j,node in enumerate(nodes[i]['nodes']):
                node2 = CONVERTER.convertNode(node,nodes[i]['type'],mode)
                legs[-1]['start_nodes'].append(node2)
            for j,node in enumerate(nodes[i+1]['nodes']):
                node2 = CONVERTER.convertNode(node,nodes[i+1]['type'],mode)
                legs[-1]['end_nodes'].append(node2);
                    
    #         segs[-1]['start_types'] = node_types[i]
    #         segs[-1]['end_types'] = node_types[i+1]

            if mode == 'ondemand':
                # segs[-1]['delivery'] = deliveries[deliv_counter]
                # deliv_counter = deliv_counter + 1;
                legs[-1]['delivery'] = delivery_grps[i];
                
        
        self.current = {};
        self.structure = legs
        self.current['cost'] = 100000000;
        self.current['traj'] = [];
        self.active = False;
        
    def queryTrip(self,PERSON,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND):

        trip_cost = 0;
        costs_to_go = [];
        # COSTS_to_go = [];
        next_inds = [];
        next_nodes = [];

        for k,segment in enumerate(self.structure[::-1]):
            end_nodes = segment['end_nodes'];
            start_nodes = segment['start_nodes'];
            group = segment['delivery_group']
    #         print(SEG['end_types'])
    #         end_types = SEG['end_types'];
    #         start_types = SEG['start_types'];
            mode = segment['mode'];
            NETWORK = NETWORKS[mode]
            costs_to_go.append([]);
            # COSTS_to_go.append([]);
            next_inds.append([]);
            next_nodes.append([]);
    #         end_type = end_types[k]
    #         start_type = start_types[k];
            for j,end in enumerate(end_nodes):
                possible_leg_costs = np.zeros(len(start_nodes));
                # possible_leg_COSTS = list(np.zeros(len(start_nodes)));
                for i,start in enumerate(start_nodes):                
                    end_node = end;#int(NODES[end_type][mode][end]);
                    start_node = start;#int(NODES[start_type][mode][start]);
                    seg = (start_node,end_node)

                    if not(seg in NETWORK.segs):
                        NETWORK.segs[seg] = SEG(seg,mode,PERSON,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND,params={'group':group})

                    # SEG = NETWORK.segs[seg];

                    ############################################################
                    # def querySeg(seg,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group='group0'):
                    # cost,path,_ = querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group=group)
                    ############### WAS querySeg ##################





    # start = seg_details[0]
    # end = seg_details[1];
    # if not((start,end) in WORLD[mode]['trips']):#.keys()):  
    #     if mode == 'gtfs':
    #         planGTFSSeg(seg_details,mode,GRAPHS,WORLD,NODES,mass=0);
    #     elif mode == 'ondemand':
    #         addDelivSeg((start,end),mode,GRAPHS,DELIVERY,WORLD,group=group,mass=0,PERSON=PERSON);

    #     elif (mode == 'drive') or (mode == 'walk'):
    #         planDijkstraSeg((start,end),mode,GRAPHS,WORLD,mass=0);
    # # print(PERSON['prefs'].keys())
    # # print('got here for mode...',mode)

    # tot_cost = 0;
    # weights = [];
    # costs = [];
    # COSTS = {};
    # for l,factor in enumerate(PERSON['prefs'][mode]):
    #     weights.append(PERSON['weights'][mode][factor])
    #     costs.append(WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor])
    #     COSTS['factor'] = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor]
    #     # cost = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor] # possibly change for delivery
    #     # diff = cost; #-PERSON['prefs'][mode][factor]
    #     # tot_cost = tot_cost + PERSON['weights'][mode][factor]*diff;
    # weights = np.array(weights);
    # costs = np.array(costs);
    # if 'logit_version' in PERSON: logit_version = PERSON['logit_version']
    # else: logit_version = 'weighted_sum'
    # tot_cost = applyLogitChoice(weights,costs,ver=logit_version);
    # try: 
    #     path = WORLD[mode]['trips'][(start,end)]['current_path']
    # except:
    #     path = None;

    # return tot_cost,path,COSTS

                    
                    weights = [];
                    costs = [];
                    for l,factor in enumerate(PERSON.weights[mode]):
                        weights.append(PERSON.weights[mode][factor])
                        costs.append(NETWORK.segs[seg].costs['current_'+factor])
                    weights = np.array(weights);
                    costs = np.array(costs);
                    if hasattr(PERSON,'logit_version'): logit_version = PERSON.logit_version
                    else: logit_version = 'weighted_sum'
                    tot_cost = self.applyLogitChoice(weights,costs,ver=logit_version);
                    try: 
                        path = self.segs[seg].current_path;
                    except:
                        path = None;
                    cost = tot_cost
                    ############################################################

                    
                    possible_leg_costs[i] = cost;
                    # possible_leg_COSTS[i] = COSTS.copy();
                ind = np.argmax(possible_leg_costs)
                next_inds[-1].append(ind)
                next_nodes[-1].append(end_nodes[ind])
                costs_to_go[-1].append(possible_leg_costs[ind]);
                # COSTS_to_go[-1].append(possible_leg_COSTS[ind]);
            

        
        
        init_ind = np.argmin(costs_to_go[-1]);
        init_cost = costs_to_go[-1][init_ind]
        init_node = self.structure[0]['start_nodes'][init_ind];

        costs_to_go = costs_to_go[::-1];
        next_inds = next_inds[::-1]
        next_nodes = next_nodes[::-1]
        

        inds = [init_ind];
        nodes = [init_node];
        
        prev_mode = self.structure[0]['mode'];

        from_type = 'graph';
        if prev_mode == 'gtfs': from_type = 'feed';

        for k,segment in enumerate(self.structure):
            mode = segment['mode']
    #         SEG['opt_start'] = 
    #         SEG['opt_end'] = next_nodes[k][inds[-1]]
            next_ind = next_inds[k][inds[-1]];
            next_node = next_nodes[k][inds[-1]];
            
            to_type = 'graph';
            if mode == 'gtfs': to_type = 'feed';

            segment['opt_start'] = CONVERTER.convertNode(nodes[-1],prev_mode,mode,from_type,to_type);
            segment['opt_end'] = next_node;    
            inds.append(next_ind)
            nodes.append(next_node);
            prev_mode = segment['mode']
        
        trip_cost = costs_to_go[init_ind][0];
        self.current['cost'] = trip_cost
        self.current['traj'] = nodes;

    def applyLogitChoice(self,weights,values,offsets = [],ver='weighted_sum'):
        cost = 100000000.;
        if len(offsets) == 0: offsets = np.zeros(len(weights));
        if ver == 'weighted_sum':
            cost = weights@values;
        return cost

class SEG:
    def __init__(self,seg,mode,PERSON,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND,params={}): #WORLD,params={}):
    # def querySeg(seg,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group='group0'):
                    # cost,path,_ = querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group=group)
        factors = ['distance','time','money','conven','switches']
        self.mode = mode
        self.source = seg[0]
        self.target = seg[0]
        self.start_time = 0;
        self.end_time = 14400;
        self.factors = factors;
        self.active = True;

        self.uncongested = {};


        # self.pickup_time_window_start = PERSON.pickup_time_window_start;
        # self.pickup_time_window_end = PERSON.pickup_time_window_end;
        # self.dropoff_time_window_start = PERSON.dropoff_time_window_start
        # self.dropoff_time_window_end = PERSON.dropoff_time_window_end

        start1 = PERSON.pickup_time_window_start
        end1 = PERSON.pickup_time_window_end;

        # try: 
        #     ##### NEEDS TO BE UPDATED 

        source_drive = CONVERTER.convertNode(self.source,mode,'drive');#,to_type='feed')                
        target_drive = CONVERTER.convertNode(self.target,mode,'drive');#,to_type='feed')                


        dist = nx.shortest_path_length(GRAPHS['drive'], source=source_drive, target=target_drive, weight='c');
        maxtriptime = dist*4;
        pickup_start = np.random.uniform(low=start1, high=end1-maxtriptime, size=1)[0];
        pickup_end = pickup_start + 60*20;
        dropoff_start = pickup_start;
        dropoff_end = pickup_end + maxtriptime;

        self.pickup_time_window_start = int(pickup_start);
        self.pickup_time_window_end = int(pickup_end);
        self.dropoff_time_window_start = int(dropoff_start);
        self.dropoff_time_window_end = int(dropoff_end);

        # except:
        #     print('balked on adding deliv seg...')
        #     self.pickup_time_window_start = PERSON.pickup_time_window_start;
        #     self.pickup_time_window_end = PERSON.pickup_time_window_end;
        #     self.dropoff_time_window_start = PERSON.dropoff_time_window_start
        #     self.dropoff_time_window_end = PERSON.dropoff_time_window_end



        self.booking_id = PERSON.booking_id;
        self.pickup_node_id = PERSON.pickup_node_id;
        self.dropoff_node_id = PERSON.dropoff_node_id;
        self.am = PERSON.am;
        self.wc = PERSON.wc;


        # start1 = WORLD['main']['start_time']; #seg_details[2]
        # end1 = WORLD['main']['end_time']; #seg_details[4]
        distance = 0;
        travel_time = 0;
        
        current_costs = {'current_distance':distance,'current_time':travel_time,
                         'current_money': 0,'current_conven': 0,'current_switches': 0}
        self.costs = {factor:[] for factor in factors}
        self.costs = {**self.costs,**current_costs}

        self.path = [];
        self.mass = 0;

        #### just for delivery...
        self.delivery_history = [];        
        self.delivery = None;
        self.typ = None;  #shuttle or direct        
        self.current_path = None; #path;          
              
        if 'group' in params: self.group = params['group']
        else: self.group = 'group0'

        NETWORK = NETWORKS[mode];

        booking_id = int(len(NETWORK.booking_ids));
        self.booking_id = booking_id;
        NETWORK.booking_ids.append(booking_id);

        if mode == 'ondemand':
            for _,group in enumerate(ONDEMAND.groups):
                GROUP = ONDEMAND.groups[group];
                time_matrix = GROUP.time_matrix
                shp = np.shape(time_matrix)
                time_matrix = np.block([[time_matrix,np.zeros([shp[0],2])],[np.zeros([2,shp[1]]),np.zeros([2,2]) ]]);
                GROUP.time_matrix = time_matrix;
                self.pickup_node_id = shp[0];
                self.dropoff_node_id = shp[0]+1;


        self.planSeg(mode,GRAPHS,FEEDS,NETWORKS,CONVERTER,ONDEMAND,track=False);

        NETWORK.active_segs.append(seg)

            # if len(PERSON)>0:
            #     if 'group_options' in PERSON: WORLD[mode]['trips'][trip]['group_options'] = PERSON['group_options'];

            # WORLD[mode]['trips'][trip][''] asdfasd



    def planSeg(self,mode,GRAPHS,FEEDS,NETWORKS,CONVERTER,ONDEMAND,track=False):
        NETWORK = NETWORKS[mode];
        if mode == 'drive' or mode == 'walk':
            self.planDijkstraSeg(mode,NETWORK,GRAPHS,track=track);
        if mode == 'gtfs':
            self.planGTFSSeg(mode,NETWORK,GRAPHS,FEEDS,CONVERTER,ONDEMAND,track=track);
        if mode == 'ondemand':
            pass
            #self.planDelivSeg(mode,NETWORKS,GRAPHS,FEEDS,CONVERTER,ONDEMAND,track=track);



    def planDijkstraSeg(self,mode,NETWORK,GRAPHS,track=False,mass=0):

        # NETWORK = NETWORKS[mode]
        GRAPH = GRAPHS[mode];
        try:
            temp = nx.multi_source_dijkstra(GRAPH, [self.source], target=self.target, weight='c'); #Dumbfunctions
            travel_time = temp[0];
            path = temp[1]; 
            distance = 0; 
        except: 
            #print('no path found for bus trip ',trip,'...')
            travel_time = 10000000000;
            distance = 1000000000000;
            path = [];

        self.costs['current_distance'] = distance
        self.costs['current_time'] = travel_time
        self.costs['current_money'] = 1;
        self.costs['current_conven'] = 0;
        self.costs['current_switches'] = 0;
        self.current_path = path;
        self.mass = 0;
                    
        if track==True:
            self.costs['distance'].append(distance)
            self.costs['time'].append(travel_time)
            self.costs['money'].append(1)
            self.costs['conven'].append(1)
            self.costs['switches'].append(1)
            self.path.append(path);

        if mass > 0:
            for j,node in enumerate(path):
                if j < len(path)-1:
                    edge = (path[j],path[j+1],0)
                    edge_mass = NETWORK.edge_masses[edge][-1] + mass;
                    # edge_cost = 1.*edge_mass + 1.;
                    NETWORK.edge_masses[edge][-1] = edge_mass;
                    # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                    # WORLD[mode]['current_edge_costs'][edge] = edge_cost;
                    NETWORK.current_edge_masses[edge] = edge_mass;





    def planGTFSSeg(self,mode,NETWORK,GRAPHS,FEEDS,CONVERTER,ONDEMAND,mass=1,track=True,verbose=False):
        

        # source = trip0[0]; target = trip0[1];
        # source = str(source); target = str(target);
        # trip = (source,target);



        mode = self.mode
        FEED = FEEDS[mode]; #[GRAPHS['gtfs']
        GRAPH = GRAPHS[mode];

        # NETWORK = self;  NETWORKS[mode]

        REACHED = NETWORK.precompute['reached']
        PREV_NODES = NETWORK.precompute['prev_nodes'];
        PREV_TRIPS = NETWORK.precompute['prev_trips'];

        try:
            distance = REACHED[source][-1][target]
            # print(REACHED[source])
        except:
            distance = 1000000000000.;

        try:
            stop_list,trip_list = create_chains(source,target,PREV_NODES,PREV_TRIPS);
            #### Computing Number of switches...
            init_stop = stop_list[0];
            num_segments = 1;
            for i,stop in enumerate(stop_list):
                if stop == init_stop: num_segments = len(stop_list)-i-1;

            _,stopList,edgeList = list_inbetween_stops(FEED,stop_list,trip_list); #,GRAPHS);
            path,_ = gtfs_to_transit_nodesNedges(stopList,NODES)
        except: 
            #print('no path found for bus trip ',trip,'...')
            num_segments = 1;
            path = [];
        # print('num segs gtfs: ',num_segments)                

        self.costs['current_time'] = 1.*distance
        self.costs['current_money'] = 1;
        self.costs['current_conven'] = 1; 
        self.costs['current_switches'] = num_segments - 1; 
        self.current_path = path;
        self.mass = mass;
        
        if track==True:
            self.costs['time'].append(distance)
            self.costs['money'].append(1)
            self.costs['conven'].append(1)
            self.costs['switches'].append(num_segments-1)
            self.path.append(path);

        num_missing_edges = 0;
        if mass > 0:
            #print(path)
            for j,node in enumerate(path):
                if j < len(path)-1:
                    edge = (path[j],path[j+1],0)
                    if edge in NETWORK['edge_masses']:
                        edge_mass = NETWORK['edge_masses'][edge][-1] + mass;
                        # edge_cost = 1.*edge_mass + 1.;
                        NETWORK['edge_masses'][edge][-1] = edge_mass;
                        # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                        # WORLD[mode]['current_edge_costs'][edge] = edge_cost;    
                        NETWORK['current_edge_masses'][edge] = edge_mass;
                    else:
                        num_missing_edges = num_missing_edges + 1
                        # if np.mod(num_missing_edges,10)==0:
                        #     print(num_missing_edges,'th missing edge...')
        if verbose:
            if num_missing_edges > 1:
                print('# edges missing in gtfs segment...',num_missing_edges)



    # def addDelivSeg(seg_details,mode,GRAPHS,DELIVERIES,WORLD,group='group0',mass=1,track=False,PERSON={}):
    def planDelivSeg(self,mode,NETWORKS,GRAPHS,FEEDS,CONVERTER,ONDEMAND,group='group0',mass=1,track=False,PERSON={}): ###### ADDED TO CLASS 


        NETWORK = NETWORKS[mode]

        start1 = self.start_time; #seg_details[2]
        end1 = self.end_time; #seg_details[4]
        

        trip = (source,target);#,start1,start2,end1,end2);
        GRAPH = GRAPHS[mode];
        # if mode == 'transit' and mass > 0:
        # #     print(mode)
        # try:
        #     temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c'); #Dumbfunctions
        #     distance = temp[0];
        #     # if mode == 'walk':
        #     #     distance = distance/1000;
        #     path = temp[1]; 
        # except: 
        #     #print('no path found for bus trip ',trip,'...')
        #     distance = 1000000000000;
        #     path = [];
        distance = 0;

        if not(seg in NETWORK.segs.keys()):

            # print('adding delivery segment...')

            NETWORK.segs[seg] = {};
            NETWORK.segs[seg].costs = {'time':[],'money':[],'conven':[],'switches':[]}
            #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
            NETWORK.segs[seg].path = [];
            NETWORK.segs[seg].delivery_history = [];
            NETWORK.segs[seg].mass = 0;
            NETWORK.segs[seg].delivery = None; #delivery;

            ##### NEW
            NETWORK.segs[seg].typ = None;  #shuttle or direct

            if len(PERSON)>0:
                if hasattr(PERSON,'group_options'): NETWORK.segs[seg].group_options = PERSON.group_options;

            NETWORK.segs[seg].costs['current_time'] = distance;
            NETWORK.segs[seg].costs['current_money'] = 1;
            NETWORK.segs[seg].costs['current_conven'] = 1; 
            NETWORK.segs[seg].costs['current_switches'] = 1; 
            NETWORK.segs[seg].current_path = None; #path;        

            # WORLD[mode]['trips'][trip][''] asdfasdf

            

            NETWORK.segs[seg].am = int(1);
            NETWORK.segs[seg].wc = int(0);


            NETWORK.segs[seg].group = group;

            booking_id = int(len(NETWORK.booking_ids));
            NETWORK.segs[seg].booking_id= booking_id;
            NETWORK.booking_ids.append(booking_id);


            for _,group in enumerate(ONDEMAND.groups):
                GROUP = ONDEMAND.groups[group];
                time_matrix = GROUP.time_matrix;
                shp = np.shape(time_matrix)
                time_matrix = np.block([[time_matrix,np.zeros([shp[0],2])],[np.zeros([2,shp[1]]),np.zeros([2,2]) ]]);
                GROUP.time_matrix = time_matrix;

                NETWORK.segs[seg].pickup_node_id = shp[0];
                NETWORK.segs[seg].dropoff_node_id = shp[0]+1;


            try: 
                ##### NEEDS TO BE UPDATED 
                dist = nx.shortest_path_length(GRAPHS['drive'], source=trip[0], target=trip[1], weight='c');
                maxtriptime = dist*4;
                pickup_start = np.random.uniform(low=start1, high=end1-maxtriptime, size=1)[0];
                pickup_end = pickup_start + 60*20;
                dropoff_start = pickup_start;
                dropoff_end = pickup_end + maxtriptime;

                NETWORK.segs[seg].pickup_time_window_start = int(pickup_start);
                NETWORK.segs[seg].pickup_time_window_end = int(pickup_end);
                NETWORK.segs[seg].dropoff_time_window_start = int(dropoff_start);
                NETWORK.segs[seg].dropoff_time_window_end = int(dropoff_end);

            except:
                print('balked on adding deliv seg...')
                NETWORK.segs[seg].pickup_time_window_start = int(start1);
                NETWORK.segs[seg].pickup_time_window_end = int(end1);
                NETWORK.segs[seg].dropoff_time_window_start = int(start1);
                NETWORK.segs[seg].dropoff_time_window_end = int(end1);










class NETWORK:
    def __init__(self,mode,GRAPHS,FEEDS,params={}):

        self.mode = mode; #params['mode'];
        self.graph = params['graph']
        self.GRAPHS = GRAPHS
        self.FEEDS = FEEDS;
        # self.CONVERTER = CONVERTER
        self.GRAPH = params['GRAPH']
        if 'feed' in params: self.feed = params['feed'];
        else: self.feed = None
        self.segs = {}
        self.edge_masses = {}
        self.edge_costs = {}
        self.current_edge_masses = {}
        self.current_edge_costs = {}
        self.edge_cost_poly = {};
        self.people = [];
        self.active_segs = [];

        self.booking_ids = []; ### new

        nodes = self.GRAPH.nodes
        edges = self.GRAPH.edges
        if mode == 'drive' or mode == 'walk':        
            self.cost_fx = self.createEdgeCosts(self.GRAPH)
        else:
            self.cost_fx = {}; 
        for j,edge in enumerate(edges):
            self.edge_masses[edge] = [0];
            self.edge_costs[edge]=[1];
            self.current_edge_masses[edge]=0;
            if (mode=='drive') & (mode=='walk'): #############  
                self.edge_cost_poly[edge]=self.cost_fx[edge]; ##############
            # if mode == 'ondemand':
            #     WORLD[mode]['current_edge_masses1'][edge]=0;
            #     WORLD[mode]['current_edge_masses2'][edge]=0;
            self.current_edge_costs[edge]=1;

        if mode == 'gtfs':
            # gtfs_data_file = params['gtfs_data_file'];
            gtfs_data_file = params['gtfs_precomputed_file'] #'data/gtfs/gtfs_trips.obj'
            reload_gtfs = True;        
                
            self.precompute = {};
            self.precompute['reached'] = {};
            self.precompute['reached'] = {};
            self.precompute['prev_nodes'] = {};
            self.precompute['prev_trips'] = {};

            if reload_gtfs:
                file = open(gtfs_data_file, 'rb')
                data = pickle.load(file)
                file.close()
                self.precompute['reached'] = data['REACHED_NODES']
                self.precompute['prev_nodes'] = data['PREV_NODES']
                self.precompute['prev_trips'] = data['PREV_TRIPS']


        if mode == 'ondemand':
            self.booking_ids = [];
            self.time_matrix = np.zeros([1,1]);
            self.expected_cost = [0.];
            self.current_expected_cost = 0.;
            self.actual_average_cost = [];


    def createEdgeCosts(self,GRAPH):
    # def createEdgeCosts(mode,GRAPHS):   ##### ADDED TO CLASS 
        POLYS = {};
        if (self.mode == 'drive') or (self.mode == 'ondemand'):
            # keys '25 mph'
            GRAPH = self.GRAPH; #S[mode];
            for i,edge in enumerate(GRAPH.edges):
                EDGE = GRAPH.edges[edge];
                #maxspeed = EDGE['maxspeed'];

                length = EDGE['length']; 
                ##### maxspeed 
                if 'maxspeed' in  EDGE:
                    maxspeed = EDGE['maxspeed'];
                    if isinstance(maxspeed,str): maxspeed = str2speed(maxspeed)
                    elif isinstance(maxspeed,list): maxspeed = np.min([str2speed(z) for z in maxspeed]);
                else: maxspeed = 45; # in mph
                maxspeed = maxspeed * 0.447 # assumed in meters? 1 mph = 0.447 m/s

                ##### lanes 
                if 'lanes' in  EDGE:
                    lanes = EDGE['lanes'];
                    if isinstance(lanes,str): lanes = int(lanes)
                    elif isinstance(lanes,list): lanes = np.min([int(z) for z in lanes]);
                else: lanes = 1.; # in mph

                # 1900 vehicles/hr/lane
                # 0.5278 vehicles /second/lane
                capacity = lanes * 0.5278
                travel_time = length/maxspeed
                pwr = 1;
                t0 = travel_time;
                fac = 0.15; # SHOULD BE 0.15
                t1 = travel_time*fac*np.power((1./capacity),pwr);
                EDGE['cost_fx'] = [t0,0,0,t1];
                # EDGE['inv_cost_fx'] = lambda c,poly : 
                # model for 
                POLYS[edge] = [t0,t1];

        if (self.mode == 'walk'):
            # keys '25 mph'
            GRAPH = self.GRAPH; #S[mode];
            for i,edge in enumerate(GRAPH.edges):
                EDGE = GRAPH.edges[edge];
                #maxspeed = EDGE['maxspeed'];
                maxspeed = 3.; # in mph
                length = EDGE['length'] ; # assumed in meters? 1 mph = 0.447 m/s
                t0 = length/(maxspeed * 0.447)
                EDGE['cost_fx'] = [t0];
                POLYS[edge] = [t0];

        return POLYS            

    def removeMassFromEdges(self): #mode,WORLD,GRAPHS):  #DONE1
        # NETWORK = WORLD[mode]
        # if self.mode=='gtfs':
        #     GRAPH = GRAPHS['transit'];
        # else: 
        #     GRAPH = GRAPHS[mode];
        # edges = list(NETWORK['current_edge_masses']);
        # NETWORK['current_edge_costs'] = {};
        self.current_edge_masses = {};
        for e,edge in enumerate(self.GRAPH.edges):
            # NETWORK['current_edge_costs'][edge] = 0;
            self.current_edge_masses[edge] = 0;        

    def addTripMassToEdges(self,seg): #,NETWORK): DONE1
        if seg in self.segs:
            SEG = self.segs[seg];
            mass = SEG.mass
            for j,node in enumerate(SEG.path):
                if j < len(SEG.path)-1:
                    edge = (SEG.path[j],SEG.path[j+1],0)
                    edge_mass = self.edge_masses[edge][-1] + mass;
                    edge_cost = 1.*edge_mass + 1.;
                    self.edge_masses[edge][-1] = edge_mass;
                    self.edge_costs[edge][-1] = edge_cost;
                    current_mass = self.current_edge_masses[edge];
                    self.current_edge_masses[edge] = edge_cost;            
                    self.current_edge_costs[edge] = edge_cost;






    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 




    def UPDATE(self,params,FEED,ONDEMAND,verbose=True,clear_active=True):

        mode = self.mode;
        kk = params['iter']
        alpha = params['alpha'];

        GRAPHS = self.GRAPHS
        FEEDS = self.FEEDS


        # clear_active = False; 



        if self.mode == 'drive':
        # def world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=False,clear_active=True): ##### ADDED TO CLASS 
            #graph,costs,sources, targets):
            if verbose: print('starting driving computations...')
            GRAPH = self.GRAPH;


            if kk == 0: #(not('edge_masses' in WORLD[mode].keys()) or (kk==0)): #?????/
                self.edge_masses = {};
                self.edge_costs = {};
                self.current_edge_costs = {};
                self.current_edge_masses = {};
                # self.edge_a0 = {};
                # self.edge_a1 = {};
                for e,edge in enumerate(GRAPH.edges):
                    self.edge_masses[edge] = [0]
                    current_edge_cost = self.cost_fx[edge][0];
                    self.edge_costs[edge] = [current_edge_cost]; #WORLD[mode]['cost_fx'][edge][0]];
                    self.current_edge_costs[edge] = current_edge_cost; 
                    self.current_edge_masses[edge] = 0. 
                    # WORLD[mode]['edge_a0'][edge] = 1;
                    # WORLD[mode]['edge_a1'][edge] = 1;

            else: #### GRADIENT UPDATE STEP.... 
                for e,edge in enumerate(GRAPH.edges):
                    # WORLD[mode]['edge_masses'][edge].append(0)
                    current_cost = self.current_edge_costs[edge]
                    poly = self.cost_fx[edge]
                    current_edge_mass = self.current_edge_masses[edge];
                    if hasattr(self,'base_edge_masses'):# in WORLD[mode]:
                        current_edge_mass = current_edge_mass + self.base_edge_masses[edge];
                    expected_mass = invDriveCostFx(current_cost,poly)
                    #diff = WORLD[mode]['current_edge_masses'][edge] - expected_mass;
                    diff = current_edge_mass - expected_mass; 

                    new_edge_cost = self.current_edge_costs[edge] + alpha * diff
                    min_edge_cost = poly[0];
                    max_edge_cost = 10000000.;
                    new_edge_cost = np.min([np.max([min_edge_cost,new_edge_cost]),100000.])
                    self.edge_costs[edge].append(new_edge_cost)
                    self.current_edge_costs[edge] = new_edge_cost;            
                
            # if True: #'current_edge_costs' in WORLD[mode].keys():
            current_costs = self.current_edge_costs;
            # else: # INITIALIZE 
            #     current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}


            # print(current_costs)
            nx.set_edge_attributes(GRAPH,current_costs,'c');
            mode = 'drive'
            self.removeMassFromEdges()
            segs = self.active_segs
            print('...with ',len(segs),' active trips...')    
            for i,seg in enumerate(self.segs):
                if seg in segs: 
                    if np.mod(i,500)==0: print('>>> segment',i,'...')
                    source = seg[0];
                    target = seg[1];
                    seg = (source,target)
                    SEG = self.segs[seg]
                    mass = SEG.mass
                    SEG.planDijkstraSeg(mode,self,GRAPHS,mass=mass,track=True);
                    SEG.active = False;
                else:
                    if len(self.segs[seg].costs['time'])==0:
                        self.segs[seg].costs['time'].append(0)
                    else:
                        recent_value = self.segs[seg].costs['time'][-1]
                        self.segs[seg].costs['time'].append(recent_value)

            if clear_active:
                self.active_segs  = [];

            ######## REMOVING MASS ON TRIPS ##########
            ######## REMOVING MASS ON TRIPS ##########
            for i,seg in enumerate(self.segs):
                SEG = self.segs[seg];
                SEG.mass = 0;


        if self.mode == 'walk':

        ####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
        ####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
        # def world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=False,clear_active=True):  ### ADDED TO CLASS 
                #graph,costs,sources, targets):
            if verbose: print('starting walking computations...')
            # mode = 'walk'
            # kk = WORLD['main']['iter']
            # alpha = WORLD['main']['alpha']

            GRAPH = self.GRAPH; #GRAPHS[WALK['graph']];    
            if (not(hasattr(self,'edge_masses')) or (kk==0)):
                self.edge_masses = {};
                self.edge_costs = {};
                self.current_edge_costs = {};
                self.edge_a0 = {};
                self.edge_a1 = {};
                for e,edge in enumerate(GRAPH.edges):
                    # WALK['edge_masses'][edge] = [0]
                    # WALK['edge_costs'][edge] = [1]
                    # WALK['current_edge_costs'][edge] = 1.;#GRAPH.edges[edge]['cost_fx'][0];
                    # WALK['edge_a0'][edge] = 1;
                    # WALK['edge_a1'][edge] = 1;               
                    self.edge_masses[edge] = [0]
                    current_edge_cost = self.cost_fx[edge][0];
                    self.edge_costs[edge] = [current_edge_cost]; #WORLD[mode]['cost_fx'][edge][0]];
                    self.current_edge_costs[edge] = current_edge_cost; 
                    self.current_edge_masses[edge] = 0. 
                    # WORLD[mode]['edge_a0'][edge] = 1;
                    # WORLD[mode]['edge_a1'][edge] = 1;

            else: 
                for e,edge in enumerate(GRAPH.edges):
                    # WALK['edge_masses'][edge].append(0)
                    # WALK['edge_costs'][edge].append(0)
                    # WALK['current_edge_costs'][edge] = 1;
                    # WORLD[mode]['edge_masses'][edge].append(0)

                    current_edge_cost = self.current_edge_costs[edge];# + alpha * WORLD[mode]['current_edge_masses'][edge] 
                    self.edge_costs[edge].append(current_edge_cost)
                    self.current_edge_costs[edge] = current_edge_cost;            
                
            # if True: #'current_edge_costs' in WORLD[mode].keys():
            current_costs = self.current_edge_costs;
            # else: # INITIALIZE 
            #     current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}        


            nx.set_edge_attributes(self.GRAPH,current_costs,'c');     

            
            self.removeMassFromEdges()  
            segs = self.active_segs
            print('...with ',len(segs),' active trips...')        
            for i,seg in enumerate(self.segs):
                if seg in segs:

                    SEG = self.segs[seg];
                    if np.mod(i,500)==0: print('>>> segment',i,'...')
                    source = seg[0];
                    target = seg[1];
                    # trip = (source,target)
                    mass = self.segs[seg].mass
                    SEG.planDijkstraSeg(mode,self,GRAPHS,mass=mass,track=True);
                    self.segs[seg].active = False;
                else:
                    if len(self.segs[seg].costs['time'])==0:
                        self.segs[seg].costs['time'].append(0)
                    else:
                        recent_value = self.segs[seg].costs['time'][-1]
                        self.segs[seg].costs['time'].append(recent_value)



            if clear_active:    
                self.active_segs  = [];

            ######## REMOVING MASS ON TRIPS ##########
            ######## REMOVING MASS ON TRIPS ##########
            for i,seg in enumerate(self.segs):
                SEG = self.segs[seg];
                SEG.mass = 0;



        if self.mode == 'ondemand':
        # def world_of_ondemand(WORLD,PEOPLE,DELIVERIES,GRAPHS,verbose=False,show_delivs='all',clear_active=True): ##### ADDED TO CLASSA 
            ##### ADDED TO CLASSA 
            if verbose: print('starting on-demand computations...')    

            #### PREV: tsp_wgrps
            #lam = grads['lam']
            #pickups = current_pickups(lam,all_pickup_nodes) ####
            #nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
            # kk = WORLD['main']['iter']
            # alpha = WORLD['main']['alpha']

            # mode = 'ondemand'
            GRAPH = self.GRAPH; #GRAPHS['ondemand'];
            # ONDEMAND = WORLD['ondemand'];

            total_segs_to_plan = self.active_segs;#_direct'] + WORLD['ondemand']['active_segs_shuttle'];
            num_total_segs = len(total_segs_to_plan);

            ##### SORTING TRIPS BY GROUPS: 
            if hasattr(ONDEMAND,'groups'):
                GROUPS_OF_SEGS = {};
                for _,group in enumerate(ONDEMAND.groups):
                    GROUPS_OF_SEGS[group] = [];

                for i,seg in enumerate(self.active_segs):
                    SEG = self.segs[seg];
                    if hasattr(SEG,'group'):
                        group = SEG.group;
                        GROUPS_OF_SEGS[group].append(seg);

                #### TO ADD: SORT TRIPS BY GROUP
                for _,group in enumerate(GROUPS_OF_SEGS):
                # try: 
                    segs_to_plan = GROUPS_OF_SEGS[group]
                    GROUP = ONDEMAND.groups[group]

                    ########### -- XX START HERE ---- ####################
                    ########### -- XX START HERE ---- ####################
                    ########### -- XX START HERE ---- ####################
                    print('...with',len(segs_to_plan),'active ondemand trips...')

                    # poly = WORLD[mode]['cost_poly'];
                    # poly = np.array([-6120.8676711, 306.5130127]) # 1st order
                    # poly = np.array([-8205.87778054,   342.32193064]) # 1st order 
                    # poly = np.array([5047.38255623,-288.78570445,6.31107635]); # 2nd order


                    if hasattr(GROUP,'fit'):
                        poly  = GROUP.fit['poly'];
                    elif hasattr(ONDEMAND,'fit'):
                        poly = ONDEMAND.fit['poly'];
                    else:
                        poly = ONDEMAND['poly'];


                    # poly = np.array([6.31107635, -288.78570445, 5047.38255623]) # 2nd order 


                    num_segs = len(segs_to_plan);
                    ### dumby values...
                    MDF = []; nodes = [];
                    if len(segs_to_plan)>0:
                        nodes_to_update = [GROUP.depot_node];
                        nodeids_to_update = [0];
                        for j,seg in enumerate(segs_to_plan):
                            SEG = self.segs[seg]
                            nodes_to_update.append(seg[0]);
                            nodes_to_update.append(seg[1]);
                            nodeids_to_update.append(SEG.pickup_node_id)
                            nodeids_to_update.append(SEG.dropoff_node_id)


                        GROUP.updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,ONDEMAND); #GRAPHS['ondemand']);
                        payload,nodes = GROUP.constructPayload(segs_to_plan,ONDEMAND,self,GRAPHS);




                        # print('')
                        # print(payload.keys())
                        # print('')
                        # print(payload['date'])
                        # print(payload['depot'])	
                        # print('')
                        # print(payload['driver_runs'])
                        # print('')
                        # print(payload['time_matrix'][0][:10])
                        # print('')
                        # print(payload['requests'])
                        # asdfasdf

                        #asdfasdfasdfasdfasdf
                        # import pickle
                        # fileObj = open('payload1', 'wb')
                        # pickle.dump(payload,fileObj)
                        # fileObj.close()
                        # print(payload['driver_runs'])
                        # print(payload['time_matrix'][0][:10])
                        # print('')
                        # print(payload['requests'][0])
                        # print(payload['requests'][1])
                        # print(payload['requests'][2])
                        # print(payload['requests'][3])
                        # asdfasdf

                        manifests = optimizer.offline_solver(payload) 
                        # manifest = optimizer(payload) ####
                        # manifest = fakeManifest(payload)

                        PDF = GROUP.payloadDF(payload,GRAPHS)
                        MDF = GROUP.manifestDF(manifests,PDF)

                        GROUP.current_PDF = PDF;
                        GROUP.current_MDF = MDF;
                        GROUP.current_payload = payload;
                        GROUP.current_manifest = manifests;


                    #average_time,times_to_average = GROUP.assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,WORLD[mode]['current_expected_cost'])
                    average_time,times_to_average = GROUP.assignCostsFromManifest(segs_to_plan,self.segs,nodes,MDF,self,GROUP.current_expected_cost)

                    # print(poly)
                    # print(WORLD[mode]['current_expected_cost'])

                    # expected_num_trips = invDriveCostFx(WORLD[mode]['current_expected_cost'],poly)
                    expected_num_segs = invDriveCostFx(GROUP.current_expected_cost,poly)            


                    print('...expected num of trips given cost:',expected_num_segs)
                    print('...actual num trips:',num_segs)

                    diff = num_segs - expected_num_segs

                    print('...adjusting cost estimate by',alpha*diff);
                    # new_ave_cost = WORLD[mode]['current_expected_cost'] + alpha * diff
                    new_ave_cost = GROUP.current_expected_cost + alpha * diff

                    min_ave_cost = poly[0];
                    max_ave_cost = 100000000.;
                    new_ave_cost = np.min([np.max([min_ave_cost,new_ave_cost]),max_ave_cost])

                    print('...changing cost est from',GROUP.current_expected_cost,'to',new_ave_cost )

                    GROUP.expected_cost.append(new_ave_cost)
                    GROUP.current_expected_cost = new_ave_cost

                    GROUP.actual_average_cost.append(average_time)
                # except:
                #     GROUP.expected_cost.append(100000000)
                #     GROUP.current_expected_cost = 100000000
                #     GROUP.actual_average_cost.append(100000000)



            for i,seg in enumerate(self.segs):        
                if not(seg in total_segs_to_plan):
                    if len(self.segs[seg].costs['time'])==0:
                        self.segs[seg].costs['time'].append(0)
                        self.segs[seg].costs['current_time']= 0;
                    else:
                        recent_value = self.segs[seg].costs['time'][-1]
                        self.segs[seg].costs['time'].append(recent_value)
                        self.segs[seg].costs['current_time'] = recent_value

            if clear_active: 
                self.active_segs  = [];



            # else:
                ########### -- XX START HERE ---- ####################
                ########### -- XX START HERE ---- ####################
                ########### -- XX START HERE ---- ####################


        if self.mode == 'gtfs':

        ####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 
        ####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 
        # def world_of_gtfs(WORLD,PEOPLE,GRAPHS,NODES,verbose=False,clear_active=True):  ##### ADDED TO CLASS 
            if verbose: print('starting gtfs computations...')
            #raptor_gtfs
            # kk = WORLD['main']['iter']
            # alpha = WORLD['main']['alpha']

            # GTFS = WORLD['gtfs'];
            GRAPH = self.GRAPH; #GRAPHS['transit']; 
            # FEED = self.FEED; #GRAPHS['gtfs'];
            if (not(hasattr(self,'edge_masses_gtfs')) or (kk==0)):
                self.edge_costs = {};
                self.edge_masses = {};
                self.current_edge_costs = {};
                self.edge_a0 = {};
                self.edge_a1 = {};        
                for e,edge in enumerate(GRAPH.edges):
                    self.edge_masses[edge] = [0]
                    self.edge_costs[edge] = [1]
                    self.current_edge_costs[edge] = 1;
                    self.edge_a0[edge] = 1;
                    self.edge_a1[edge] = 1;               
                
            if hasattr(self,'current_edge_costs'):
                current_costs = self.current_edge_costs;
            else: 
                current_costs = {k:v for k,v in zip(self.GRAPH.edges,np.ones(len(self.GRAPH.edges)))}
                
            #nx.set_edge_attributes(GRAPH,current_costs,'c');

            mode = self.mode
            self.removeMassFromEdges() 
            segs = self.active_segs
            print('...with ',len(segs),' active trips...')        

            for i,seg in enumerate(self.segs):
            # for i,seg in enumerate(segs):
                if seg in segs:
                    source = seg[0];
                    target = seg[1];
                    # trip = (source,target)
                    SEG = self.segs[seg];
                    mass = SEG.mass
                    SEG.planGTFSSeg(mode,self,GRAPHS,FEEDS,CONVERTER,ONDEMAND,mass=mass,track=True);
                    self.segs[seg].active = False;
                else:
                    if len(self.segs[seg].costs['time'])==0:
                        self.segs[seg].costs['time'].append(0)
                    else:
                        recent_value = self.segs[seg].costs['time'][-1]
                        self.segs[seg].costs['time'].append(recent_value)
                    self.segs[seg].active = False;                        


            if clear_active:
                self.active_segs  = [];
                ######## REMOVING MASS ON TRIPS ##########
                ######## REMOVING MASS ON TRIPS ##########
                print('REMOVING MASS FROM GTFS TRIPS...')
                for i,seg in enumerate(self.segs):
                    SEG = self.segs[seg];
                    SEG.mass = 0;

        
        # print('ACTIVE SEGS FOR MODE -- ',self.mode,'--')
        # print(self.active_segs)



    def computeUncongestedEdgeCosts(self):  #### ADDED TO CLASS 
    # def compute_UncongestedEdgeCosts(WORLD,GRAPHS):  #### ADDED TO CLASS 
        # modes = ['drive','walk'];
        # for mode in modes:
        GRAPH = self.GRAPH
        edge_costs0 = {}
        for e,edge in enumerate(GRAPH.edges):
            poly = self.cost_fx[edge];
            edge_costs0[edge] = poly[0]
        nx.set_edge_attributes(GRAPH,edge_costs0,'c0')

    def computeUncongestedTripCosts(self,verbose=True): #WORLD,GRAPHS):   ### ADDED TO CLASS 
    # def compute_UncongestedTripCosts(WORLD,GRAPHS):   ### ADDED TO CLASS         

        if self.mode == 'ondemand': GRAPH = self.GRAPHS['drive']
        else: GRAPH = self.GRAPHS[self.mode]
        if verbose == True: print('computing uncongested trip costs for mode',self.mode,'...')
        for t,seg in enumerate(self.segs):
            SEG  = self.segs[seg]
            if verbose == True:
                if np.mod(t,50)==0: print('...segment number',t);
            source = seg[0]; target = seg[1];
            try:
                temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c0'); #Dumbfunctions
                distance = temp[0];
                path = temp[1]; 
            except: 
                #print('no path found for bus trip ',trip,'...')
                distance = 1000000000000;
                path = [];
            SEG.uncongested = {};
            SEG.uncongested['path'] = path;
            SEG.uncongested['costs'] = {}; 
            SEG.uncongested['costs']['time'] = distance;
            SEG.uncongested['costs']['money'] = 0;
            SEG.uncongested['costs']['conven'] = 0;
            SEG.uncongested['costs']['switches'] = 0;        



    def NOTEBOOK_UNCONGESTED_TRIPS(self):

        compute_UncongestedEdgeCosts(WORLD,GRAPHS)
        compute_UncongestedTripCosts(WORLD,GRAPHS)

        # datanames = ['full','2regions','4regions','small','tiny'];
        #datanames = ['large1']; #,'medium1','small1','tiny1'];
        datanames = ['regions2','regions4','regions7','tiny1']
        filenames = {name:name+'.obj' for name in datanames}
        folder = 'runs/'

        print('COMPUTING DATA FOR DASHBOARD...')
        # DATA = computeData(WORLD,GRAPHS,DELIVERY)
        DATAS = loadDataRuns(folder,filenames,GRAPHS);        

    ####### ===================== TRIP COMPUTATION ======================== ################
    ####### ===================== TRIP COMPUTATION ======================== ################
    ####### ===================== TRIP COMPUTATION ======================== ################






# class PEOPLE: 

class PERSON:

    def __init__(self,person,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND,PRE,params): 

        mass_scale = 1.;
        modes = params['modes']
        factors = params['factors']

        DELIVERY = ONDEMAND

        mean = 500; stdev = 25;
        means = [mean,mean,mean,mean,mean];
        stdevs = [stdev,stdev,stdev,stdev,stdev];
        STATS = {}
        for m,mode in enumerate(modes):
            STATS[mode] = {};
            STATS[mode]['mean'] = dict(zip(factors,means));
            STATS[mode]['stdev'] = dict(zip(factors,stdevs));



        # if np.mod(i,print_interval)==0:
        #     end_time3 = time.time()
        #     if verbose:
        #         print(person)
        #         print('time to add',print_interval, 'people: ',end_time3-start_time3)
        #     start_time3 = time.time()

        self.CONVERTER = CONVERTER
        self.NETWORKS = NETWORKS
        self.GRAPHS = GRAPHS
        self.FEEDS = FEEDS;
        self.ONDEMAND = ONDEMAND;

        self.mass = None; 
        self.current_choice = None;
        self.current_cost = 0;
        self.choice_traj = [];
        self.delivery_grps = {'straight':None,'initial':None,'final':None};
        self.delivery_grp = None;
        self.delivery_grp_initial = None;
        self.delivery_grp_final = None;
        self.logit_version = 'weighted_sum';
        self.trips = [];
        self.opts = ['drive','ondemand','walk','transit'];
        self.opts2 = [random.choice(['walk','ondemand','drive']),'transit',random.choice(['walk','ondemand','drive'])];
        self.cost_traj = [];

        self.am =  int(1);
        self.wc = int(0);


        self.pickup_time_window_start = 0;
        self.pickup_time_window_end = 14400;
        self.dropoff_time_window_start = 0;
        self.dropoff_time_window_end = 14400;

        self.time_windows = {};


        current_booking_id = len(ONDEMAND.booking_ids)
        self.booking_id = current_booking_id;
        self.pickup_node_id = 1;
        self.dropoff_node_id = 2;

        self.mass_total = PRE[person]['pop']
        self.mass = PRE[person]['pop']*mass_scale

        if False:
            orig_loc = ORIG_LOC[i];
            dest_loc = DEST_LOC[i];
        else:
            orig_loc = PRE[person]['orig_loc'];
            dest_loc = PRE[person]['dest_loc'];


        self.pickup_pt = {'lat': 35.0296296, 'lon': -85.2301767};
        self.dropoff_pt = {'lat': 35.0734152, 'lon': -85.1315328};

        # PRE[person]['weight'][mode][factor]

        #### PREFERENCES #### PREFERENCES #### PREFERENCES #### PREFERENCES 
        ###################################################################
        if 'logit_version' in PRE[person]: self.logit_version = PRE[person]['logit_version']; #OVERWRITES ABOVE

        self.costs = {}; #opt, factor
        self.prefs = {}; #opt, factor                    
        self.weights = {}; #opt, factor

        for m,mode in enumerate(modes):
            self.costs[mode] = {};
            self.prefs[mode] = {};
            self.weights[mode] = {};
            for j,factor in enumerate(factors):
                sample_pt = STATS[mode]['mean'][factor] + STATS[mode]['stdev'][factor]*(np.random.rand()-0.5)
                self.prefs[mode][factor] = 0; #sample_pt
                self.costs[mode][factor] = 0.

                try:
                    self.weights[mode][factor] = PRE[person]['weight'][mode][factor]
                except:
                    if factor == 'time': self.weights[mode][factor] = 1.
                    else: self.weights[mode][factor] = 1.
            # PERSON['weights'][mode] = dict(zip(factors,np.ones(len(factors))));

        ###### DELIVERY ###### DELIVERY ###### DELIVERY ###### DELIVERY ###### DELIVERY 
        ###############################################################################




        picked_deliveries = {'direct':None,'initial':None,'final':None}
        # person_loc = orig_loc
        # dist = 10000000;
        # picked_delivery = None;    
        # for k,delivery in enumerate(DELIVERY['direct']):
        #     DELIV = DELIVERY['direct'][delivery]
        #     loc = DELIV['loc']

        #     diff = np.array(list(person_loc))-np.array(list(loc));
        #     if mat.norm(diff)<dist:
        #         PERSON['delivery_grps']['direct'] = delivery
        #         dist = mat.norm(diff);
        #         picked_deliveries['direct'] = delivery;


        picked_deliveries = {'direct':None,'initial':None,'final':None}        
        person_loc = orig_loc
        dist = 10000000;
        picked_delivery = None;    
        for k,delivery in enumerate(ONDEMAND.DELIVERY['shuttle']):
            DELIV = ONDEMAND.DELIVERY['shuttle'][delivery]
            loc = DELIV['loc']
            diff = np.array(list(person_loc))-np.array(list(loc));
            if mat.norm(diff)<dist:
                self.delivery_grps['initial'] = delivery
                dist = mat.norm(diff);
                picked_deliveries['initial'] = delivery;            


        person_loc = dest_loc;
        dist = 10000000;
        picked_delivery = None;    
        for k,delivery in enumerate(ONDEMAND.DELIVERY['shuttle']):
            DELIV = ONDEMAND.DELIVERY['shuttle'][delivery]
            loc = DELIV['loc']
            diff = np.array(list(person_loc))-np.array(list(loc));
            if mat.norm(diff)<dist:
                self.delivery_grps['final'] = delivery
                dist = mat.norm(diff);
                picked_deliveries['final'] = delivery;              

        # if not(picked_deliveries['direct']==None):
        #     DELIVERY['direct'][picked_deliveries['direct']]['people'].append(person)
        if not(picked_deliveries['initial']==None):        
            ONDEMAND.DELIVERY['shuttle'][picked_deliveries['initial']]['people'].append(person)    
        if not(picked_deliveries['final']==None):        
            ONDEMAND.DELIVERY['shuttle'][picked_deliveries['final']]['people'].append(person)             




        ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 
        ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 
        ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1     

        
        self.trips = {}; 


        if 'seg_types' in PRE[person]:
            seg_types = PRE[person]['seg_types']
            self.trips_to_consider = PRE[person]['seg_types']

        else: 
            samp = np.random.rand(1);
            if PRE[person]['take_car'] == 0.:
                self.trips_to_consider = [('ondemand',),
                             ('walk','gtfs','walk'),
                             ('walk','gtfs','ondemand'),
                             ('ondemand','gtfs','walk'),
                             ('ondemand','gtfs','ondemand')];              
            elif (PRE[person]['take_car']==1) & (samp < 0.3):
                self.trips_to_consider = [('drive',),
                             ('ondemand',),
                             ('walk','gtfs','walk'),
                             ('walk','gtfs','ondemand'),
                             ('ondemand','gtfs','walk'),
                             ('ondemand','gtfs','ondemand')];
            else:
                self.trips_to_consider = [('drive',)]

        start_time4 = time.time()

        new_trips_to_consider = [];
        for _,segs in enumerate(self.trips_to_consider):
            add_new_trip = True; 
            end_time4 = time.time()
            #print('trip time: ',end_time4-start_time4)
            start_time4 = time.time()
            start_time2 = time.time()

            #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT 
            #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT 
            #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT #### 1-SEGMENT 

            if len(segs)==1:

                start_time = time.time();
                mode1 = segs[0];
                start_node = ox.distance.nearest_nodes(GRAPHS[mode1], orig_loc[0],orig_loc[1]);        
                end_node = ox.distance.nearest_nodes(GRAPHS[mode1], dest_loc[0],dest_loc[1]); 


                CONVERTER.addNodeToConverter(start_node,mode1,'graph');#GRAPHS); #,NODES);
                CONVERTER.addNodeToConverter(end_node,mode1,'graph');#GRAPHS); #,NODES);
                #updateNodesDF(NODES);

                nodes_temp = [{'nodes':[start_node],'type':mode1},
                              {'nodes':[end_node],'type':mode1}]


                ##### SELECTING GROUP ######
                if mode1 == 'ondemand':
                    grouptag,groupind = ONDEMAND.selectDeliveryGroup((start_node,end_node),GRAPHS,typ='direct')
                    if len(grouptag)==0: add_new_trip = False;
                    else:  ONDEMAND.grpsDF.iloc[groupind]['num_possible_trips'] = ONDEMAND.grpsDF.iloc[groupind]['num_possible_trips'] + 1;



                ############################
                deliveries_temp = [];
                deliveries_temp2 = [];
                if mode1=='ondemand':
                    # if not(picked_deliveries['direct']==None):
                    #     DELIVERY['direct'][picked_deliveries['direct']]['sources'].append(start_node);
                    #     DELIVERY['direct'][picked_deliveries['direct']]['targets'].append(end_node);
                    # deliveries_temp.append(picked_deliveries['direct'])


                    # deliveries_temp2.append('group0'); #### EDITTING 
                    deliveries_temp2.append(grouptag); #### EDITTING 

                end_time = time.time();
                #print("segment1: ",np.round(end_time-start_time,4))


            #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT 
            #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT 
            #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT 
            #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT 
            #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT 
            #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT #### 3-SEGMENT 

            if len(segs)==3:
                mode1 = segs[0]; 
                mode2 = segs[1];
                mode3 = segs[2];
                start_time = time.time();

                start_node = ox.distance.nearest_nodes(GRAPHS[mode1], orig_loc[0],orig_loc[1]);        
                end_node = ox.distance.nearest_nodes(GRAPHS[mode3], dest_loc[0],dest_loc[1]);        

                end_time = time.time();
                #print("nearest nodes: ",np.round(end_time-start_time,4))
                ##### VARIATION ##### VARIATION ##### VARIATION ##### VARIATION ##### VARIATION 

                start_time = time.time()


                if (mode1 == 'walk') or (mode3 == 'walk'):
                    transit1_walk,transit2_walk = nearest_applicable_gtfs_node(mode2,
                                                                 GRAPHS,NETWORKS['gtfs'],CONVERTER,
                                                                 orig_loc[0],orig_loc[1],
                                                                 dest_loc[0],dest_loc[1]);
                                                                 # rad1=rad1,rad2=rad2);  ####HERE 
                    # print('transit node 1 is type...',type(transit2_walk))

                if mode1 == 'ondemand':
                    initial_delivery = self.delivery_grps['initial'];
                    transit_node1 = ONDEMAND.DELIVERY['shuttle'][initial_delivery]['nodes']['transit2'];
                    if mode2=='gtfs':
                        transit_node1 = CONVERTER.convertNode(transit_node1,'gtfs','gtfs',to_type='feed')                
                elif mode1 == 'walk':
                    transit_node1 = transit1_walk;
                    # transit_node1 = nearest_nodes(mode2,GRAPHS,NODES,orig_loc[0],orig_loc[1]);  ####HERE 
                    # print('old transit node 1 is type...',type(transit_node1))
                    #transit_node1 = ox.distance.nearest_nodes(GRAPHS[mode2], orig_loc[0],orig_loc[1]);

                if mode3 == 'ondemand':
                    final_delivery = self.delivery_grps['final'];
                    transit_node2 = ONDEMAND.DELIVERY['shuttle'][final_delivery]['nodes']['transit2'];                                
                    if mode2=='gtfs':
                        transit_node2 = CONVERTER.convertNode(transit_node2,'gtfs','gtfs',to_type='feed')

                elif mode3 =='walk':
                    transit_node2 = transit2_walk
                    # transit_node2 = nearest_applicable_gtfs_node(mode2,GRAPHS,WORLD,NODES,dest_loc[0],dest_loc[1]); ##### HERE..
                    # transit_node2 = nearest_nodes(mode2,GRAPHS,NODES,dest_loc[0],dest_loc[1]); ##### HERE..


                    #transit_node2 = ox.distance.nearest_nodes(GRAPHS[mode2], dest_loc[0],dest_loc[1]);
                end_time = time.time();
                #print("something 1: ",np.round(end_time-start_time,4))

                ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------


                ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK 
                ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK             

                start_time = time.time()
                CONVERTER.addNodeToConverter(start_node,mode1,'graph'); #,GRAPHS,NODES);
                CONVERTER.addNodeToConverter(end_node,mode3,'graph'); #,GRAPHS,NODES);            
                CONVERTER.addNodeToConverter(transit_node1,mode2,'graph'); #,GRAPHS,NODES);
                CONVERTER.addNodeToConverter(transit_node2,mode2,'graph'); #,GRAPHS,NODES);
                

                #updateNodesDF(NODES);            
                end_time = time.time();
                #print("add nodes: ",np.round(end_time-start_time,4))

                ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------  
                ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------            


                start_time = time.time()
                nodes_temp = [{'nodes':[start_node],'type':mode1},
                         {'nodes':[transit_node1],'type':mode2},
                         {'nodes':[transit_node2],'type':mode2},
                         {'nodes':[end_node],'type':mode3}]


                ##### SELECTING GROUP ######


                # transit_node1 = WORLD.CONVERTER.convertNode(transit_node1,'transit','gtfs',NODES)

                if mode1 == 'ondemand':
                    if mode2=='gtfs':
                        transit_node_converted = CONVERTER.convertNode(transit_node1,'gtfs','ondemand')
                    grouptag1,groupind1 = ONDEMAND.selectDeliveryGroup((start_node,transit_node_converted),GRAPHS,typ='shuttle')
                    if len(grouptag1)==0: add_new_trip = False;
                    else:  ONDEMAND.grpsDF.iloc[groupind1]['num_possible_trips'] = ONDEMAND.grpsDF.iloc[groupind1]['num_possible_trips'] + 1;

                if mode3=='ondemand':
                    if mode2=='gtfs':
                        transit_node_converted = CONVERTER.convertNode(transit_node2,'gtfs','ondemand')
                    grouptag3,groupind3 = ONDEMAND.selectDeliveryGroup((transit_node_converted,end_node),GRAPHS,typ='shuttle')
                    if len(grouptag3)==0: add_new_trip = False;
                    else:  ONDEMAND.grpsDF.iloc[groupind3]['num_possible_trips'] = ONDEMAND.grpsDF.iloc[groupind3]['num_possible_trips'] + 1;





                deliveries_temp = [None,None,None];
                deliveries_temp2 = [None,None,None];
                if mode1=='ondemand':  
                    if not(picked_deliveries['initial']==None):                
                        ONDEMAND.DELIVERY['shuttle'][picked_deliveries['initial']]['sources'].append(start_node);
                        deliveries_temp.append(picked_deliveries['initial'])
                    # deliveries_temp2[0] = 'group1' #### EDITTING 
                    deliveries_temp2[0] = grouptag1 #### EDITTING 
                if mode3=='ondemand':
                    if not(picked_deliveries['final']==None):
                        ONDEMAND.DELIVERY['shuttle'][picked_deliveries['final']]['targets'].append(end_node);
                        deliveries_temp.append(picked_deliveries['final'])
                    # deliveries_temp2[2] = 'group1'  #### EDITTING 
                    deliveries_temp2[2] = grouptag3  #### EDITTING 

                end_time = time.time();
                #print("something 2: ",np.round(end_time-start_time,4))


            if add_new_trip:
                new_trips_to_consider.append(segs)
                ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK 
                ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK ### TIME SINK             
                start_time = time.time()
                # TRIP = makeTrip(segs,nodes_temp,NODES,deliveries_temp2,deliveries_temp)
                self.trips[segs] = TRIP(segs,nodes_temp,CONVERTER,deliveries_temp2);#,deliveries_temp);
                end_time = time.time()
            #print('time to make trip: ',np.round(end_time-start_time,4))

        self.trips_to_consider = new_trips_to_consider 
        # ONDEMAND.grpsDF = grpsDF.copy();
        ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------  
        ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------            
    # return PEOPLE


    def UPDATE(self,NETWORKS,takeall=False):
    # def update_choices(PEOPLE, DELIVERY, NODES, GRAPHS, WORLD, version=1,verbose=False,takeall=False):
        # if verbose: print('updating choices')
        ## clear options
        #     for o,opt in enumerate(WORLD):
        #         WORLD[opt]['people'] = [];

        # people_chose = {};
        # for i,person in enumerate(PEOPLE):
        #     if np.mod(i,200)==0: print(person,'...')
        #     PERSON = PEOPLE[person];
            
        delivery_grp = self.delivery_grp;        
        delivery_grp_inital = self.delivery_grp_initial;
        delivery_grp_final = self.delivery_grp_final;        
        COMPARISON = [];

        trips_to_consider = self.trips_to_consider;

        for k,trip in enumerate(trips_to_consider):
            TRIP = self.trips[trip];
            # try: 
            TRIP.queryTrip(self,self.CONVERTER,self.GRAPHS,self.FEEDS,self.NETWORKS,self.ONDEMAND)
            COMPARISON.append(TRIP.current['cost']);
            cost = TRIP.current['cost'];
                # print('cost of',trip,'...',cost)
            # except:
            #     COMPARISON.append(10000000000000000);

        ind = np.argmin(COMPARISON);
        trip_opt = trips_to_consider[ind];

        current_choice = ind;
        current_cost = COMPARISON[ind]

        self.current_choice = current_choice;
        self.current_cost = current_cost;

        self.choice_traj.append(current_choice);
        self.cost_traj.append(current_cost);
        # updating world choice...

        if takeall==False:
            tripstotake = [self.trips[trip_opt]];
        else:
            tripstotake = [self.trips[zz] for _,zz in enumerate(trips_to_consider)];


        for _,CHOSEN_TRIP in enumerate(tripstotake):
            num_segs = len(CHOSEN_TRIP.structure)
            for k,leg in enumerate(CHOSEN_TRIP.structure):
                # print(leg)
                # try:
                mode = leg['mode'];
                NETWORK = NETWORKS[mode]

                # if mode == 'gtfs':
                # start = leg['start_nodes'][leg['opt_start']]
                # end = leg['end_nodes'][leg['opt_end']]
                # else:
                start = leg['opt_start']
                end = leg['opt_end']
                #print(WORLD[mode]['trips'][(start,end)])
                #if not(mode in ['transit']):

                # print(mode)
                # print((start,end))
                # if mode=='transit':
                #     print(WORLD[mode]['trips'])
                # try: 
                seg = (start,end)

                if seg in NETWORK.segs:
                    NETWORK.segs[seg].mass = NETWORK.segs[seg].mass + self.mass;
                    NETWORK.segs[seg].active = True; 
                    # print('trying to add seg...',seg)
                    if not(seg in NETWORK.active_segs):
                        # print('ADDING SEG...',seg)
                        NETWORK.active_segs.append(seg);

                    # if (num_segs == 3) & (mode == 'ondemand'):
                    #     # NETWORK.active_segs_shuttle.append(seg);
                    #     NETWORK.active_segs.append(seg);                            
                    # if (num_segs == 1) & (mode == 'ondemand'):
                    #     # NETWORK.active_segs_direct.append(seg);                            
                    #     NETWORK.active_segs.append(seg);                            

                else:
                    pass#print('missing segment...')

                # except:
                #     pass #print('trip balked for mode ',mode)
                #     continue



class RAPTOR:
    def __init__(self,params): pass



###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 
###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR ###### RAPTOR 





    ############ =================== RAPTOR FUNCTIONS ===================== #####################
    ############ =================== RAPTOR FUNCTIONS ===================== #####################
    ############ =================== RAPTOR FUNCTIONS ===================== #####################


    # the following functions implement the RAPTOR algorithm to compute shortest paths using GTFS feeds... 
    ############ ----------------- MAIN FUNCTIONS ---------------- #################
    ############ ----------------- MAIN FUNCTIONS ---------------- #################

    def get_trip_ids_for_stop(feed, stop_id, departure_time,max_wait=100000*60): ### ADDED TO CLASS
        """Takes a stop and departure time and get associated trip ids.
        max_wait: maximum time (in seconds) that passenger can wait at a bus stop.
        --> actually important to keep the algorithm from considering too many trips.  

        """
        mask_1 = feed.stop_times.stop_id == stop_id 
        mask_2 = feed.stop_times.departure_time >= departure_time # departure time is after arrival to stop
        mask_3 = feed.stop_times.departure_time <= departure_time + max_wait # deparature time is before end of waiting period.
        potential_trips = feed.stop_times[mask_1 & mask_2 & mask_3].trip_id.unique().tolist() # extract the list of qualifying trip ids
        return potential_trips


    def get_trip_profile(feed, stop1_ids,stop2_ids,stop1_times,stop2_times): ### ADDED TO CLASS
        for i,stop1 in enumerate(stop1_ids):
            stop2 = stop2_ids[i];
            time1 = stop1_times[i];
            time2 = stop2_times[i];
            stop1_mask = feed.stop_times.stop_id == stop1
            stop2_mask = feed.stop_times.stop_id == stop2
            time1_mask = feed.stop_times.departure_time == time1
            time2_mask = feed.stop_times.arrival_time == time2
        potential_trips = feed.stop_times[stop1_mask & stop2_mask & time1_mask & time2_mask]



    def stop_times_for_kth_trip(params):  ### ADDED TO CLASS
        # max_wait in minutes...
        # prevent upstream mutation of dictionary 

        # IMPORTING PARAMETERS & ASSIGNING DEFAULTS....
        feed = params['feed'];
        prev_trips_list = params['prev_trips'].copy();
        prev_stops_list = params['prev_stops'].copy();
        from_stop_id = params['from_stop_id'];
        stop_ids = list(params['stop_ids']);
        time_to_stops = params['time_to_stops'].copy();  # time to reach each stop with k-1 trips...

        # number of stops to jump by.... if 1 checks every stop; useful hack for speeding up algorithm
        if not('stop_skip_num' in params):
            stop_skip_num = 1;
        else: 
            stop_skip_num = params['stop_skip_num'];
        # maximum time to wait at each 
        if not('max_wait' in params):
            max_wait = 15*60;
        else: 
            max_wait = params['max_wait'];
        departure_secs = params['departure_secs']
        
    #   print('NUM OF STOP IDS: ',len(stop_ids))
        for i, ref_stop_id in enumerate(stop_ids):
            # how long it took to get to the stop so far (0 for start node)
            baseline_cost = time_to_stops[ref_stop_id]
            # get list of all trips associated with this stop
            potential_trips = get_trip_ids_for_stop(feed, ref_stop_id, departure_secs+time_to_stops[ref_stop_id],max_wait)
    #         print('num potential trips: ',len(potential_trips))
            for potential_trip in potential_trips:
                # get all the stop time arrivals for that trip
                stop_times_sub = feed.stop_times[feed.stop_times.trip_id == potential_trip]
                stop_times_sub = stop_times_sub.sort_values(by="stop_sequence")
                # get the "hop on" point
                from_here_subset = stop_times_sub[stop_times_sub.stop_id == ref_stop_id]
                from_here = from_here_subset.head(1).squeeze()
                # get all following stops
                stop_times_after_mask = stop_times_sub.stop_sequence >= from_here.stop_sequence
                stop_times_after = stop_times_sub[stop_times_after_mask]
                stop_times_after = stop_times_after[::stop_skip_num]
                # for all following stops, calculate time to reach
                arrivals_zip = zip(stop_times_after.arrival_time, stop_times_after.stop_id)        
                # for arrive_time, arrive_stop_id in enumerate(list(arrivals_zip)):            
                for i,out in enumerate(list(arrivals_zip)):
                    arrive_time = out[0]
                    arrive_stop_id = out[1];
                    # time to reach is diff from start time to arrival (plus any baseline cost)
                    arrive_time_adjusted = arrive_time - departure_secs + baseline_cost
                    # only update if does not exist yet or is faster
                    if arrive_stop_id in time_to_stops:
                        if time_to_stops[arrive_stop_id] > arrive_time_adjusted:
                            time_to_stops[arrive_stop_id] = arrive_time_adjusted
                            prev_stops_list[arrive_stop_id] = ref_stop_id
                            prev_trips_list[arrive_stop_id] = potential_trip
                            
                    else:
                        time_to_stops[arrive_stop_id] = arrive_time_adjusted
                        prev_stops_list[arrive_stop_id] = ref_stop_id
                        prev_trips_list[arrive_stop_id] = potential_trip                    
                        
                        
        return time_to_stops,prev_stops_list,prev_trips_list;#,stop_times_after.stop_id #,departure_times,arrival_times

    def compute_footpath_transfers(stop_ids,time_to_stops_inputs,stops_gdf,transfer_cost,FOOT_TRANSFERS):  ### ADDED TO CLASS
        # stops_ids = params['stop_ids'];
        # time_to_stops_inputs = params['time_to_stops_inputs'];
        # stops_gdf = params['stops_gdf'];
        # transfer_cost = params['transfer_cost'];
        # FOOT_TRANSFERS = params['FOOT_TRANSFERS'];
        
        # prevent upstream mutation of dictionary


        time_to_stops = time_to_stops_inputs.copy()
        stop_ids = list(stop_ids)
        # add in transfers to nearby stops
        for stop_id in stop_ids:
            foot_transfers = FOOT_TRANSFERS[stop_id]
            for k, arrive_stop_id in enumerate(foot_transfers):
                arrive_time_adjusted = time_to_stops[stop_id] + foot_transfers[arrive_stop_id];            
                if arrive_stop_id in time_to_stops:
                    if time_to_stops[arrive_stop_id] > arrive_time_adjusted:
                        time_to_stops[arrive_stop_id] = arrive_time_adjusted
                else:
                    time_to_stops[arrive_stop_id] = arrive_time_adjusted
        return time_to_stops






    # params = {};         
    # params = {from_stops: from_bus_stops, max_wait: max_wait }

    def raptor_shortest_path(params): ### ADDED TO CLASS
        feed = params['feed']
        from_stop_ids = params['from_stops'].copy();
        transfer_limit = params['transfer_limit']
        max_wait = params['max_wait'];
        add_footpath_transfers = params['add_footpath_transfers']
        gdf = params['gdf'];
        foot_transfer_cost = params['foot_transfer_cost']    
        FOOT_TRANSFERS = params['FOOT_TRANSFERS']    
        stop_skip_num = params['stop_skip_num']
        departure_secs = params['departure_secs']
        REACHED_BUS_STOPS = {};
        TIME_TO_STOPS = {};
        PREV_NODES = {};    
        PREV_TRIPS = {};        
        for i,from_stop_id in enumerate(from_stop_ids):
            start_time4 = time.time()
            if np.mod(i,1)==0:
                print('stop number: ',i)
            REACHED_BUS_STOPS[from_stop_id] = [];
            PREV_NODES[from_stop_id] = [];        
            PREV_TRIPS[from_stop_id] = [];                
            TIME_TO_STOPS[from_stop_id] = 0;
            init_start_time1 = time.time();
            
            
            time_to_stops = {from_stop_id : 0}
            list_of_stops = {from_stop_id : from_stop_id}
            list_of_trips = {from_stop_id : None}
            #arrrival_times = {from_stop_id:0}
            
            #     for j in range(len(types)):
            #         REACHED_NODES[types[j]][from_stop_id] = [];    
            for k in range(transfer_limit + 1):
                start_time = time.time();
                stop_ids = list(time_to_stops)
                prev_stops = list_of_stops
                prev_trips = list_of_trips
                params2 = {}
                params2['feed'] = feed;
                params2['prev_trips'] = prev_trips;
                params2['prev_stops'] = prev_stops;
                params2['from_stop_id'] = from_stop_id;
                params2['stop_ids'] = stop_ids;
                params2['time_to_stops'] = time_to_stops;
                params2['stop_skip_num'] = stop_skip_num;
                params2['max_wait'] = max_wait
                params2['departure_secs'] = departure_secs;
                time_to_stops,list_of_stops,list_of_trips = stop_times_for_kth_trip(params2)
    #             if k==2:
    #                 print(time_to_stops)
    #                 asdf
                if (add_footpath_transfers):
                    time_to_stops = compute_footpath_transfers(stop_ids, time_to_stops, gdf,foot_transfer_cost,FOOT_TRANSFERS)    
                end_time = time.time();
        
                REACHED_BUS_STOPS[from_stop_id].append(time_to_stops);
                PREV_NODES[from_stop_id].append(list_of_stops);
                PREV_TRIPS[from_stop_id].append(list_of_trips);
    #             for l in range(len(types)):
    #                 REACHED_NODES[types[l]][from_stop_id].append([]);
    #                 if (l>=1):
    #             for j, bus_stop in enumerate(list(time_to_stops.keys())):
    #                 if bus_stop in list(BUS_STOP_NODES['drive'].keys()):
    #                     REACHED_NODES['drive'][from_stop_id][k].append(BUS_STOP_NODES['drive'][bus_stop]);
                    
            TIME_TO_STOPS[from_stop_id] = time_to_stops.copy();
            end_time4 = time.time()
            print('time to add stop is...',end_time4- start_time4)

            
        return TIME_TO_STOPS,REACHED_BUS_STOPS,PREV_NODES,PREV_TRIPS;


    def get_trip_lists(feed,orig,dest,stop_plan,trip_plan):    ### ADDED TO CLASS
        stop_chain = [];
        trip_chain = [];
        prev_stop = dest;
        for i,stop_leg in enumerate(stop_plan[::-1]):
            trip_leg = trip_plan[-i]
            if not(prev_stop in stop_leg):
                continue
            else:
                prev_stop2 = stop_leg[prev_stop]
                if not(prev_stop2 == prev_stop):
                    stop_chain.append(prev_stop)
                    trip_chain.append(trip_leg[prev_stop])                
                prev_stop = prev_stop2
        stop_chain = stop_chain[::-1];
        stop_chain.insert(0,orig)
        trip_chain = trip_chain[::-1];
        return stop_chain,trip_chain

    ########### -------------------- FROM RAPTOR FORMAT TO NETWORKx FORMAT ------------------- #################
    ########### -------------------- FROM RAPTOR FORMAT TO NETWORKx FORMAT ------------------- #################
     
    def create_chains(stop1,stop2,PREV_NODES,PREV_TRIPS,max_trans = 4): ### ADDED TO CLASS
        STOP_LIST = [stop2];
        TRIP_LIST = [];
        for i in range(max_trans):
            stop = STOP_LIST[-(i+1)]
            stop2 = PREV_NODES[stop1][-(i+1)][stop]           
            trip = PREV_TRIPS[stop1][-(i+1)][stop]
            STOP_LIST.insert(0,stop2);
            TRIP_LIST.insert(0,trip);        

        return STOP_LIST, TRIP_LIST

    def list_inbetween_stops(feed,STOP_LIST,TRIP_LIST):#,GRAPHS):  ### ADDED TO CLASS


        # feed_dfs = GRAPHS['feed_details'];

        stopList = [];
        edgeList = []; 
        segs = [];
        prev_node = STOP_LIST[0];
        for i,trip in enumerate(TRIP_LIST):
            if not(trip==None):
                stop1 = STOP_LIST[i];
                stop2 = STOP_LIST[i+1];    
                stops = [];

                # stop_times = feed_dfs['stop_times'];
                df = feed.stop_times[feed.stop_times['trip_id']==trip]

                seq1 = list(df[df['stop_id'] == stop1]['stop_sequence'])[0]
                seq2 = list(df[df['stop_id'] == stop2]['stop_sequence'])[0]
                mask1 = df['stop_sequence'] >= seq1;
                mask2 = df['stop_sequence'] <= seq2;
                df = df[mask1 & mask2]
                

                df = df.sort_values(by=['stop_sequence'])
                seg = list(df['stop_id'])
                segs.append(seg)


                
                for j,node in enumerate(seg):
                    if j<(len(seg)-1):
                        stopList.append(node)
                        edgeList.append((prev_node,node,0))
                        prev_node = node
        return segs,stopList,edgeList[1:]


    def gtfs_to_transit_nodesNedges(stopList,NODES): #### ADDED TO CLASS
        nodeList = [];
        edgeList = [None];
        prev_node = None;
        for i,stop in enumerate(stopList):
            try: 
                new_node = WORLD.CONVERTER.convertNode(stop,'gtfs','transit')
                nodeList.append(new_node)
                edgeList.append((prev_node,new_node,0))
            except:
                pass
        edgeList = edgeList[1:]
        return nodeList,edgeList


    def calculateGTFStrips(feed): #,gdf=[]):  ### ADDED TO CLASS

        gtfs_routes = feed.routes
        gtfs_trips = feed.trips
        gtfs_stops = feed.stops
        gtfs_stop_times = feed.stop_times
        gtfs_shapes = feed.shapes

        gtfs_stops = gtfs_stops.set_index('stop_id')


        all_bus_stops = list(feed.stops.stop_id)#li[str(z) for _,z in enumerate(list(feed.stops.index))];
        from_stop_ids = all_bus_stops

        print('preparing to implement raptor for...',len(from_stop_ids),'bus stops')
        #from_stop_id = from_stop_ids[0]
        
        transfer_cost = 5*60;
        recompute_foot_transfers = True;
        start_time = time.time()

        # stops_gdfs = [] #hackkk
        if (recompute_foot_transfers):
            list_stop_ids = list(feed.stops.stop_id)
            rad_miles = 0.1;
            meters_in_miles = 1610
            # stops_gdf = gdf;
            stops_gdf = gtfs_stops;

            FOOT_TRANSFERS = {}
            for k, stop_id in enumerate(list_stop_ids):
                FOOT_TRANSFERS[stop_id] = {};
                stop_pt = stops_gdf.loc[stop_id].geometry

                qual_area = stop_pt.buffer(meters_in_miles *rad_miles)
                mask = stops_gdf.intersects(qual_area)
                for arrive_stop_id, row in stops_gdf[mask].iterrows():
                    if not(arrive_stop_id==stop_id):
                        FOOT_TRANSFERS[stop_id][arrive_stop_id] = transfer_cost
            recompute_foot_transfers = False;


        end_time = time.time()
        print('time to compute foot transfers: ' , end_time - start_time)    
        
        REACHED_BUS_STOPS = {};
        REACHED_NODES = {};

        # for i in range(len(types)):
        #     REACHED_NODES[types[i]] = {};

        #to_stop_id = list(dict2[list(dict2.keys())[0]].keys())[0]
        #from_stop_id = list(ORIG_CONNECTIONS['drive'][list(FRIG_CONNECTIONS['drive'].keys())[0]].keys())[0];
        # to_stop_id = list(DEST_CONNECTIONS['drive'][list(DEST_CONNECTIONS['drive'].keys())[0]].keys())[0];
        # to_stop_id = list(DEST_CONNECTIONS['drive'][list(DEST_CONNECTIONS['drive'].keys())[0]].keys())[0];
        # time_to_stops = {from_stop_id: 0}


        departure_secs = 8.5 * 60 * 60
        # setting transfer limit at 2
        add_footpath_transfers = True; #False; #True;
        stop_skip_num = 1;
        TRANSFER_LIMIT = 5;
        max_wait = 20*60;
        print_yes = False;
        init_start_time2 = time.time();

        params = {'feed':feed,
                  'from_stops': from_stop_ids,
                  'max_wait': max_wait,
                  'add_footpath_transfers':add_footpath_transfers,
                  'FOOT_TRANSFERS': FOOT_TRANSFERS,
                  'gdf': stops_gdf,
                  'foot_transfer_cost': transfer_cost,
                  'stop_skip_num': stop_skip_num,
                  'transfer_limit': TRANSFER_LIMIT,
                  'departure_secs': departure_secs
                 }

        time_to_stops,REACHED_NODES,PREV_NODES,PREV_TRIPS = raptor_shortest_path(params);

        print(); print(); print()
        end_time = time.time();
        print('TOTAL TIME: ',end_time-init_start_time2)
        # for i, from_stop_id in enumerate(from_stop_ids):
        #     for i in range(len(types)):
        #         if (i>=1):
        #             time_to_stops = REACHED_BUS_STOPS[from_stop_id].append(time_to_stops);                
        #             for j, bus_stop in enumerate(list(time_to_stops.keys())):              
        #                 if (bus_stop in BUS_STOP_NODES[types[i]].keys()):
        #                     REACHED_NODES[types[i]].append(BUS_STOP_NODES[types[i]][bus_stop]);
        



        SOLVED = {};
        SOLVED['time_to_stops'] = time_to_stops;
        SOLVED['REACHED_NODES'] = REACHED_NODES;
        SOLVED['PREV_NODES'] = PREV_NODES;
        SOLVED['PREV_TRIPS'] = PREV_TRIPS;
        return SOLVED



    ##### ---------- END OF RAPTOR --------- ##### ---------- END OF RAPTOR --------- #####
    ##### ---------- END OF RAPTOR --------- ##### ---------- END OF RAPTOR --------- #####
    ##### ---------- END OF RAPTOR --------- ##### ---------- END OF RAPTOR --------- #####
    ##### ---------- END OF RAPTOR --------- ##### ---------- END OF RAPTOR --------- #####





########## ====================== PLOTTING =================== ###################
########## ====================== PLOTTING =================== ###################
########## ====================== PLOTTING =================== ###################



def ptsBoundary(corners,nums):
    out = [];
    for i,pt0 in enumerate(corners):
        if i==len(corners)-1:pt1 = corners[0];
        else: pt1 = corners[i+1]
        for k in range(nums[i]):
            alpha = k/nums[i];
            out.append((1-alpha)*pt0+alpha*pt1);
    out = np.array(out)
    return out



def plotCvxHull(ax,points,params = {}):
    """ 
    DESCRIPTION: plots the convex hull of a list of points (in 2D)
    INPUTS: 
    ax: plot handle
    points: list of points
    params: extra parameters 
    -- color: 
    -- linewidth:
    -- alpha: 
    OUTPUTS: 
    none
    """
    # INITIALIZING PARAMETERS: 
    if 'color' in params: color = params['color'];
    else: color = [0,0,0,0.5];
    if 'linewidth' in params: linewidth = params['linewidth'];
    else: linewidth = 1;
    if 'alpha' in params: alpha = params['alpha'];
    else: alpha = 0.05
        
    hull = ConvexHull(points,qhull_options='QJ') # construct convex hull
    #poly = np.zeros([0,2]);
    for i,simplex in enumerate(hull.simplices): # loop through faces stored as indices of points
        ax.plot(points[simplex,0], points[simplex,1],color=color,linewidth=linewidth) # draw a line indicating that face

    poly = points[hull.vertices] # ordering points to plot polygon 
    ax.add_patch(plt.Polygon(poly,color = color,alpha=alpha)) # plotting polygon
    

def plotODs(GRAPHS,SIZES,NODES,scale=1.,figsize=(10,10),ax=None):
    home_sizes = SIZES['home_sizes']
    work_sizes = SIZES['work_sizes']
    home_nodes = NODES['orig']
    work_nodes = NODES['dest']
    
    plot_ODs = True;
    if plot_ODs:
        bgcolor = [0.8,0.8,0.8,1];
        node_colors = [];
        node_sizes = [];
        # scale = 1;
        for k,node in enumerate(GRAPHS['drive'].nodes):
            if False:
                print('asdf')          
            elif (node in home_nodes):        
                node_colors.append([0,0,1,0.5]);
                #node_sizes.append(scale*10);#home_sizes[node]);
                node_sizes.append(scale*home_sizes[node]);            
            elif (node in work_nodes):
                node_colors.append([1,0,0,0.5]);
                #node_sizes.append(scale*10);#work_sizes[node]);
                node_sizes.append(scale*work_sizes[node]);            
            else: 
                node_colors.append([1,1,1]); #[0,0,1])
                node_sizes.append(0)    
    
        edge_colors = [];
        edge_widths = [];
        for k,edge in enumerate(GRAPHS['drive'].edges):
            if False:
                print('asdf')          
            else: 
                edge_colors.append([1,1,1]); #[0,0,1])
                edge_widths.append(2)
    
        # if (not(ax==None)):
        ox.plot_graph(GRAPHS['drive'],ax=ax,bgcolor=bgcolor,  
                            node_color=node_colors,
                            node_size = node_sizes,
                            edge_color=edge_colors,
                            edge_linewidth=edge_widths,
                            figsize=figsize,
                            show=False); #file_format='svg')
        # else:
        #     fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=bgcolor,  
        #                         node_color=node_colors,
        #                         node_size = node_sizes,
        #                         edge_color=edge_colors,
        #                         edge_linewidth=edge_widths,
        #                         figsize=figsize,
        #                         show=False); #file_format='svg')



def plotShapesOnGraph(GRAPHS,shapes,figsize=(10,10),ax=None):
    # if not(ax==None):
    ox.plot_graph(GRAPHS['drive'],ax=ax,bgcolor=[0.8,0.8,0.8,0.8],
                         node_color = [1,1,1,1],node_size = 1,
                         edge_color = [1,1,1,1],
                         edge_linewidth = 2,
                         figsize = figsize,
                         show = False);
    # else:
    #     fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=[0.8,0.8,0.8,0.8],
    #                          node_color = [1,1,1,1],node_size = 1,
    #                          edge_color = [1,1,1,1],
    #                          edge_linewidth = 2,
    #                          figsize = figsize,
    #                          show = False,
    #                          ax=ax);



    for shape in shapes:
        plotCvxHull(ax,shape,params = {});
        # group_polygons = [np.random.rand(10,2) for i in range(5)];




