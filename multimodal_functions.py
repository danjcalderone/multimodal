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


# def nearest_applicable_gtfs_node(mode,gtfs_target,GRAPHS,WORLD,NDF,x,y,radius=0.25/69.):
def nearest_applicable_gtfs_node(mode,GRAPHS,WORLD,NDF,x1,y1,x2,y2):#,rad1=0.25/69.,rad2=0.25/69.):  ### ADDED TO CLASS
    ### ONLY WORKS IF 'gtfs' and 'transit' NODES ARE THE SAME...
    rad_miles = 2.;
    rad1=rad_miles/69.; rad2=rad_miles/69.;

    #(69 mi/ 1deg)*(1 hr/ 3 mi)*(3600 s/1 hr)
    walk_speed = (69./1.)*(1./3.)*(3600./1.)
    GRAPH = GRAPHS['transit'];
    PRECOMPUTE = WORLD['gtfs']['precompute']
    REACHED = PRECOMPUTE['reached']
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


def nearest_nodes(mode,GRAPHS,NODES,x,y): ### ADDED TO CLASS
    if mode == 'gtfs':
        node = ox.distance.nearest_nodes(GRAPHS['transit'], x,y);
        out = convertNode(node,'transit','gtfs',NODES)
        # print(out)
    else:

        out = ox.distance.nearest_nodes(GRAPHS[mode], x,y);
    return out


def subgraph_bnd_box(graph,bot_bnd,top_bnd): ### ADDED TO CLASS
    nodes1 = list(graph);
    nodes2 = [];
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




######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 
######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT ######## CONVERT 



########## ====================== NODE CONVERSION =================== ###################
########## ====================== NODE CONVERSION =================== ###################
########## ====================== NODE CONVERSION =================== ###################


####  ---------------- MAIN FUNCTIONS ------------- ##########
####  ---------------- MAIN FUNCTIONS ------------- ##########

def createEmptyNodesDF(): ### ADDED TO CLASS
    NODES = {};
    NODES['all'] = pd.DataFrame({'drive':[],'walk':[],'transit':[],'ondemand':[],'gtfs':[]},index=[])
    NODES['drive'] = pd.DataFrame({'walk':[],'transit':[],'ondemand':[],'gtfs':[]},index=[])
    NODES['walk'] = pd.DataFrame({'drive':[],'transit':[],'ondemand':[],'gtfs':[]},index=[])
    NODES['ondemand'] = pd.DataFrame({'drive':[],'walk':[],'transit':[],'gtfs':[]},index=[])
    NODES['transit'] = pd.DataFrame({'drive':[],'walk':[],'ondemand':[],'gtfs':[]},index=[])
    NODES['gtfs'] = pd.DataFrame({'drive':[],'walk':[],'transit':[],'ondemand':[]},index=[])
    return NODES

#### ADDING NODES TO DATAFRAME 

def addNodeToDF(node,mode,GRAPHS,NODES): ### ADDED TO CLASS

    if not(node in NODES['all'][mode]):




        self.NETWORKS = {}

        # for mode in self.modes:


        ##### LOAD GRAPH/FEED OBJECTS ######
        self.GRAPHS = {};
        self.FEEDS = {};
        self.RGRAPHS = {}; ## REVERSE GRAPHS
        if 'gtfs' in self.modes:
            feed = gtfs.Feed('data/gtfs/carta_gtfs.zip', time_windows=[0, 6, 10, 12, 16, 19, 24]);
            self.FEEDS['gtfs'] = feed

        for mode in self.modes:
            if verbose: print('loading graph/feed for',mode,'mode...')
            # self.NETWORKS[mode] = NETWORK(mode);
            if mode == 'gtfs':
                self.GRAPHS[mode] = self.graphFromFeed(self.FEEDS['gtfs']);#,start,end) ## NEW
            elif mode == 'ondemand':
                self.GRAPHS[mode] = ox.graph_from_place('chattanooga',network_type='drive'); #ox.graph_from_polygon(graph_boundary,network_type='drive')
            elif mode == 'drive' or mode == 'walk' or mode == 'bike':
                self.GRAPHS[mode] = ox.graph_from_place('chattanooga',network_type='drive'); #ox.graph_from_polygon(graph_boundary,network_type='drive')                

            if verbose: print('constructing NETWORK ',mode,'mode...')
            params2 = {'graph':mode,'GRAPH':self.GRAPHS[mode]}
            self.NETWORKS[mode] = NETWORK(mode,params2);


        if verbose: print('cutting graphs to boundaries...')
        if len(self.bnds) > 0:
            for mode in self.modes:
                self.GRAPHS[mode] = self.trimGraph(self.GRAPHS[mode],self.bnds)

        if verbose: print('composing graphs...')
        self.GRAPHS['all'] = nx.compose_all([self.GRAPHS[mode] for mode in self.modes]);
        if verbose: print('computing reverse graphs...')
        RGRAPHS = {};
        for i,mode in enumerate(self.GRAPHS):
            print('...reversing',mode,'graph...')
            self.RGRAPHS[mode] = self.GRAPHS[mode].reverse();


        params2 = {'modes':self.modes}
        params2 = {'GRAPHS':self.GRAPHS}
        params2 = {'FEEDS':self.FEEDS}
        self.CONVERTER = CONVERTER(params2);


        self.GTFS_DATA = {};




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



    ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA 
    ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA 

    def NOTEBOOKloadBackgroundTrafficData(self):  #### FROM NOTEBOOK
        reload_base = True;
        filename = 'runs/data2524.obj';

        load_base_layer = True
        if reload_base:
            file = open(filename, 'rb')
            DATA = pickle.load(file)
            DATA = pd.read_pickle(filename)
            file.close()

        reread_data = True;
        if reread_data:
            WORLD0 = DATA['WORLD'];
            NDF = DATA['NDF']
            NODES = DATA['NODES']
            BUS_STOP_NODES = DATA['BUS_STOP_NODES']
            load_base_layer = True;


    def add_base_edge_masses(GRAPHS,WORLD,WORLD0):
        modes = []
        for i,mode in enumerate(WORLD):
            if not(mode=='main' or mode=='transit' or mode=='gtfs'):
                WORLD[mode]['base_edge_masses'] = {};
                for e,edge in enumerate(GRAPHS[mode].edges):
                    if edge in WORLD0[mode]['current_edge_masses']:
                        WORLD[mode]['base_edge_masses'][edge] = WORLD0[mode]['current_edge_masses'][edge];


    def NOTEBOOK_INITIALIZE(self):

        poly = np.array([406.35315058,  18.04891652]);
        WORLD['ondemand']['poly'] = poly


        nk = 1; takeall = True;
        # GRADIENT DESCENT...
        for k in range(nk):
            start_time = time.time();
            print('------------------ITERATION',int(WORLD['main']['iter']),'-----------')
            # alpha =1/(k+10.);

            clear_active=True;
            if k == nk-1: clear_active=False;
                
            world_of_gtfs(WORLD,PEOPLE,GRAPHS,NDF,verbose=True,clear_active=clear_active);    
            world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            world_of_ondemand(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=True,show_delivs='all',clear_active=clear_active);
            world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            #world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=True);
            
            print('updating individual choices...')
            update_choices(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=takeall);
            end_time = time.time()
            
            print('iteration time: ',end_time-start_time)
            WORLD['main']['iter'] = WORLD['main']['iter'] + 1.;
            WORLD['main']['alpha'] = 10./(WORLD['main']['iter']+1.);
            # WORLD['main']['alpha'] = 10.;


    def NOTEBOOK_SAVE_PRE(self):

        # #### DON"T CHANGE 
        rewrite_data = True; 
        num_people = len(list(PEOPLE))
        # filename = 'runs/small_data'+str(num_people)+'_blank.obj'
        filename = 'runs/' + group_version + 'b_blank.obj'
        # filename = 'current.obj'
        # if os.path.isfile(filename):
        #     filename
        # import pickle
        # import os.path
        # filename = 'gtfs_trips.obj'
        # if os.path.isfile(filename):

        if rewrite_data:
            fileObj = open(filename, 'wb')
            GRAPHSnoGTFS = {}
            for j,mode in enumerate(GRAPHS):
                if not(mode == 'gtfs'):
                    GRAPHSnoGTFS[mode] = GRAPHS[mode]
                    
            DATA = {
                    'PEOPLE':PEOPLE,
                    'WORLD':WORLD,
                    'DELIVERY':DELIVERY,
                    'NDF':NDF,
                    'NODES':NODES,
                    'LOCS':LOCS,
                    'PRE':PRE,        
                    'BUS_STOP_NODES':BUS_STOP_NODES,
                    # 'GRAPHS':GRAPHSnoGTFS,
                    'SIZES':SIZES
                   }
        #             'GRAPHS':GRAPHS}
            pickle.dump(DATA,fileObj)
            fileObj.close()

    def NOTEBOOK_SAVE_POST(self):

        # #### DON"T CHANGE 
        rewrite_data = True; 
        num_people = len(list(PEOPLE))
        # filename = 'runs/small_data'+str(num_people)+'_full2.obj'
        filename = 'runs/'+group_version+'b.obj'
        # filename = 'current.obj'
        # if os.path.isfile(filename):
        #     filename
        # import pickle
        # import os.path
        # filename = 'gtfs_trips.obj'
        # if os.path.isfile(filename):

        compute_UncongestedEdgeCosts(WORLD,GRAPHS)
        compute_UncongestedTripCosts(WORLD,GRAPHS)
        POST = computeData(WORLD,GRAPHS,DELIVERY)

        if rewrite_data:
            fileObj = open(filename, 'wb')
            GRAPHSnoGTFS = {}
            for j,mode in enumerate(GRAPHS):
                if not(mode == 'gtfs'):
                    GRAPHSnoGTFS[mode] = GRAPHS[mode]
                    
            DATA = {
                    'PEOPLE':PEOPLE,
                    'WORLD':WORLD,
                    'DELIVERY':DELIVERY,
                    'NDF':NDF,
                    'NODES':NODES,
                    'LOCS':LOCS,
                    'PRE':PRE,        
                    'BUS_STOP_NODES':BUS_STOP_NODES,
                    # 'GRAPHS':GRAPHSnoGTFS,
                    'SIZES':SIZES,
                    'POST':POST
                   }
        #             'GRAPHS':GRAPHS}
            pickle.dump(DATA,fileObj)
            fileObj.close()



    def NOTEBOOK_LOAD(self):

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

        reload_gtfs = True;
        filename = 'data/gtfs/gtfs_trips.obj'
        # filename = 'data/gtfs/gtfs_trips_transfers5.obj'
        # filename = 'data/gtfs/gtfs_trips_wait20trans5footyes.obj'
        if reload_gtfs:
            file = open(filename, 'rb')
            data = pickle.load(file)
            file.close()

            REACHED_NODES = data['REACHED_NODES']
            PREV_NODES = data['PREV_NODES']
            PREV_TRIPS = data['PREV_TRIPS']

        assign_gtfs = False;
        WORLD['gtfs']['precompute'] = {};
        WORLD['gtfs']['precompute']['reached'] = REACHED_NODES;
        WORLD['gtfs']['precompute']['prev_nodes'] = PREV_NODES;
        WORLD['gtfs']['precompute']['prev_trips'] = PREV_TRIPS;


    def NOTEBOOK_RUN(self):


        add_base_edge_masses(GRAPHS,WORLD,WORLD0);

        mode = 'ondemand'
        # poly = np.array([-6120.8676711, 306.5130127])
        # poly = np.array([5047.38255623, -288.78570445,    6.31107635]); # 2nd order

        # poly = np.array([696.29355592, 10.31124288])
        # poly = np.array([406.35315058,  18.04891652]);
        # WORLD['ondemand']['poly'] = poly
        # poly = WORLD['ondemand']['fit']['poly']
        # pop_guess = 50.;
        # exp_cost = poly[0] + poly[1]*pop_guess; # + poly[2]*(pop_guess*pop_guess);

        for _,group in enumerate(DELIVERY['groups']):
            poly = DELIVERY['groups'][group]['fit']['poly']
            pop_guess = 25.;
            exp_cost = poly[0] + poly[1]*pop_guess; # + poly[2]*(pop_guess*pop_guess);
            DELIVERY['groups'][group]['expected_cost'] = [exp_cost];
            DELIVERY['groups'][group]['actual_average_cost'] = [0];
            DELIVERY['groups'][group]['current_expected_cost'] = exp_cost;

        WORLD['main']['iter'] = 0.;
        WORLD['main']['alpha'] = 10./(WORLD['main']['iter']+1.);

        WORLD['main']['start_time'] = 0;
        WORLD['main']['end_time'] = 3600*4.;    

        nk = 5; 

        # nk = 2
        # GRADIENT DESCENT...
        for k in range(nk):
            start_time = time.time();
            print('------------------ITERATION',int(WORLD['main']['iter']),'-----------')
            # alpha =1/(k+10.);

            clear_active=True;
            if k == nk-1: clear_active=False;
                
            world_of_gtfs(WORLD,PEOPLE,GRAPHS,NDF,verbose=True,clear_active=clear_active);    
            world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            world_of_ondemand(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=True,show_delivs='all',clear_active=clear_active);
            world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=True,clear_active=clear_active); #graph,costs,sources, targets):    
            #world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=True);
            

            print('updating individual choices...')
            update_choices(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=False);
            end_time = time.time()
            
            print('iteration time: ',end_time-start_time)
            WORLD['main']['iter'] = WORLD['main']['iter'] + 1.;
            WORLD['main']['alpha'] = 10./(WORLD['main']['iter']+1.);
            # WORLD['main']['alpha'] = 10.;






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


    # def nearest_applicable_gtfs_node(mode,gtfs_target,GRAPHS,WORLD,NDF,x,y,radius=0.25/69.):
    def nearest_applicable_gtfs_node(mode,GRAPHS,WORLD,NDF,x1,y1,x2,y2):#,rad1=0.25/69.,rad2=0.25/69.):  ### ADDED TO CLASS
        ### ONLY WORKS IF 'gtfs' and 'transit' NODES ARE THE SAME...
        rad_miles = 2.;
        rad1=rad_miles/69.; rad2=rad_miles/69.;

        #(69 mi/ 1deg)*(1 hr/ 3 mi)*(3600 s/1 hr)
        walk_speed = (69./1.)*(1./3.)*(3600./1.)
        GRAPH = GRAPHS['transit'];
        PRECOMPUTE = WORLD['gtfs']['precompute']
        REACHED = PRECOMPUTE['reached']
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


    def nearest_nodes(mode,GRAPHS,NODES,x,y): ### ADDED TO CLASS
        if mode == 'gtfs':
            node = ox.distance.nearest_nodes(GRAPHS['transit'], x,y);
            out = convertNode(node,'transit','gtfs',NODES)
            # print(out)
        else:

            out = ox.distance.nearest_nodes(GRAPHS[mode], x,y);
        return out




           



    def generatePolygons(self):

        # group_polygons = generate_polygons('reg2',center_point);
        # group_version = 'huge1';
        # group_version = 'large1';
        # group_version = 'medium1';
        # group_version = 'small1';
        group_version = 'regions11';
        path2 = './DAN/group_sections/'+group_version+'.geojson'; #'/map.geojson'
        group_polygons = generate_polygons('from_geojson',path = path2);

        plotShapesOnGraph(GRAPHS,group_polygons,figsize=(10,10));    


    # WORLD['ondemand']['people'] = people_tags.copy();
    #  #people_tags.copy();
    # # WORLD['ondemand+transit']['people'] = people_tags.copy();
    # WORLD['ondemand']['trips'] = {};








class ONDEMAND:

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



    def __init__(self,params):
    # def generateDeliveries(GRAPHS,WORLD,NDF,params,DELIVERYDF={}):   ##### ADDED TO CLASS 
        if 'direct_locs' in params: direct_locs = params['direct_locs'];
        else: direct_locs = [];
        if 'shuttle_locs' in params: shuttle_locs = params['shuttle_locs'];
        else: shuttle_locs = [];



        if 'driver_info' in params: driver_info = params['driver_info'];

        if 'center_point' in params: center_point = params['center_point'];
        else: center_point: center_point = (-85.3094,35.0458)
     
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


        ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS 
        ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS 
        ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS 

        # GROUPS = 

        DELIVERY['groups'] = {};


        if len(DELIVERYDF)>0:
            grpsDF = DELIVERYDF['grps']
            driversDF = DELIVERYDF['drivers']
            DELIVERY['grpsDF'] = grpsDF
            DELIVERY['driversDF'] = driversDF
            DELIVERY['grpsDF']['num_drivers'] = list(np.zeros(len(grpsDF)))


            num_groups = len(grpsDF);
            for k in range(num_groups):
                ROW = grpsDF.iloc[k];
                group = ROW['group']
                DELIVERY['groups'][group] = {};

                DELIVERY['groups'][group]['driver_runs'] = [];
                DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);

                if 'date' in ROW: DELIVERY['groups'][group]['date'] = ROW['date'];
                else: DELIVERY['groups'][group]['date'] = '2023-07-31'

                num_drivers = 8; 
                loc = ROW['depot_loc']
                DELIVERY['groups'][group]['depot'] = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
                DELIVERY['groups'][group]['depot_node'] = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

                DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);
                DELIVERY['groups'][group]['expected_cost'] = [0.];
                DELIVERY['groups'][group]['current_expected_cost'] = 0.;
                DELIVERY['groups'][group]['actual_average_cost'] = [];

                DRVDF = driversDF[group]
                if len(DRVDF) > 0:
                    num_drivers = len(DRVDF);
                    for i in range(num_drivers):
                        ROW2 = DRVDF.iloc[i];
                        driver_start_time = ROW2['start_time'];
                        driver_end_time = ROW2['end_time']; 
                        am_capacity = ROW2['am_capacity'];
                        wc_capacity = ROW2['wc_capacity'];

                        DELIVERY['groups'][group]['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
                            'am_capacity': am_capacity,'wc_capacity': wc_capacity})
                else: 
                    for i in range(num_drivers):
                        driver_start_time = WORLD['main']['start_time'];
                        driver_end_time = WORLD['main']['end_time'];
                        am_capacity = 8
                        wc_capacity = 2
                        DELIVERY['groups'][group]['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
                            'am_capacity': am_capacity,'wc_capacity': wc_capacity})

                DELIVERY['grpsDF']['num_drivers'].iloc[k] = num_drivers;

        else: 
            if 'num_groups' in params: num_groups = params['num_groups'];
            else: num_groups = 2;

            for i  in range(num_groups):
                group = 'group'+str(i);
                DELIVERY['groups'][group] = {};
                num_drivers = 8; 
                driver_start_time = WORLD['main']['start_time'];
                driver_end_time = WORLD['main']['end_time'];
                am_capacity = 8
                wc_capacity = 2
                DELIVERY['groups'][group]['driver_runs'] = [];
                DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);

                DELIVERY['groups'][group]['date'] = '2023-07-31'
                loc = center_point
                DELIVERY['groups'][group]['depot'] = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
                DELIVERY['groups'][group]['depot_node'] = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

                DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);
                DELIVERY['groups'][group]['expected_cost'] = [0.];
                DELIVERY['groups'][group]['current_expected_cost'] = 0.;
                DELIVERY['groups'][group]['actual_average_cost'] = [];

                for j in range(num_drivers):
                    DELIVERY['groups'][group]['driver_runs'].append({'run_id': j,'start_time': driver_start_time,'end_time': driver_end_time,
                        'am_capacity': am_capacity,'wc_capacity': wc_capacity})


        if len(DELIVERYDF)>0:
            DELIVERY['grpsDF']['num_possible_trips'] = list(np.zeros(len(DELIVERY['grpsDF'])))




        ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 
        ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 

        num_drivers = 8; 
        driver_start_time = WORLD['main']['start_time'];
        driver_end_time = WORLD['main']['end_time'];
        am_capacity = 8
        wc_capacity = 2
        DELIVERY['driver_runs'] = [];
        DELIVERY['time_matrix'] = np.zeros([1,1]);

        DELIVERY['date'] = '2023-07-31'
        loc = center_point
        DELIVERY['depot'] = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
        DELIVERY['depot_node'] = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

        for i in range(num_drivers):
            DELIVERY['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
                'am_capacity': am_capacity,'wc_capacity': wc_capacity})





        ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 
        ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 
        ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 

        # nopts = len(opts);
        # people_tags = [];
        delivery1_tags = [];
        delivery2_tags = [];

        delivery_nodes = [];
        for i,loc in enumerate(direct_locs):
            tag = 'delivery1_' + str(i);
            delivery1_tags.append(tag);
            DELIVERY['direct'][tag] = {};
            DELIVERY['direct'][tag]['active_trips'] = [];
            DELIVERY['direct'][tag]['active_trip_history'] = [];    
            DELIVERY['direct'][tag]['loc'] = loc; #DELIVERY1_LOC[i]
            DELIVERY['direct'][tag]['current_path'] = []    

            DELIVERY['direct'][tag]['nodes'] = {};
            node = ox.distance.nearest_nodes(GRAPHS['ondemand'], loc[0], loc[1]);
            DELIVERY['direct'][tag]['nodes']['source'] = node
            direct_node_lists['source'].append(node)


            # node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
            # node = BUS_STOP_NODES['ondemand'][node0];
            node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
            node = int(convertNode(node0,'transit','ondemand',NDF))

            DELIVERY['direct'][tag]['nodes']['transit'] = node
            DELIVERY['direct'][tag]['nodes']['transit2'] = node0
            direct_node_lists['transit'].append(node)    

            DELIVERY['direct'][tag]['people'] = [];
            DELIVERY['direct'][tag]['sources'] = [];
            DELIVERY['direct'][tag]['targets'] = [];


        delivery2_nodes = [];
        for i,loc in enumerate(shuttle_locs):
            tag = 'delivery2_' + str(i);
            delivery2_tags.append(tag);
            DELIVERY['shuttle'][tag] = {};
            DELIVERY['shuttle'][tag]['active_trips'] = [];    
            DELIVERY['shuttle'][tag]['active_trip_history'] = [];    
            DELIVERY['shuttle'][tag]['loc'] = loc;
            DELIVERY['shuttle'][tag]['current_path'] = []

            DELIVERY['shuttle'][tag]['nodes'] = {};
            node = ox.distance.nearest_nodes(GRAPHS['ondemand'], loc[0], loc[1]);
            DELIVERY['shuttle'][tag]['nodes']['source'] = node
            shuttle_node_lists['source'].append(node)    

            # node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
            # node = BUS_STOP_NODES['ondemand'][node0];
            node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
            node = int(convertNode(node0,'transit','ondemand',NDF))



            
            DELIVERY['shuttle'][tag]['nodes']['transit'] = node
            DELIVERY['shuttle'][tag]['nodes']['transit2'] = node0
            shuttle_node_lists['transit'].append(node)    

            DELIVERY['shuttle'][tag]['people'] = [];
            DELIVERY['shuttle'][tag]['sources'] = [];
            DELIVERY['shuttle'][tag]['targets'] = [];
            
        return DELIVERY



    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
    ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 






class GROUP: 
    def __init_(self,params):
        self.booking_ids = [];
        self.time_matrix = np.zeros([1,1]);
        self.expected_cost = [0.];
        self.current_expected_cost = 0.;
        self.actual_average_cost = [];





    def createGroupsDF(self):
    # def createGroupsDF(polygons,types0 = []):  ### ADDED TO CLASS
        ngrps = len(polygons);
        groups = ['group'+str(i) for i in range(len(polygons))];
        depotlocs = [];
        for polygon in polygons:
            depotlocs.append((1./np.shape(polygon)[0])*np.sum(polygon,0));    
        if len(types0)==0: types = [['direct','shuttle'] for i in range(ngrps)]
        else: types = types0;
        GROUPS = pd.DataFrame({'group':groups,'depot_loc':depotlocs,'type':types,'polygon':polygons}); #,index=list(range(ngrps))
        # new_nodes = pd.DataFrame(node_tags,index=[index_node])
        # NODES[mode] = pd.concat([NODES[mode],new_nodes]);
        return GROUPS

    def createDriversDF(self): 
    # def createDriversDF(params,WORLD):  ### ADDED TO CLASS
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
        



    def NOTEBOOK(self):
        num_counts = 5; num_per_count = 1;
        pts = generateOndemandCongestion(WORLD,PEOPLE,DELIVERY,GRAPHS,num_counts = num_counts,num_per_count=num_per_count)
        for _,group in enumerate(DELIVERY['groups']):
            counts = DELIVERY['groups'][group]['fit']['counts']
            pts = DELIVERY['groups'][group]['fit']['pts']
            shp = np.shape(pts);
            xx = np.reshape(counts,shp[0]*shp[1])
            coeffs = DELIVERY['groups'][group]['fit']['poly']
            Yh = DELIVERY['groups'][group]['fit']['Yh'];
            for j in range(np.shape(pts)[1]):
                plt.plot(counts,pts[:,j],'o',color='blue')        
            plt.plot(xx,Yh,'--',color='red')


    def generateCongestionModel(self):
    # def generateOndemandCongestion(WORLD,PEOPLE,DELIVERIES,GRAPHS,num_counts=3,num_per_count=1,verbose=True,show_delivs='all',clear_active=True):
        if 'groups' in DELIVERIES:
            mode = 'ondemand'
            all_possible_trips = list(WORLD[mode]['trips']);
            if verbose: print('starting on-demand curve computations for a total of',len(all_possible_trips),'trips...')

            trips_by_group = {group:[] for _,group in enumerate(DELIVERIES['groups'])}
            for _,trip in enumerate(all_possible_trips):
                group = WORLD[mode]['trips'][trip]['group'];
                trips_by_group[group].append(trip)

            for k, group in enumerate(DELIVERIES['groups']):

                possible_trips = trips_by_group[group]

                if len(possible_trips)>0:
                    # counts = [20,30,40,50,60,70,80];
                    num_pts = num_counts; num_per_count = num_per_count; 

                    counts = np.linspace(1,len(possible_trips)-1,num_pts)
                    counts = [int(count) for _,count in enumerate(counts)];
                    
                    if verbose: print('starting on-demand curve computation for',group) #'with',len(possible_trips),'...')
                    
                    GRAPH = GRAPHS['ondemand'];
                    ONDEMAND = WORLD['ondemand'];
                    DELIVERY = DELIVERIES['groups'][group]

                    print('...for a total number of trips of',len(possible_trips))
                    print('counts to compute: ',counts)

                    pts = np.zeros([len(counts),num_per_count]);


                    ptsx = {};
                    for i,count in enumerate(counts):
                        print('...computing averages for',count,'active ondemand trips in',group,'...')
                        for j in range(num_per_count):

                            try: 
                                trips_to_plan = sample(possible_trips,count)
                                nodes_to_update = [DELIVERY['depot_node']];
                                nodeids_to_update = [0];
                                for _,trip in enumerate(possible_trips):
                                    TRIP = WORLD[mode]['trips'][trip]
                                    nodes_to_update.append(trip[0]);
                                    nodes_to_update.append(trip[1]);
                                    nodeids_to_update.append(TRIP['pickup_node_id'])
                                    nodeids_to_update.append(TRIP['dropoff_node_id'])

                                updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                                payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

                                manifests = optimizer.offline_solver(payload) 
                                PDF = payloadDF(payload,GRAPHS)
                                MDF = manifestDF(manifests,PDF)
                                average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,0)
                                pts[i,j] = average_time
                            except:
                                pass


                            # ptsx[(i,j)] = times_to_average

                    DELIVERY['fit'] = {}
                    DELIVERY['fit']['counts'] = np.outer(counts,np.ones(num_per_count))
                    DELIVERY['fit']['pts'] = pts.copy();

                    shp = np.shape(pts);
                    xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
                    yy = np.reshape(pts,shp[0]*shp[1])
                    order = 1;
                    coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
                    Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
                    DELIVERY['fit']['poly'] = coeffs[::-1]
                    DELIVERY['fit']['Yh'] = Yh.copy();

                else:
                    DELIVERY = DELIVERIES['groups'][group]
                    counts = np.linspace(1,100,num_counts)
                    DELIVERY['fit'] = {}
                    DELIVERY['fit']['counts'] = np.outer(counts,np.ones(num_per_count))
                    pts = 1000.*np.ones([num_counts,num_per_count]);
                    DELIVERY['fit']['pts'] = pts; #1000.*np.ones([num_counts,num_per_count]);

                    shp = np.shape(pts);
                    xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
                    yy = np.reshape(pts,shp[0]*shp[1])
                    order = 1;
                    coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
                    Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
                    DELIVERY['fit']['poly'] = coeffs[::-1]
                    DELIVERY['fit']['Yh'] = Yh.copy();
            out = None


        else: 
            DELIVERY = DELIVERIES;
            if verbose: print('starting on-demand curve computation...')    
            mode = 'ondemand'
            GRAPH = GRAPHS['ondemand'];
            ONDEMAND = WORLD['ondemand'];
            possible_trips = list(WORLD[mode]['trips']) ;

            print('...for a total number of trips of',len(possible_trips))
            print('counts to compute: ',counts)

            pts = np.zeros([len(counts),num_per_count]);

            ptsx = {};
            for i,count in enumerate(counts):
                print('...computing averages for',count,'active ondemand trips...')
                for j in range(num_per_count):
                    trips_to_plan = sample(possible_trips,count)

                    nodes_to_update = [DELIVERY['depot_node']];
                    nodeids_to_update = [0];
                    for _,trip in enumerate(trips_to_plan):
                        TRIP = WORLD[mode]['trips'][trip]
                        nodes_to_update.append(trip[0]);
                        nodes_to_update.append(trip[1]);
                        nodeids_to_update.append(TRIP['pickup_node_id'])
                        nodeids_to_update.append(TRIP['dropoff_node_id'])

                    updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                    payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

                    manifests = optimizer.offline_solver(payload) 
                    PDF = payloadDF(payload,GRAPHS)
                    MDF = manifestDF(manifests,PDF)
                    average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,0)

                    pts[i,j] = average_time
                    # ptsx[(i,j)] = times_to_average
            out = pts

    #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS 
    #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS 
    #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS 




    def travel_time_matrix(self): #odes,GRAPH):   ### ADDED TO CLASS
    # def travel_time_matrix(nodes,GRAPH):   ### ADDED TO CLASS
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
        return MAT;



    def updateTravelTimeMatrix(self): #nodes,ids,GRAPH,DELIVERY,WORLD,group=[0]):  ### ADDED TO CLASS
    # def updateTravelTimeMatrix(nodes,ids,GRAPH,DELIVERY,WORLD,group=[0]):  ### ADDED TO CLASS
        MAT = travel_time_matrix(nodes,GRAPH)
        for i,id1 in enumerate(ids):
            for j,id2 in enumerate(ids):
                if len(group)>0:
                    DELIVERY['time_matrix'][id1][id2] = MAT[i,j];
                else:
                    WORLD['ondemand']['time_matrix'][id1][id2] = MAT[i,j];


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



    def constructPayload(self): #active_trips,DELIVERY,WORLD,GRAPHS,group=[0]): #PAY,ids_to_keep):   ### ADDED TO CLASS
    # def constructPayload(active_trips,DELIVERY,WORLD,GRAPHS,group=[0]): #PAY,ids_to_keep):   ### ADDED TO CLASS
        payload = {};
        payload['date'] = DELIVERY['date']
        payload['depot'] = DELIVERY['depot']
        payload['driver_runs'] = DELIVERY['driver_runs']

        if len(group)>0:
            MAT = DELIVERY['time_matrix'];
        else:
            MAT = WORLD['ondemand']['time_matrix'];

        GRAPH = GRAPHS['ondemand']
        
        nodes = {};
        inds = [0];
        curr_node_id = 1;

        mode = 'ondemand'
        requests = [];
        for i,trip in enumerate(active_trips):
            requests.append({})
            node1 = trip[0];
            node2 = trip[1];
            TRIP = WORLD[mode]['trips'][trip];

            requests[i]['booking_id'] = TRIP['booking_id']
            requests[i]['pickup_node_id'] = curr_node_id; curr_node_id = curr_node_id + 1; #TRIP['pickup_node_id']
            requests[i]['dropoff_node_id'] = curr_node_id; curr_node_id = curr_node_id + 1; #TRIP['dropoff_node_id']

            requests[i]['am'] = int(TRIP['am'])
            requests[i]['wc'] = int(TRIP['wc'])
            requests[i]['pickup_time_window_start'] = TRIP['pickup_time_window_start']
            requests[i]['pickup_time_window_end'] = TRIP['pickup_time_window_end']
            requests[i]['dropoff_time_window_start'] = TRIP['dropoff_time_window_start']
            requests[i]['dropoff_time_window_end'] = TRIP['dropoff_time_window_end']

            requests[i]['pickup_pt'] = {'lat': float(GRAPH.nodes[node1]['y']), 'lon': float(GRAPH.nodes[node1]['x'])}
            requests[i]['dropoff_pt'] = {'lat': float(GRAPH.nodes[node2]['y']), 'lon': float(GRAPH.nodes[node2]['x'])}

            inds.append(requests[i]['pickup_node_id'])
            inds.append(requests[i]['dropoff_node_id'])

            nodes[node1] = {'main':TRIP['pickup_node_id'],'curr':requests[i]['pickup_node_id']}
            nodes[node2] = {'main':TRIP['dropoff_node_id'],'curr':requests[i]['dropoff_node_id']}

            # 'pickup_pt': {'lat': 35.0296296, 'lon': -85.2301767},
            # 'dropoff_pt': {'lat': 35.0734152, 'lon': -85.1315328},
            # 'booking_id': 39851211,
            # 'pickup_node_id': 1,
            # 'dropoff_node_id': 2},

        inds = np.array(inds);
        MAT = np.array(MAT)
        MAT = MAT[np.ix_(inds,inds)]
        MAT2 = []
        for i,row in enumerate(MAT):
            MAT2.append(list(row.astype('int')))
        payload['time_matrix'] = MAT2
        payload['requests'] = requests
        return payload,nodes;


    def payloadDF(self): #PAY,GRAPHS,include_drive_nodes = False):  ### ADDED TO CLASS
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


    def manifestDF(self):
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

    def assignCostsFromManifest(self): #active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS
    # def assignCostsFromManifest(active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS
        mode = 'ondemand';

        times_to_average = [0.];

        actual_costs = {};
        runs_info = {};
        run_ids = {};
        for i,trip in enumerate(active_trips):
            node1 = trip[0]
            node2 = trip[1]
            booking_id = WORLD[mode]['trips'][trip]['booking_id']
            main_node_id1 = nodes[trip[0]]['main']
            main_node_id2 = nodes[trip[1]]['main']
            curr_node_id1 = nodes[trip[0]]['curr']
            curr_node_id2 = nodes[trip[1]]['curr']


            try: 

                # mask1 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
                # mask2 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');
                mask1 = (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
                mask2 = (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');

                time1 = MDF[mask1]['scheduled_time'][0];
                time2 = MDF[mask2]['scheduled_time'][0];

                run_id = int(MDF[mask1]['run_id'][0]);


                time_diff = np.abs(time2-time1)

                actual_costs[trip] = time_diff;
                run_ids[trip] = run_id;

                runs_info[trip] = {}
                runs_info[trip]['runid'] = run_id;
                
                MDF2 = MDF[MDF['run_id'] == run_id]
                ind1 = np.where(MDF2['node_id'] == curr_node_id1)[0][0];
                ind2 = np.where(MDF2['node_id'] == curr_node_id2)[0][0];
                MDF2 = MDF2.iloc[ind1:ind2+1]
                runs_info[trip]['num_passengers'] = MDF2['booking_id'].nunique();

                if time_diff < 7200:
                    times_to_average.append(time_diff);

            except: 
                actual_costs[trip] = 1000000000.;
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
        for i,trip in enumerate(WORLD[mode]['trips']):#active_trips):
            #if not(trip in active_trips):

            WORLD[mode]['trips'][trip]['costs']['current_time'] = est_ave_time;
            if trip in active_trips:
                WORLD[mode]['trips'][trip]['costs']['time'].append(actual_costs[trip]);#est_ave_time);
                WORLD[mode]['trips'][trip]['costs']['current_time'] = actual_costs[trip];#est_ave_time);
                try:
                    WORLD[mode]['trips'][trip]['run_id'] = run_ids[trip] 
                    WORLD[mode]['trips'][trip]['num_passengers'] = runs_info[trip]['num_passengers'];
                except:
                    pass
            else:
                if len(WORLD[mode]['trips'][trip]['costs']['time'])==0: prev_time = est_ave_time;
                else: prev_time = WORLD[mode]['trips'][trip]['costs']['time'][-1]
                WORLD[mode]['trips'][trip]['costs']['time'].append(prev_time);
                WORLD[mode]['trips'][trip]['costs']['current_time'] = prev_time;


            factors = ['time','money','conven','switches']
            for j,factor in enumerate(factors):#WORLD[mode]['trips'][trip]['costs']):
                if not(factor == 'time'): 
                    # WORLD[mode]['trips']['costs'][factor] = 0. 
                    WORLD[mode]['trips'][trip]['costs'][factor].append(0.);
                    WORLD[mode]['trips'][trip]['costs']['current_'+factor] = 0. 

        return average_time,times_to_average




class TRIP:
    def makeTrip_OLD(modes,nodes,node_types,NODES,deliveries = []): 
        trip = {};
        segs = [];
        deliv_counter = 0;
        for i,mode in enumerate(modes):
            segs.append({});
            segs[-1]['mode'] = mode;
            segs[-1]['start_nodes'] = []; #nodes[i]
            segs[-1]['end_nodes'] = []; #nodes[i+1]
            segs[-1]['path'] = [];
            for j,node in enumerate(nodes[i]):
                node2 = convertNode(node,node_types[i],mode,NODES)
                segs[-1]['start_nodes'].append(node2)
            for j,node in enumerate(nodes[i+1]):
                node2 = convertNode(node,node_types[i+1],mode,NODES)
                segs[-1]['end_nodes'].append(convertNode(node,node_types[i+1],mode,NODES))
                    
    #         segs[-1]['start_types'] = node_types[i]
    #         segs[-1]['end_types'] = node_types[i+1]
            if mode == 'ondemand':
                segs[-1]['delivery'] = deliveries[deliv_counter]
                deliv_counter = deliv_counter + 1;
                
        trip = {}
        trip['current'] = {};
        trip['structure'] = segs; 
        trip['current']['cost'] = 100000000;
        trip['current']['traj'] = []; 
        return trip


    def makeTrip(modes,nodes,NODES,delivery_grps=[],deliveries=[]): #,node_types,NODES,deliveries = []):
        
        trip = {};
        segs = [];

        deliv_counter = 0;
        for i,mode in enumerate(modes):
            segs.append({});
            segs[-1]['mode'] = mode;
            segs[-1]['start_nodes'] = []; #nodes[i]
            segs[-1]['end_nodes'] = []; #nodes[i+1]
            segs[-1]['path'] = [];
            
            if len(delivery_grps)>0:
                if mode == 'ondemand': segs[-1]['delivery_group'] = delivery_grps[i];
                else: segs[-1]['delivery_group'] = None;

            for j,node in enumerate(nodes[i]['nodes']):
                node2 = convertNode(node,nodes[i]['type'],mode,NODES)

                segs[-1]['start_nodes'].append(node2)
            for j,node in enumerate(nodes[i+1]['nodes']):
                node2 = convertNode(node,nodes[i+1]['type'],mode,NODES)

                segs[-1]['end_nodes'].append(node2);
                    
    #         segs[-1]['start_types'] = node_types[i]
    #         segs[-1]['end_types'] = node_types[i+1]

            if mode == 'ondemand':
                segs[-1]['delivery'] = deliveries[deliv_counter]
                deliv_counter = deliv_counter + 1;
                
        trip = {}
        trip['current'] = {};
        trip['structure'] = segs; 
        trip['current']['cost'] = 100000000;
        trip['current']['traj'] = [];
        trip['active'] = False;
        return trip



    def queryTrip(TRIP,PERSON,NODES,GRAPHS,DELIVERY,WORLD):
        trip_cost = 0;
        costs_to_go = [];
        # COSTS_to_go = [];
        next_inds = [];
        next_nodes = [];

        for k,SEG in enumerate(TRIP['structure'][::-1]):
            end_nodes = SEG['end_nodes'];
            start_nodes = SEG['start_nodes'];
            group = SEG['delivery_group']
    #         print(SEG['end_types'])
    #         end_types = SEG['end_types'];
    #         start_types = SEG['start_types'];
            mode = SEG['mode'];
            NETWORK = WORLD[mode];
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
                    seg_details = (start_node,end_node,)
                    cost,path,_ = querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group=group)
                    possible_leg_costs[i] = cost;
                    # possible_leg_COSTS[i] = COSTS.copy();
                ind = np.argmax(possible_leg_costs)
                next_inds[-1].append(ind)
                next_nodes[-1].append(end_nodes[ind])
                costs_to_go[-1].append(possible_leg_costs[ind]);
                # COSTS_to_go[-1].append(possible_leg_COSTS[ind]);
            

        
        
        init_ind = np.argmin(costs_to_go[-1]);
        init_cost = costs_to_go[-1][init_ind]
        init_node = TRIP['structure'][0]['start_nodes'][init_ind];

        costs_to_go = costs_to_go[::-1];
        next_inds = next_inds[::-1]
        next_nodes = next_nodes[::-1]
        

        inds = [init_ind];
        nodes = [init_node];
        
        prev_mode = TRIP['structure'][0]['mode'];

        for k,SEG in enumerate(TRIP['structure']):
            mode = SEG['mode']
    #         SEG['opt_start'] = 
    #         SEG['opt_end'] = next_nodes[k][inds[-1]]
            next_ind = next_inds[k][inds[-1]];
            next_node = next_nodes[k][inds[-1]];
            
            SEG['opt_start'] = convertNode(nodes[-1],prev_mode,mode,NODES);
            SEG['opt_end'] = next_node;    
            inds.append(next_ind)
            nodes.append(next_node);
            prev_mode = SEG['mode']
        
        trip_cost = costs_to_go[init_ind][0];
        TRIP['current']['cost'] = trip_cost

        TRIP['current']['traj'] = nodes;



class SEG:

    def addDelivSeg(seg_details,mode,GRAPHS,DELIVERIES,WORLD,group='group0',mass=1,track=False,PERSON={}):

        source = seg_details[0]
        target = seg_details[1]

        start1 = WORLD['main']['start_time']; #seg_details[2]
        end1 = WORLD['main']['end_time']; #seg_details[4]
        

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

        if not(trip in WORLD[mode]['trips'].keys()):

            # print('adding delivery segment...')

            WORLD[mode]['trips'][trip] = {};
            WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
            #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
            WORLD[mode]['trips'][trip]['path'] = [];
            WORLD[mode]['trips'][trip]['delivery_history'] = [];
            WORLD[mode]['trips'][trip]['mass'] = 0;
            WORLD[mode]['trips'][trip]['delivery'] = None; #delivery;

            ##### NEW
            WORLD[mode]['trips'][trip]['typ'] = None;  #shuttle or direct

            if len(PERSON)>0:
                if 'group_options' in PERSON: WORLD[mode]['trips'][trip]['group_options'] = PERSON['group_options'];

            WORLD[mode]['trips'][trip]['costs']['current_time'] = distance;
            WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
            WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
            WORLD[mode]['trips'][trip]['costs']['current_switches'] = 1; 
            WORLD[mode]['trips'][trip]['current_path'] = None; #path;        

            # WORLD[mode]['trips'][trip][''] asdfasdf

            

            WORLD[mode]['trips'][trip]['am'] = int(1);
            WORLD[mode]['trips'][trip]['wc'] = int(0);


            WORLD[mode]['trips'][trip]['group'] = group;

            booking_id = int(len(WORLD[mode]['booking_ids']));
            WORLD[mode]['trips'][trip]['booking_id'] = booking_id;
            WORLD[mode]['booking_ids'].append(booking_id);


            for _,group in enumerate(DELIVERIES['groups']):
                DELIVERY = DELIVERIES['groups'][group];
                time_matrix = DELIVERY['time_matrix'];
                shp = np.shape(time_matrix)
                time_matrix = np.block([[time_matrix,np.zeros([shp[0],2])],[np.zeros([2,shp[1]]),np.zeros([2,2]) ]]);
                DELIVERY['time_matrix'] = time_matrix;

                WORLD[mode]['trips'][trip]['pickup_node_id'] = shp[0];
                WORLD[mode]['trips'][trip]['dropoff_node_id'] = shp[0]+1;


            try: 
                ##### NEEDS TO BE UPDATED 
                dist = nx.shortest_path_length(GRAPHS['drive'], source=trip[0], target=trip[1], weight='c');
                maxtriptime = dist*4;
                pickup_start = np.random.uniform(low=start1, high=end1-maxtriptime, size=1)[0];
                pickup_end = pickup_start + 60*20;
                dropoff_start = pickup_start;
                dropoff_end = pickup_end + maxtriptime;

                WORLD[mode]['trips'][trip]['pickup_time_window_start'] = int(pickup_start);
                WORLD[mode]['trips'][trip]['pickup_time_window_end'] = int(pickup_end);
                WORLD[mode]['trips'][trip]['dropoff_time_window_start'] = int(dropoff_start);
                WORLD[mode]['trips'][trip]['dropoff_time_window_end'] = int(dropoff_end);

            except:
                print('balked on adding deliv seg...')
                WORLD[mode]['trips'][trip]['pickup_time_window_start'] = int(start1);
                WORLD[mode]['trips'][trip]['pickup_time_window_end'] = int(end1);
                WORLD[mode]['trips'][trip]['dropoff_time_window_start'] = int(start1);
                WORLD[mode]['trips'][trip]['dropoff_time_window_end'] = int(end1);






            
            # WORLD[mode]['active_trips'].append(trip)
        # print(np.shape(WORLD[mode]['time_matrix']))



        
        # if track==True:
        #     WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
        #     WORLD[mode]['trips'][trip]['costs']['money'].append(1)
        #     WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
        #     WORLD[mode]['trips'][trip]['costs']['switches'].append(1)
        #     WORLD[mode]['trips'][trip]['path'].append(path);

        


        # if mass > 0:
        #     for j,node in enumerate(path):
        #         if j < len(path)-1:
        #             edge = (path[j],path[j+1],0)
        #             edge_mass = WORLD[mode]['edge_masses'][edge][-1] + mass;
        #             # edge_cost = 1.*edge_mass + 1.;
        #             WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
        #             # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
        #             # WORLD[mode]['current_edge_costs'][edge] = edge_cost;
        #             WORLD[mode]['current_edge_masses'][edge] = edge_mass;    


    def planDijkstraSeg(seg_details,mode,GRAPHS,WORLD,mass=1,track=False):
        
        
        source = seg_details[0];
        target = seg_details[1];

        trip = (source,target);
        GRAPH = GRAPHS[mode];
        # if mode == 'transit' and mass > 0:
        #     print(mode)
        try:
            temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c'); #Dumbfunctions
            distance = temp[0];
            # if mode == 'walk':
            #     distance = distance/1000;
            path = temp[1]; 
        except: 
            #print('no path found for bus trip ',trip,'...')
            distance = 1000000000000;
            path = [];

        if not(trip in WORLD[mode]['trips'].keys()):
            WORLD[mode]['trips'][trip] = {};
            WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
            #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
            WORLD[mode]['trips'][trip]['path'] = [];
            WORLD[mode]['trips'][trip]['mass'] = 0;

        WORLD[mode]['trips'][trip]['costs']['current_time'] = distance
        WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
        WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
        WORLD[mode]['trips'][trip]['costs']['current_switches'] = 1; 
        WORLD[mode]['trips'][trip]['current_path'] = path;
        WORLD[mode]['trips'][trip]['mass'] = 0;
        
        if track==True:
            WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
            WORLD[mode]['trips'][trip]['costs']['money'].append(1)
            WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
            WORLD[mode]['trips'][trip]['costs']['switches'].append(1)
            WORLD[mode]['trips'][trip]['path'].append(path);

        if mass > 0:
            for j,node in enumerate(path):
                if j < len(path)-1:
                    edge = (path[j],path[j+1],0)
                    edge_mass = WORLD[mode]['edge_masses'][edge][-1] + mass;
                    # edge_cost = 1.*edge_mass + 1.;
                    WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
                    # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                    # WORLD[mode]['current_edge_costs'][edge] = edge_cost;
                    WORLD[mode]['current_edge_masses'][edge] = edge_mass;



    def planGTFSSeg(trip0,mode,GRAPHS,WORLD,NODES,mass=1,track=True,verbose=False):
        
        source = trip0[0];
        target = trip0[1];
        source = str(source);
        target = str(target);
        trip = (source,target);
        FEED = GRAPHS['gtfs']
        GRAPH = GRAPHS['transit'];

        PRECOMPUTE = WORLD['gtfs']['precompute']
        REACHED = PRECOMPUTE['reached']
        PREV_NODES = PRECOMPUTE['prev_nodes'];
        PREV_TRIPS = PRECOMPUTE['prev_trips'];

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

        if not(trip in WORLD[mode]['trips'].keys()):
            WORLD[mode]['trips'][trip] = {};
            WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
            #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
            WORLD[mode]['trips'][trip]['path'] = [];
            WORLD[mode]['trips'][trip]['mass'] = 0;

        WORLD[mode]['trips'][trip]['costs']['current_time'] = 1.*distance
        WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
        WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
        WORLD[mode]['trips'][trip]['costs']['current_switches'] = num_segments - 1; 
        WORLD[mode]['trips'][trip]['current_path'] = path;
        WORLD[mode]['trips'][trip]['mass'] = mass;
        
        if track==True:
            WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
            WORLD[mode]['trips'][trip]['costs']['money'].append(1)
            WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
            WORLD[mode]['trips'][trip]['costs']['switches'].append(num_segments-1)
            WORLD[mode]['trips'][trip]['path'].append(path);

        num_missing_edges = 0;
        if mass > 0:
            #print(path)
            for j,node in enumerate(path):
                if j < len(path)-1:
                    edge = (path[j],path[j+1],0)
                    if edge in WORLD[mode]['edge_masses']:
                        edge_mass = WORLD[mode]['edge_masses'][edge][-1] + mass;
                        # edge_cost = 1.*edge_mass + 1.;
                        WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
                        # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                        # WORLD[mode]['current_edge_costs'][edge] = edge_cost;    
                        WORLD[mode]['current_edge_masses'][edge] = edge_mass;
                    else:
                        num_missing_edges = num_missing_edges + 1
                        # if np.mod(num_missing_edges,10)==0:
                        #     print(num_missing_edges,'th missing edge...')
        if verbose:
            if num_missing_edges > 1:
                print('# edges missing in gtfs segment...',num_missing_edges)






    def querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group='group0'):
        """
        description -- queries the cost of travel (for a person) along a particular segment in a given mode
        inputs -- 
               start: start node
               end: end node
               mode: mode of travel
               PERSON: person object
               NODES: node conversion dict
               GRAPHS: graphs object
               WORLD: network information
        returns --
               cost: subjective cost of the trip
               path: optimal path (list of nodes)
        """
        
        start = seg_details[0]
        end = seg_details[1];
        if not((start,end) in WORLD[mode]['trips']):#.keys()):  
            if mode == 'gtfs':
                planGTFSSeg(seg_details,mode,GRAPHS,WORLD,NODES,mass=0);
            elif mode == 'ondemand':
                addDelivSeg((start,end),mode,GRAPHS,DELIVERY,WORLD,group=group,mass=0,PERSON=PERSON);

            elif (mode == 'drive') or (mode == 'walk'):
                planDijkstraSeg((start,end),mode,GRAPHS,WORLD,mass=0);
        # print(PERSON['prefs'].keys())
        # print('got here for mode...',mode)

        tot_cost = 0;
        weights = [];
        costs = [];
        COSTS = {};
        for l,factor in enumerate(PERSON['prefs'][mode]):
            weights.append(PERSON['weights'][mode][factor])
            costs.append(WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor])
            COSTS['factor'] = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor]
            # cost = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor] # possibly change for delivery
            # diff = cost; #-PERSON['prefs'][mode][factor]
            # tot_cost = tot_cost + PERSON['weights'][mode][factor]*diff;
        weights = np.array(weights);
        costs = np.array(costs);
        if 'logit_version' in PERSON: logit_version = PERSON['logit_version']
        else: logit_version = 'weighted_sum'
        tot_cost = applyLogitChoice(weights,costs,ver=logit_version);
        try: 
            path = WORLD[mode]['trips'][(start,end)]['current_path']
        except:
            path = None;

        return tot_cost,path,COSTS



        
    def applyLogitChoice(weights,values,offsets = [],ver='weighted_sum'):
        cost = 100000000.;
        if len(offsets) == 0: offsets = np.zeros(len(weights));
        if ver == 'weighted_sum':
            cost = weights@values;
        return cost




class NETWORK:
    def __init__(self,mode,params={}):
        self.mode = mode; #params['mode'];
        self.graph = params['graph']
        self.GRAPH = params['GRAPH']
        if 'feed' in params: self.feed = params['feed'];
        else: self.feed = None
        self.trips = {}
        self.edge_masses = {}
        self.edge_costs = {}
        self.current_edge_masses = {}
        self.current_edge_costs = {}
        self.edge_cost_poly = {};
        self.people = [];
        self.active_trips = [];

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
            self.precompute = {};
            self.precompute['reached'] = {};
            self.precompute['reached'] = {};
            self.precompute['prev_nodes'] = {};
            self.precompute['prev_trips'] = {};

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

    def removeMassFromEdges(mode,WORLD,GRAPHS):
        NETWORK = WORLD[mode]
        if mode=='gtfs':
            GRAPH = GRAPHS['transit'];
        else: 
            GRAPH = GRAPHS[mode];
        # edges = list(NETWORK['current_edge_masses']);
        # NETWORK['current_edge_costs'] = {};
        NETWORK['current_edge_masses'] = {};    
        for e,edge in enumerate(GRAPH.edges):
            # NETWORK['current_edge_costs'][edge] = 0;
            NETWORK['current_edge_masses'][edge] = 0;        

    def addTripMassToEdges(trip,NETWORK):
        if trip in NETWORK['trips']:
            TRIP = NETWORK['trips'][trip]
            mass = 1;
            path = TRIP['path']
            for j,node in enumerate(path):
                if j < len(path)-1:
                    edge = (path[j],path[j+1],0)
                    edge_mass = NETWORK['edge_masses'][edge][-1] + mass;
                    edge_cost = 1.*edge_mass + 1.;
                    NETWORK['edge_masses'][edge][-1] = edge_mass;
                    NETWORK['edge_costs'][edge][-1] = edge_cost;
                    current_mass = NETWORK['current_edge_masses'][edge];
                    NETWORK['current_edge_masses'][edge] = edge_cost;            
                    NETWORK['current_edge_costs'][edge] = edge_cost;            







    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
    ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 



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






    def UPDATE(self,params,verbose=True):

        mode = self.mode;
        kk = params['iter']
        alpha = params['alpha'];


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
                    if 'base_edge_masses' in WORLD[mode]:
                        current_edge_mass = current_edge_mass + WORLD[mode]['base_edge_masses'][edge];
                    expected_mass = invDriveCostFx(current_cost,poly)
                    #diff = WORLD[mode]['current_edge_masses'][edge] - expected_mass;
                    diff = current_edge_mass - expected_mass; 

                    new_edge_cost = WORLD[mode]['current_edge_costs'][edge] + alpha * diff
                    min_edge_cost = poly[0];
                    max_edge_cost = 10000000.;
                    new_edge_cost = np.min([np.max([min_edge_cost,new_edge_cost]),100000.])
                    WORLD[mode]['edge_costs'][edge].append(new_edge_cost)
                    WORLD[mode]['current_edge_costs'][edge] = new_edge_cost;            
                
            # if True: #'current_edge_costs' in WORLD[mode].keys():
            current_costs = WORLD[mode]['current_edge_costs'];
            # else: # INITIALIZE 
            #     current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}


            # print(current_costs)
            nx.set_edge_attributes(GRAPH,current_costs,'c');
            mode = 'drive'
            removeMassFromEdges(mode,WORLD,GRAPHS)
            segs = WORLD[mode]['active_trips']
            print('...with ',len(segs),' active trips...')    
            for i,seg in enumerate(WORLD[mode]['trips']):
                if seg in segs: 
                    if np.mod(i,500)==0: print('>>> segment',i,'...')
                    source = seg[0];
                    target = seg[1];
                    trip = (source,target)
                    mass = WORLD[mode]['trips'][trip]['mass']
                    planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=mass,track=True);
                    WORLD[mode]['trips'][trip]['active'] = False;
                else:
                    if len(WORLD[mode]['trips'][seg]['costs']['time'])==0:
                        WORLD[mode]['trips'][seg]['costs']['time'].append(0)
                    else:
                        recent_value = WORLD[mode]['trips'][seg]['costs']['time'][-1]
                        WORLD[mode]['trips'][seg]['costs']['time'].append(recent_value)

            if clear_active:
                WORLD[mode]['active_trips']  = [];

            ######## REMOVING MASS ON TRIPS ##########
            ######## REMOVING MASS ON TRIPS ##########
            for i,trip in enumerate(WORLD[mode]['trips']):
                TRIP = WORLD[mode]['trips'][trip];
                TRIP['mass'] = 0;
        if self.mode == 'walk':

        ####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
        ####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
        # def world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=False,clear_active=True):  ### ADDED TO CLASS 
                #graph,costs,sources, targets):
            if verbose: print('starting walking computations...')
            mode = 'walk'
            kk = WORLD['main']['iter']
            alpha = WORLD['main']['alpha']

            WALK = WORLD[mode];
            GRAPH = GRAPHS[WALK['graph']];    
            if (not('edge_masses' in WALK.keys()) or (kk==0)):
                WALK['edge_masses'] = {};
                WALK['edge_costs'] = {};
                WALK['current_edge_costs'] = {};
                WALK['edge_a0'] = {};
                WALK['edge_a1'] = {};
                for e,edge in enumerate(GRAPH.edges):
                    # WALK['edge_masses'][edge] = [0]
                    # WALK['edge_costs'][edge] = [1]
                    # WALK['current_edge_costs'][edge] = 1.;#GRAPH.edges[edge]['cost_fx'][0];
                    # WALK['edge_a0'][edge] = 1;
                    # WALK['edge_a1'][edge] = 1;               
                    WORLD[mode]['edge_masses'][edge] = [0]
                    current_edge_cost = WORLD[mode]['cost_fx'][edge][0];
                    WORLD[mode]['edge_costs'][edge] = [current_edge_cost]; #WORLD[mode]['cost_fx'][edge][0]];
                    WORLD[mode]['current_edge_costs'][edge] = current_edge_cost; 
                    WORLD[mode]['current_edge_masses'][edge] = 0. 
                    # WORLD[mode]['edge_a0'][edge] = 1;
                    # WORLD[mode]['edge_a1'][edge] = 1;

            else: 
                for e,edge in enumerate(GRAPH.edges):
                    # WALK['edge_masses'][edge].append(0)
                    # WALK['edge_costs'][edge].append(0)
                    # WALK['current_edge_costs'][edge] = 1;
                    # WORLD[mode]['edge_masses'][edge].append(0)

                    current_edge_cost = WORLD[mode]['current_edge_costs'][edge];# + alpha * WORLD[mode]['current_edge_masses'][edge] 
                    WORLD[mode]['edge_costs'][edge].append(current_edge_cost)
                    WORLD[mode]['current_edge_costs'][edge] = current_edge_cost;            
                
            # if True: #'current_edge_costs' in WORLD[mode].keys():
            current_costs = WORLD[mode]['current_edge_costs'];
            # else: # INITIALIZE 
            #     current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}        


            nx.set_edge_attributes(GRAPH,current_costs,'c');     

            mode = 'walk'
            removeMassFromEdges(mode,WORLD,GRAPHS)  
            segs = WORLD[mode]['active_trips']
            print('...with ',len(segs),' active trips...')        
            for i,seg in enumerate(WORLD[mode]['trips']):
                if seg in segs:
                    if np.mod(i,500)==0: print('>>> segment',i,'...')
                    source = seg[0];
                    target = seg[1];
                    trip = (source,target)
                    mass = WORLD[mode]['trips'][trip]['mass']
                    planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=mass,track=True);
                    WORLD[mode]['trips'][trip]['active'] = False;
                else:
                    if len(WORLD[mode]['trips'][seg]['costs']['time'])==0:
                        WORLD[mode]['trips'][seg]['costs']['time'].append(0)
                    else:
                        recent_value = WORLD[mode]['trips'][seg]['costs']['time'][-1]
                        WORLD[mode]['trips'][seg]['costs']['time'].append(recent_value)



            if clear_active:    
                WORLD[mode]['active_trips']  = [];

            ######## REMOVING MASS ON TRIPS ##########
            ######## REMOVING MASS ON TRIPS ##########
            for i,trip in enumerate(WORLD[mode]['trips']):
                TRIP = WORLD[mode]['trips'][trip];
                TRIP['mass'] = 0;



        if self.mode == 'ondemand':
        # def world_of_ondemand(WORLD,PEOPLE,DELIVERIES,GRAPHS,verbose=False,show_delivs='all',clear_active=True): ##### ADDED TO CLASSA 
            ##### ADDED TO CLASSA 
            if verbose: print('starting on-demand computations...')    

            #### PREV: tsp_wgrps
            #lam = grads['lam']
            #pickups = current_pickups(lam,all_pickup_nodes) ####
            #nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
            kk = WORLD['main']['iter']
            alpha = WORLD['main']['alpha']

            mode = 'ondemand'
            GRAPH = GRAPHS['ondemand'];
            ONDEMAND = WORLD['ondemand'];

            total_trips_to_plan = WORLD[mode]['active_trips'];#_direct'] + WORLD['ondemand']['active_trips_shuttle'];
            num_total_trips = len(total_trips_to_plan);

            ##### SORTING TRIPS BY GROUPS: 
            if 'groups' in DELIVERIES:
                GROUPS_OF_TRIPS = {};
                for _,group in enumerate(DELIVERIES['groups']):
                    GROUPS_OF_TRIPS[group] = [];

                for i,trip in enumerate(WORLD[mode]['active_trips']):
                    TRIP = WORLD[mode]['trips'][trip];
                    if 'group' in TRIP:
                        group = TRIP['group'];
                        GROUPS_OF_TRIPS[group].append(trip);

                #### TO ADD: SORT TRIPS BY GROUP
                for _,group in enumerate(GROUPS_OF_TRIPS):

                    trips_to_plan = GROUPS_OF_TRIPS[group]
                    DELIVERY = DELIVERIES['groups'][group]



                    ########### -- XX START HERE ---- ####################
                    ########### -- XX START HERE ---- ####################
                    ########### -- XX START HERE ---- ####################
                    print('...with',len(trips_to_plan),'active ondemand trips...')

                    # poly = WORLD[mode]['cost_poly'];
                    # poly = np.array([-6120.8676711, 306.5130127]) # 1st order
                    # poly = np.array([-8205.87778054,   342.32193064]) # 1st order 
                    # poly = np.array([5047.38255623,-288.78570445,6.31107635]); # 2nd order


                    if 'fit' in DELIVERY:
                        poly  = DELIVERY['fit']['poly'];
                    elif 'fit' in WORLD['ondemand']:
                        poly = WORLD['ondemand']['fit']['poly'];
                    else:
                        poly = WORLD['ondemand']['poly'];

                    # poly = np.array([6.31107635, -288.78570445, 5047.38255623]) # 2nd order 


                    num_trips = len(trips_to_plan);

                    ### dumby values...
                    MDF = []; nodes = [];
                    if len(trips_to_plan)>0:
                        nodes_to_update = [DELIVERY['depot_node']];
                        nodeids_to_update = [0];
                        for j,trip in enumerate(trips_to_plan):
                            TRIP = WORLD[mode]['trips'][trip]
                            nodes_to_update.append(trip[0]);
                            nodes_to_update.append(trip[1]);
                            nodeids_to_update.append(TRIP['pickup_node_id'])
                            nodeids_to_update.append(TRIP['dropoff_node_id'])


                        updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                        payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);


                        # import pickle
                        # fileObj = open('payload1', 'wb')
                        # pickle.dump(payload,fileObj)
                        # fileObj.close()

                        manifests = optimizer.offline_solver(payload) 
                        # manifest = optimizer(payload) ####
                        # manifest = fakeManifest(payload)

                        PDF = payloadDF(payload,GRAPHS)
                        MDF = manifestDF(manifests,PDF)

                        DELIVERY['current_PDF'] = PDF;
                        DELIVERY['current_MDF'] = MDF;
                        DELIVERY['current_payload'] = payload;
                        DELIVERY['current_manifest'] = manifests;


                    average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,WORLD[mode]['current_expected_cost'])

                    # print(poly)
                    # print(WORLD[mode]['current_expected_cost'])

                    # expected_num_trips = invDriveCostFx(WORLD[mode]['current_expected_cost'],poly)
                    expected_num_trips = invDriveCostFx(DELIVERY['current_expected_cost'],poly)            


                    print('...expected num of trips given cost:',expected_num_trips)
                    print('...actual num trips:',num_trips)

                    diff = num_trips - expected_num_trips

                    print('...adjusting cost estimate by',alpha*diff);
                    # new_ave_cost = WORLD[mode]['current_expected_cost'] + alpha * diff
                    new_ave_cost = DELIVERY['current_expected_cost'] + alpha * diff

                    min_ave_cost = poly[0];
                    max_ave_cost = 100000000.;
                    new_ave_cost = np.min([np.max([min_ave_cost,new_ave_cost]),max_ave_cost])

                    print('...changing cost est from',DELIVERY['current_expected_cost'],'to',new_ave_cost )

                    DELIVERY['expected_cost'].append(new_ave_cost)
                    DELIVERY['current_expected_cost'] = new_ave_cost

                    DELIVERY['actual_average_cost'].append(average_time)

                for i,trip in enumerate(WORLD[mode]['trips']):        
                    if not(trip in total_trips_to_plan):
                        if len(WORLD[mode]['trips'][trip]['costs']['time'])==0:
                            WORLD[mode]['trips'][trip]['costs']['time'].append(0)
                            WORLD[mode]['trips'][trip]['costs']['current_time']= 0;
                        else:
                            recent_value = WORLD[mode]['trips'][trip]['costs']['time'][-1]
                            WORLD[mode]['trips'][trip]['costs']['time'].append(recent_value)
                            WORLD[mode]['trips'][trip]['costs']['current_time'] = recent_value

                if clear_active: 
                    WORLD[mode]['active_trips']  = [];



            else:
                ########### -- XX START HERE ---- ####################
                ########### -- XX START HERE ---- ####################
                ########### -- XX START HERE ---- ####################
                print('...with',len(trips_to_plan),'active ondemand trips...')

                # poly = WORLD[mode]['cost_poly'];
                # poly = np.array([-6120.8676711, 306.5130127]) # 1st order
                # poly = np.array([-8205.87778054,   342.32193064]) # 1st order 
                # poly = np.array([5047.38255623,-288.78570445,6.31107635]); # 2nd order


                if 'fit' in WORLD['ondemand']:
                    poly = WORLD['ondemand']['fit']['poly'];
                else:
                    poly = WORLD['ondemand']['poly'];

                # poly = np.array([6.31107635, -288.78570445, 5047.38255623]) # 2nd order 


                num_trips = len(trips_to_plan);

                ### dumby values...
                MDF = []; nodes = [];
                if len(trips_to_plan)>0:
                    nodes_to_update = [DELIVERY['depot_node']];
                    nodeids_to_update = [0];
                    for j,trip in enumerate(trips_to_plan):
                        TRIP = WORLD[mode]['trips'][trip]
                        nodes_to_update.append(trip[0]);
                        nodes_to_update.append(trip[1]);
                        nodeids_to_update.append(TRIP['pickup_node_id'])
                        nodeids_to_update.append(TRIP['dropoff_node_id'])

                    updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                    payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

                    # import pickle
                    # fileObj = open('payload1', 'wb')
                    # pickle.dump(payload,fileObj)
                    # fileObj.close()

                    manifests = optimizer.offline_solver(payload) 
                    # manifest = optimizer(payload) ####
                    # manifest = fakeManifest(payload)


                    PDF = payloadDF(payload,GRAPHS)
                    MDF = manifestDF(manifests,PDF)

                    DELIVERY['current_PDF'] = PDF;
                    DELIVERY['current_MDF'] = MDF;
                    DELIVERY['current_payload'] = payload;
                    DELIVERY['current_manifest'] = manifests;



                average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,WORLD[mode]['current_expected_cost'])

                # print(poly)
                # print(WORLD[mode]['current_expected_cost'])

                expected_num_trips = invDriveCostFx(WORLD[mode]['current_expected_cost'],poly)





                print('...expected num of trips given cost:',expected_num_trips)
                print('...actual num trips:',num_trips)

                diff = num_trips - expected_num_trips

                print('...adjusting cost estimate by',alpha*diff);
                new_ave_cost = WORLD[mode]['current_expected_cost'] + alpha * diff



                min_ave_cost = poly[0];
                max_ave_cost = 100000000.;
                new_ave_cost = np.min([np.max([min_ave_cost,new_ave_cost]),max_ave_cost])

                print('...changing cost est from',WORLD[mode]['current_expected_cost'],'to',new_ave_cost )

                WORLD[mode]['expected_cost'].append(new_ave_cost)
                WORLD[mode]['current_expected_cost'] = new_ave_cost

                WORLD[mode]['actual_average_cost'].append(average_time)

                for i,trip in enumerate(WORLD[mode]['trips']):        
                    if not(trip in trips_to_plan):
                        if len(WORLD[mode]['trips'][trip]['costs']['time'])==0:
                            WORLD[mode]['trips'][trip]['costs']['time'].append(0)
                        else:
                            recent_value = WORLD[mode]['trips'][trip]['costs']['time'][-1]
                            WORLD[mode]['trips'][trip]['costs']['time'].append(recent_value)

                if clear_active: 
                    WORLD[mode]['active_trips']  = [];
            # WORLD[mode]['active_trips_shuttle'] = [];
            # WORLD[mode]['active_trips_direct'] = [];



            if False:
                ########### -- XX START HERE ---- ####################
                trips_to_plan = WORLD['ondemand']['active_trips_direct']    
                print('...with',len(trips_to_plan),'active direct trips...')
                DELIVERY0 = DELIVERY['direct'];
                maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));

                divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD,maxTrips);
                #######################################################

                if show_delivs=='all': delivs_to_show = list(range(len(list(DELIVERY0.keys()))));
                elif isinstance(show_delivs,int): delivs_to_show = list(range(show_delivs));
                else: delivs_to_show = show_delivs;

                for i,delivery in enumerate(DELIVERY0):
                    if np.mod(i,200)==0: print('direct',i,'...')
                    DELIV = DELIVERY0[delivery];
                    active_trips = DELIV['active_trips'];
                    sources = []; targets = [];
                    start = DELIV['nodes']['source'];        
                    [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
                    [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
                    planDelivSegs(sources,targets,start,delivery,DELIVERY0,GRAPHS,WORLD,track=True);
                    path = DELIV['current_path'];
                    if i in delivs_to_show:
                        for j,node in enumerate(path):
                            if j < len(path)-1:
                                edge = (path[j],path[j+1],0)
                                edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
                                ONDEMAND['current_edge_masses'][edge] = edge_mass;
                                ONDEMAND['current_edge_masses1'][edge] = edge_mass;



                ################################# 


                    # DELIV['active_trips']  = [];

                trips_to_plan = WORLD['ondemand']['active_trips_shuttle']
                print('...with',len(trips_to_plan),'active shuttle trips...')
                DELIVERY0 = DELIVERY['shuttle'];
                maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));
                divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD);

                if show_delivs=='all': delivs_to_show = list(range(len(list(DELIVERY0.keys()))));
                elif isinstance(show_delivs,int): delivs_to_show = list(range(show_delivs));
                else: delivs_to_show = show_delivs;


                for i,delivery in enumerate(DELIVERY0):
                    if np.mod(i,200)==0: print('shuttle',i,'...')
                    DELIV = DELIVERY0[delivery];
                    active_trips = DELIV['active_trips'];
                    sources = []; targets = [];
                    start = DELIV['nodes']['source'];        
                    [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
                    [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
                    planDelivSegs(sources,targets,start,delivery,DELIVERY0,GRAPHS,WORLD,track=True);
                    # DELIV['active_trips']  = [];
                    path = DELIV['current_path'];          
                    if i in delivs_to_show:        
                        for j,node in enumerate(path):
                            if j < len(path)-1:
                                edge = (path[j],path[j+1],0)
                                edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
                                ONDEMAND['current_edge_masses'][edge] = edge_mass;
                                ONDEMAND['current_edge_masses2'][edge] = edge_mass;
                WORLD[mode]['active_trips']  = [];
                WORLD[mode]['active_trips_shuttle'] = [];
                WORLD[mode]['active_trips_direct'] = [];

                ######## REMOVING MASS ON TRIPS ##########
                ######## REMOVING MASS ON TRIPS ##########
                for i,trip in enumerate(WORLD[mode]['trips']):
                    TRIP = WORLD[mode]['trips'][trip];
                    TRIP['mass'] = 0;












        if self.mode == 'gtfs':


        ####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 
        ####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 


        # def world_of_gtfs(WORLD,PEOPLE,GRAPHS,NODES,verbose=False,clear_active=True):  ##### ADDED TO CLASS 
            if verbose: print('starting gtfs computations...')
            #raptor_gtfs
            kk = WORLD['main']['iter']
            alpha = WORLD['main']['alpha']

            GTFS = WORLD['gtfs'];
            GRAPH =GRAPHS['transit']; 
            FEED = GRAPHS['gtfs'];
            if (not('edge_masses_gtfs' in GTFS.keys()) or (kk==0)):
                GTFS['edge_costs'] = {};
                GTFS['edge_masses'] = {};
                GTFS['current_edge_costs'] = {};
                GTFS['edge_a0'] = {};
                GTFS['edge_a1'] = {};        
                for e,edge in enumerate(GRAPH.edges):
                    GTFS['edge_masses'][edge] = [0]
                    GTFS['edge_costs'][edge] = [1]
                    GTFS['current_edge_costs'][edge] = 1;
                    GTFS['edge_a0'][edge] = 1;
                    GTFS['edge_a1'][edge] = 1;               
                
            if 'current_edge_costs' in GTFS.keys():
                current_costs = GTFS['current_edge_costs'];
            else: 
                current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}
                
            #nx.set_edge_attributes(GRAPH,current_costs,'c');

            mode = 'gtfs'
            removeMassFromEdges(mode,WORLD,GRAPHS) 
            segs = WORLD[mode]['active_trips']
            print('...with ',len(segs),' active trips...')        

            for i,seg in enumerate(WORLD[mode]['trips']):
            # for i,seg in enumerate(segs):
                if seg in segs:
                    source = seg[0];
                    target = seg[1];
                    trip = (source,target)
                    mass = WORLD[mode]['trips'][trip]['mass']
                    planGTFSSeg((source,target),mode,GRAPHS,WORLD,NODES,mass=mass,track=True);
                    WORLD[mode]['trips'][trip]['active'] = False;
                else:
                    if len(WORLD[mode]['trips'][seg]['costs']['time'])==0:
                        WORLD[mode]['trips'][seg]['costs']['time'].append(0)
                    else:
                        recent_value = WORLD[mode]['trips'][seg]['costs']['time'][-1]
                        WORLD[mode]['trips'][seg]['costs']['time'].append(recent_value)


            if clear_active:
                WORLD[mode]['active_trips']  = [];

                ######## REMOVING MASS ON TRIPS ##########
                ######## REMOVING MASS ON TRIPS ##########
                print('REMOVING MASS FROM GTFS TRIPS...')
                for i,trip in enumerate(WORLD[mode]['trips']):
                    TRIP = WORLD[mode]['trips'][trip];
                    TRIP['mass'] = 0;



        def world_of_transit(WORLD,PEOPLE,GRAPHS,verbose=False):   ###### ADDED TO CLASS 
            # uses GTFS feed
            if verbose: print('starting transit computations...')
            #raptor_gtfs
            TRANSIT = WORLD['transit'];
            GRAPH = GRAPHS[TRANSIT['graph']];

            ########################################################################################
            #### --- NOT SURE WE NEED THIS>>>> 
            if not('edge_masses' in TRANSIT.keys()):
                TRANSIT['edge_masses'] = {};
                TRANSIT['edge_costs'] = {};
                TRANSIT['current_edge_costs'] = {};
                TRANSIT['edge_a0'] = {};
                TRANSIT['edge_a1'] = {};        
                for e,edge in enumerate(GRAPH.edges):
                    TRANSIT['edge_masses'][edge] = [0]
                    TRANSIT['edge_costs'][edge] = [1]
                    TRANSIT['current_edge_costs'][edge] = 1;
                    TRANSIT['edge_a0'][edge] = 1;
                    TRANSIT['edge_a1'][edge] = 1;               

            if 'current_edge_costs' in TRANSIT.keys():
                current_costs = TRANSIT['current_edge_costs'];
            else: 
                current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}        
            nx.set_edge_attributes(GRAPH,current_costs,'c');
            ########################################################################################
            

            mode = 'transit'
            removeMassFromEdges(mode,WORLD,GRAPHS) 
            segs = WORLD[mode]['active_trips']
            for i,seg in enumerate(segs):
                source = seg[0];
                target = seg[1];
                trip = (source,target)
                planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=1,track=True);
                WORLD[mode]['trips'][trip]['active'] = False;    
            WORLD[mode]['active_trips']  = [];    

        def world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=False):  ### ADDED TO CLASS 
            if verbose: print('starting transit computations...')
            #raptor_gtfs
            TRANSIT = WORLD['transit'];
            GRAPH = GRAPHS[TRANSIT['graph']];
            if not('edge_masses' in TRANSIT.keys()):
                TRANSIT['edge_masses'] = {};
                TRANSIT['edge_costs'] = {};
                TRANSIT['current_edge_costs'] = {};
                TRANSIT['edge_a0'] = {};
                TRANSIT['edge_a1'] = {};        
                for e,edge in enumerate(GRAPH.edges):
                    TRANSIT['edge_masses'][edge] = [0]
                    TRANSIT['edge_costs'][edge] = [1]
                    TRANSIT['current_edge_costs'][edge] = 1;
                    TRANSIT['edge_a0'][edge] = 1;
                    TRANSIT['edge_a1'][edge] = 1;               
                
            if 'current_edge_costs' in TRANSIT.keys():
                current_costs = TRANSIT['current_edge_costs'];
            else: 
                current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}
                
            nx.set_edge_attributes(GRAPH,current_costs,'c');
            

            mode = 'transit'
            removeMassFromEdges(mode,WORLD,GRAPHS) 
            segs = WORLD[mode]['active_trips']
            print('...with ',len(segs),' active trips...')        
            for i,seg in enumerate(segs):
                source = seg[0];
                target = seg[1];
                trip = (source,target)
                mass = WORLD[mode]['trips'][trip]['mass']
                planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=mass,track=True);
                WORLD[mode]['trips'][trip]['active'] = False;    
            WORLD[mode]['active_trips']  = [];




    def compute_UncongestedEdgeCosts(self):  #### ADDED TO CLASS 
    # def compute_UncongestedEdgeCosts(WORLD,GRAPHS):  #### ADDED TO CLASS 
        modes = ['drive','walk'];
        for mode in modes:
            GRAPH = GRAPHS[mode]
            edge_costs0 = {}
            for e,edge in enumerate(GRAPH.edges):
                poly = WORLD[mode]['cost_fx'][edge];
                edge_costs0[edge] = poly[0]
            nx.set_edge_attributes(GRAPH,edge_costs0,'c0')

    def compute_UncongestedTripCosts(self): #WORLD,GRAPHS):   ### ADDED TO CLASS 
    # def compute_UncongestedTripCosts(WORLD,GRAPHS):   ### ADDED TO CLASS 
        modes = ['drive','walk','gtfs','ondemand'];
        for mode in modes:
            if mode == 'ondemand': GRAPH = GRAPHS['drive']
            else: GRAPH = GRAPHS[mode]
            for t,trip in enumerate(WORLD[mode]['trips']):
                source = trip[0]; target = trip[1];
                try:
                    temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c0'); #Dumbfunctions
                    distance = temp[0];
                    path = temp[1]; 
                except: 
                    #print('no path found for bus trip ',trip,'...')
                    distance = 1000000000000;
                    path = [];
                WORLD[mode]['trips'][trip]['uncongested'] = {};
                WORLD[mode]['trips'][trip]['uncongested']['path'] = path;
                WORLD[mode]['trips'][trip]['uncongested']['costs'] = {}; 
                WORLD[mode]['trips'][trip]['uncongested']['costs']['time'] = distance;
                WORLD[mode]['trips'][trip]['uncongested']['costs']['money'] = 0;
                WORLD[mode]['trips'][trip]['uncongested']['costs']['conven'] = 0;
                WORLD[mode]['trips'][trip]['uncongested']['costs']['switches'] = 0;        













class PEOPLE: 



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

    def __init__(self,params): 
    # def generatePopulation(GRAPHS,DELIVERY,WORLD,NODES,VEHS,LOCS,PRE,params,verbose=True):  ##### ADDED TO CLASS
        people_tags = list(PRE); #params['people_tags']
        ORIG_LOC = LOCS['orig'] #params['ORIG_LOC'];
        DEST_LOC = LOCS['dest'] #params['DEST_LOC'];
        modes = params['modes'];
        graphs = params['graphs'];
        nodes = params['nodes'];
        factors = params['factors'];
        mass_scale = params['mass_scale']

        PEOPLE = {};
        start_time3 = time.time()
        print_interval = 10;


        mean = 500; stdev = 25;
        means = [mean,mean,mean,mean,mean];
        stdevs = [stdev,stdev,stdev,stdev,stdev];
        STATS = {}
        for m,mode in enumerate(modes):
            STATS[mode] = {};
            STATS[mode]['mean'] = dict(zip(factors,means));
            STATS[mode]['stdev'] = dict(zip(factors,stdevs));

        person_str = 'person';
        print('GENERATING POPULATION OF',len(people_tags),'...')

        DELIVERY['grpsDF']['num_possible_trips'] = list(np.zeros(len(DELIVERY['grpsDF'])));

        grpsDF = DELIVERY['grpsDF']
        for i,person in enumerate(people_tags):


            if np.mod(i,print_interval)==0:
                end_time3 = time.time()
                if verbose:
                    print(person)
                    print('time to add',print_interval, 'people: ',end_time3-start_time3)
                start_time3 = time.time()

            PEOPLE[person] = {
                            'mass': None,
                            'current_choice': None,
                            'current_cost': 0,
                            'choice_traj': [],
                            'delivery_grps': {'straight':None,'initial':None,'final':None},
                            'delivery_grp': None,        
                            'delivery_grp_initial': None,
                            'delivery_grp_final': None,
                            'logit_version': 'weighted_sum',
                            'trips':[],
                            'opts': ['drive','ondemand','walk','transit'],
                            'opts2': [random.choice(['walk','ondemand','drive']),
                                      'transit',
                                      random.choice(['walk','ondemand','drive'])],
                            'cost_traj': [],
                            'costs':{}, #opt, factor
                            'prefs':{}, #opt, factor                    
                            'weights':{}, #opt, factor

                            'am': int(1),
                            'wc': int(0),
                            'pickup_time_window_start': 18000,
                            'pickup_time_window_end': 19800,
                            'dropoff_time_window_start': 18900,
                            'dropoff_time_window_end': 22200,

                            'time_windows': {},
                            'booking_id': 39851211,
                            'pickup_node_id': 1,
                            'dropoff_node_id': 2
                            }



            PERSON = PEOPLE[person];
            PERSON['mass_total'] = PRE[person]['pop']
            PERSON['mass'] = PRE[person]['pop']*mass_scale


            if False:
                orig_loc = ORIG_LOC[i];
                dest_loc = DEST_LOC[i];
            else:
                orig_loc = PRE[person]['orig_loc'];
                dest_loc = PRE[person]['dest_loc'];


            PERSON['pickup_pt'] = {'lat': 35.0296296, 'lon': -85.2301767};
            PERSON['dropoff_pt'] = {'lat': 35.0734152, 'lon': -85.1315328};

            # PRE[person]['weight'][mode][factor]

            #### PREFERENCES #### PREFERENCES #### PREFERENCES #### PREFERENCES 
            ###################################################################
            if 'logit_version' in PRE[person]: PERSON['logit_version'] = PRE[person]['logit_version']; #OVERWRITES ABOVE

            for m,mode in enumerate(modes):
                PERSON['costs'][mode] = {};
                PERSON['prefs'][mode] = {};
                PERSON['weights'][mode] = {};
                for j,factor in enumerate(factors):
                    sample_pt = STATS[mode]['mean'][factor] + STATS[mode]['stdev'][factor]*(np.random.rand()-0.5)
                    PERSON['prefs'][mode][factor] = 0; #sample_pt
                    PERSON['costs'][mode][factor] = 0.

                    try:
                        PERSON['weights'][mode][factor] = PRE[person]['weight'][mode][factor]
                    except:
                        if factor == 'time': PERSON['weights'][mode][factor] = 1.
                        else: PERSON['weights'][mode][factor] = 1.
                # PERSON['weights'][mode] = dict(zip(factors,np.ones(len(factors))));

            ###### DELIVERY ###### DELIVERY ###### DELIVERY ###### DELIVERY ###### DELIVERY 
            ###############################################################################

            picked_deliveries = {'direct':None,'initial':None,'final':None}
            person_loc = orig_loc
            dist = 10000000;
            picked_delivery = None;    
            for k,delivery in enumerate(DELIVERY['direct']):
                DELIV = DELIVERY['direct'][delivery]
                loc = DELIV['loc']

                diff = np.array(list(person_loc))-np.array(list(loc));
                if mat.norm(diff)<dist:
                    PERSON['delivery_grps']['direct'] = delivery
                    dist = mat.norm(diff);
                    picked_deliveries['direct'] = delivery;

            
            person_loc = orig_loc
            dist = 10000000;
            picked_delivery = None;    
            for k,delivery in enumerate(DELIVERY['shuttle']):
                DELIV = DELIVERY['shuttle'][delivery]
                loc = DELIV['loc']
                diff = np.array(list(person_loc))-np.array(list(loc));
                if mat.norm(diff)<dist:
                    PERSON['delivery_grps']['initial'] = delivery
                    dist = mat.norm(diff);
                    picked_deliveries['initial'] = delivery;            


            person_loc = dest_loc;
            dist = 10000000;
            picked_delivery = None;    
            for k,delivery in enumerate(DELIVERY['shuttle']):
                DELIV = DELIVERY['shuttle'][delivery]
                loc = DELIV['loc']
                diff = np.array(list(person_loc))-np.array(list(loc));
                if mat.norm(diff)<dist:
                    PERSON['delivery_grps']['final'] = delivery
                    dist = mat.norm(diff);
                    picked_deliveries['final'] = delivery;              

            if not(picked_deliveries['direct']==None):
                DELIVERY['direct'][picked_deliveries['direct']]['people'].append(person)
            if not(picked_deliveries['initial']==None):        
                DELIVERY['shuttle'][picked_deliveries['initial']]['people'].append(person)    
            if not(picked_deliveries['final']==None):        
                DELIVERY['shuttle'][picked_deliveries['final']]['people'].append(person)             


            ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 
            ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 
            ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1     

            
            PERSON['trips'] = {}; 


            if 'seg_types' in PRE[person]:
                seg_types = PRE[person]['seg_types']
                PERSON['trips_to_consider'] = PRE[person]['seg_types']

            elif False: 
                samp = np.random.rand(1);
                if samp<0.15:
                    seg_types = [('drive',),
                                 ('ondemand',),
                                 ('walk','gtfs','walk'),
                                 ('walk','gtfs','ondemand'),
                                 ('ondemand','gtfs','walk'),
                                 ('ondemand','gtfs','ondemand')
                                ];
                    PERSON['trips_to_consider'] = seg_types
                elif (samp >= 0.15) & (samp <= 0.3):
                    seg_types = [('walk','gtfs','walk'),
                                 ('walk','gtfs','ondemand'),
                                 ('ondemand','gtfs','walk'),
                                 ('ondemand','gtfs','ondemand')
                                ];
                    PERSON['trips_to_consider'] = seg_types                        
                else:
                    seg_types = [('drive',)]
                    PERSON['trips_to_consider'] = seg_types

            else: 
                samp = np.random.rand(1);
                if PRE[person]['take_car'] == 0.:
                    seg_types = [('ondemand',),
                                 ('walk','gtfs','walk'),
                                 ('walk','gtfs','ondemand'),
                                 ('ondemand','gtfs','walk'),
                                 ('ondemand','gtfs','ondemand')
                                ];              
                    PERSON['trips_to_consider'] = seg_types
                elif (PRE[person]['take_car']==1) & (samp < 0.3):
                    seg_types = [('drive',),
                                 ('ondemand',),
                                 ('walk','gtfs','walk'),
                                 ('walk','gtfs','ondemand'),
                                 ('ondemand','gtfs','walk'),
                                 ('ondemand','gtfs','ondemand')
                                ];
                    PERSON['trips_to_consider'] = seg_types                        
                else:
                    seg_types = [('drive',)]
                    PERSON['trips_to_consider'] = seg_types



            start_time4 = time.time()


            new_trips_to_consider = [];
            for _,segs in enumerate(seg_types):


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

                    NODES = addNodeToDF(start_node,mode1,GRAPHS,NODES);
                    NODES = addNodeToDF(end_node,mode1,GRAPHS,NODES);
                    #updateNodesDF(NODES);

                    nodes_temp = [{'nodes':[start_node],'type':mode1},
                                  {'nodes':[end_node],'type':mode1}]


                    ##### SELECTING GROUP ######
                    if mode1 == 'ondemand':
                        grouptag,groupind = selectDeliveryGroup((start_node,end_node),grpsDF,GRAPHS,typ='direct')
                        if len(grouptag)==0: add_new_trip = False;
                        else:  grpsDF.iloc[groupind]['num_possible_trips'] = grpsDF.iloc[groupind]['num_possible_trips'] + 1;



                    ############################

                    deliveries_temp = [];
                    deliveries_temp2 = [];
                    if mode1=='ondemand':
                        if not(picked_deliveries['direct']==None):
                            DELIVERY['direct'][picked_deliveries['direct']]['sources'].append(start_node);
                            DELIVERY['direct'][picked_deliveries['direct']]['targets'].append(end_node);
                        deliveries_temp.append(picked_deliveries['direct'])


                        # deliveries_temp2.append('group0'); #### EDITTING 
                        deliveries_temp2.append(grouptag); #### EDITTING 

                    end_time = time.time();
                    #print("segment1: ",np.round(end_time-start_time,4))

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
                        transit1_walk,transit2_walk = nearest_applicable_gtfs_node(mode2,GRAPHS,WORLD,NODES,
                                                                     orig_loc[0],orig_loc[1],
                                                                     dest_loc[0],dest_loc[1]);
                                                                     # rad1=rad1,rad2=rad2);  ####HERE 
                        # print('transit node 1 is type...',type(transit2_walk))

                    if mode1 == 'ondemand':
                        initial_delivery = PERSON['delivery_grps']['initial'];
                        transit_node1 = DELIVERY['shuttle'][initial_delivery]['nodes']['transit2'];
                        if mode2=='gtfs':
                            transit_node1 = convertNode(transit_node1,'transit','gtfs',NODES)                
                    elif mode1 == 'walk':
                        transit_node1 = transit1_walk;
                        # transit_node1 = nearest_nodes(mode2,GRAPHS,NODES,orig_loc[0],orig_loc[1]);  ####HERE 
                        # print('old transit node 1 is type...',type(transit_node1))
                        #transit_node1 = ox.distance.nearest_nodes(GRAPHS[mode2], orig_loc[0],orig_loc[1]);

                    if mode3 == 'ondemand':
                        final_delivery = PERSON['delivery_grps']['final'];
                        transit_node2 = DELIVERY['shuttle'][final_delivery]['nodes']['transit2'];                                
                        if mode2=='gtfs':
                            transit_node2 = convertNode(transit_node2,'transit','gtfs',NODES)

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
                    NODES = addNodeToDF(start_node,mode1,GRAPHS,NODES);
                    NODES = addNodeToDF(end_node,mode3,GRAPHS,NODES);            
                    NODES = addNodeToDF(transit_node1,mode2,GRAPHS,NODES);
                    NODES = addNodeToDF(transit_node2,mode2,GRAPHS,NODES);
                    

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


                    # transit_node1 = convertNode(transit_node1,'transit','gtfs',NODES)

                    if mode1 == 'ondemand':
                        if mode2=='gtfs':
                            transit_node_converted = convertNode(transit_node1,'gtfs','ondemand',NODES)
                        grouptag1,groupind1 = selectDeliveryGroup((start_node,transit_node_converted),grpsDF,GRAPHS,typ='shuttle')
                        if len(grouptag1)==0: add_new_trip = False;
                        else:  grpsDF.iloc[groupind1]['num_possible_trips'] = grpsDF.iloc[groupind1]['num_possible_trips'] + 1;

                    if mode3=='ondemand':
                        if mode2=='gtfs':
                            transit_node_converted = convertNode(transit_node2,'gtfs','ondemand',NODES)
                        grouptag3,groupind3 = selectDeliveryGroup((transit_node_converted,end_node),grpsDF,GRAPHS,typ='shuttle')
                        if len(grouptag3)==0: add_new_trip = False;
                        else:  grpsDF.iloc[groupind3]['num_possible_trips'] = grpsDF.iloc[groupind3]['num_possible_trips'] + 1;





                    deliveries_temp = [None,None,None];
                    deliveries_temp2 = [None,None,None];
                    if mode1=='ondemand':  
                        if not(picked_deliveries['initial']==None):                
                            DELIVERY['shuttle'][picked_deliveries['initial']]['sources'].append(start_node);
                            deliveries_temp.append(picked_deliveries['initial'])
                        # deliveries_temp2[0] = 'group1' #### EDITTING 
                        deliveries_temp2[0] = grouptag1 #### EDITTING 
                    if mode3=='ondemand':
                        if not(picked_deliveries['final']==None):
                            DELIVERY['shuttle'][picked_deliveries['final']]['targets'].append(end_node);
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
                    TRIP = makeTrip(segs,nodes_temp,NODES,deliveries_temp2,deliveries_temp)
                    PERSON['trips'][segs] = TRIP;
                    end_time = time.time()
                #print('time to make trip: ',np.round(end_time-start_time,4))

            PERSON['trips_to_consider'] = new_trips_to_consider 
        DELIVERY['grpsDF'] = grpsDF.copy();
            ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------  
            ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------            
        return PEOPLE


    def UPDATE(self):
    # def update_choices(PEOPLE, DELIVERY, NODES, GRAPHS, WORLD, version=1,verbose=False,takeall=False):
        if verbose: print('updating choices')
        ## clear options
    #     for o,opt in enumerate(WORLD):
    #         WORLD[opt]['people'] = [];
            
        people_chose = {};
        for i,person in enumerate(PEOPLE):
            if np.mod(i,200)==0: print(person,'...')
            PERSON = PEOPLE[person];
            
            delivery_grp = PERSON['delivery_grp'];        
            delivery_grp_inital = PERSON['delivery_grp_initial'];
            delivery_grp_final = PERSON['delivery_grp_final'];        
            COMPARISON = [];

            trips_to_consider = PERSON['trips_to_consider'];

            for k,trip in enumerate(trips_to_consider):
                TRIP = PERSON['trips'][trip];
                try: 
                    queryTrip(TRIP,PERSON,NODES,GRAPHS,DELIVERY,WORLD)
                    COMPARISON.append(TRIP['current']['cost']);
                    cost = TRIP['current']['cost'];
                    # print('cost of',trip,'...',cost)
                except:
                    COMPARISON.append(10000000000000000);

            ind = np.argmin(COMPARISON);
            trip_opt = trips_to_consider[ind];

            current_choice = ind;
            current_cost = COMPARISON[ind]

            PERSON['current_choice'] = current_choice;
            PERSON['current_cost'] = current_cost;

            PERSON['choice_traj'].append(current_choice);
            PERSON['cost_traj'].append(current_cost);
            # updating world choice...






            if takeall==False:
                tripstotake = [PERSON['trips'][trip_opt]];
            else:
                tripstotake = [PERSON['trips'][zz] for _,zz in enumerate(trips_to_consider)];


            for _,CHOSEN_TRIP in enumerate(tripstotake):
                num_segs = len(CHOSEN_TRIP['structure'])
                for k,SEG in enumerate(CHOSEN_TRIP['structure']):
                    try:
                        mode = SEG['mode'];

                        # if mode == 'gtfs': print(SEG) 
                #             start = SEG['start_nodes'][SEG['opt_start']]
                #             end = SEG['end_nodes'][SEG['opt_end']
                        start = SEG['opt_start']
                        end = SEG['opt_end']
                        #print(WORLD[mode]['trips'][(start,end)])
                        #if not(mode in ['transit']):

                        # print(mode)
                        # print((start,end))
                        # if mode=='transit':
                        #     print(WORLD[mode]['trips'])
                        # try: 
                        trip = (start,end)

                        if (start,end) in WORLD[mode]['trips']:
                            WORLD[mode]['trips'][(start,end)]['mass'] = WORLD[mode]['trips'][(start,end)]['mass'] + PERSON['mass'];
                            WORLD[mode]['trips'][(start,end)]['active'] = True; 

                            if not((start,end) in WORLD[mode]['active_trips']):
                                WORLD[mode]['active_trips'].append((start,end));

                            if (num_segs == 3) & (mode == 'ondemand'):
                                WORLD[mode]['active_trips_shuttle'].append((start,end));
                            if (num_segs == 1) & (mode == 'ondemand'):
                                WORLD[mode]['active_trips_direct'].append((start,end));                            

                        else:
                            continue#print('missing segment...')
                    except:
                        continue #print('trip balked for mode ',mode)
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
                new_node = convertNode(stop,'gtfs','transit',NODES)
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









        drive_node = find_closest_node(node,mode,'drive',GRAPHS);
        walk_node = find_closest_node(node,mode,'walk',GRAPHS);
        transit_node = find_closest_node(node,mode,'transit',GRAPHS);
        ondemand_node = find_closest_node(node,mode,'ondemand',GRAPHS);
        gtfs_node = find_closest_node(node,mode,'gtfs',GRAPHS)
        
        if False: #OLD VERSION 
            if mode == 'gtfs':
                gtfs_node = node;
                drive_node = find_close_node_gtfs_to_graph(node,GRAPHS[mode],GRAPHS['drive']);
                walk_node = find_close_node_gtfs_to_graph(node,GRAPHS[mode],GRAPHS['walk']);
                transit_node = find_close_node_gtfs_to_graph(node,GRAPHS[mode],GRAPHS['transit']);
                ondemand_node = find_close_node_gtfs_to_graph(node,GRAPHS[mode],GRAPHS['ondemand']);

            else:
                # OLD VERSION
                drive_node = find_close_node(node,GRAPHS[mode],GRAPHS['drive']);
                walk_node = find_close_node(node,GRAPHS[mode],GRAPHS['walk']);
                transit_node = find_close_node(node,GRAPHS[mode],GRAPHS['transit']);
                ondemand_node = find_close_node(node,GRAPHS[mode],GRAPHS['ondemand']);
                gtfs_node = find_close_node_graph_to_gtfs(node,GRAPHS[mode],GRAPHS['gtfs'])



        node_tags = {'drive':[drive_node],
                     'walk':[walk_node],
                     'transit':[transit_node],
                     'ondemand':[ondemand_node],
                     'gtfs':[gtfs_node]}

        node_index = 'node'+str(len(NODES['all'].index));
        new_nodes = pd.DataFrame(node_tags,index=[node_index])
        NODES['all'] = pd.concat([NODES['all'],new_nodes]);

        ### ADDING TO SUB DFS ### ADDING TO SUB DFS ### ADDING TO SUB DFS ### ADDING TO SUB DFS ### ADDING TO SUB DFS 
        ### ADDING TO SUB DFS ### ADDING TO SUB DFS ### ADDING TO SUB DFS ### ADDING TO SUB DFS ### ADDING TO SUB DFS 

        mode = 'drive'; index_node = drive_node;
        node_tags = {'walk':[walk_node],'transit':[transit_node],'ondemand':[ondemand_node],'gtfs':[gtfs_node]}
        new_nodes = pd.DataFrame(node_tags,index=[index_node])
        NODES[mode] = pd.concat([NODES[mode],new_nodes]);

        mode = 'walk'; index_node = walk_node;
        node_tags = {'drive':[drive_node],'transit':[transit_node],'ondemand':[ondemand_node],'gtfs':[gtfs_node]}
        new_nodes = pd.DataFrame(node_tags,index=[index_node])
        NODES[mode] = pd.concat([NODES[mode],new_nodes]);

        mode = 'transit'; index_node = transit_node;
        node_tags = {'drive':[drive_node],'walk':[walk_node],'ondemand':[ondemand_node],'gtfs':[gtfs_node]}
        new_nodes = pd.DataFrame(node_tags,index=[index_node])
        NODES[mode] = pd.concat([NODES[mode],new_nodes]);

        mode = 'gtfs'; index_node = gtfs_node;
        node_tags = {'drive':[drive_node],'walk':[walk_node],'transit':[transit_node],'ondemand':[ondemand_node]}
        new_nodes = pd.DataFrame(node_tags,index=[index_node])
        NODES[mode] = pd.concat([NODES[mode],new_nodes]);

        mode = 'ondemand'; index_node = ondemand_node;
        node_tags = {'drive':[drive_node],'walk':[walk_node],'transit':[transit_node],'gtfs':[gtfs_node]}
        new_nodes = pd.DataFrame(node_tags,index=[index_node])
        NODES[mode] = pd.concat([NODES[mode],new_nodes]);

    # NODES['transit'] = drop_duplicates(NODES['transit']); #[~NODES['transit'].index.duplicated(keep='first')];
    # NODES['drive'] = drop_duplicates(NODES['drive']); #[~NODES['drive'].index.duplicated(keep='first')];
    # NODES['walk'] = drop_duplicates(NODES['walk']); #[~NODES['walk'].index.duplicated(keep='first')];
    # NODES['ondemand'] = drop_duplicates(NODES['ondemand']); #[~NODES['ondemand'].index.duplicated(keep='first')];
    # NODES['gtfs'] = drop_duplicates(NODES['gtfs']); #[~NODES['gtfs'].index.duplicated(keep='first')];
        return NODES


##### UPDATING NODES DATAFRAME
    
def updateNodesDF(NODES): ### ADDED TO CLASS
    NODES['transit'] = NODES['all'].copy();
    NODES['drive'] = NODES['all'].copy();
    NODES['walk'] = NODES['all'].copy();
    NODES['ondemand'] = NODES['all'].copy();
    NODES['gtfs'] = NODES['all'].copy();

    NODES['transit']['transit2'] = NODES['transit']['transit'].copy(); 
    NODES['drive']['drive2'] = NODES['drive']['drive'].copy();
    NODES['walk']['walk2'] = NODES['walk']['walk'].copy();
    NODES['ondemand']['ondemand2'] = NODES['ondemand']['ondemand'].copy(); 
    NODES['gtfs']['gtfs2'] = NODES['gtfs']['gtfs'].copy(); 

    NODES['transit'] = NODES['transit'].set_index('transit2')
    NODES['drive'] = NODES['drive'].set_index('drive2')
    NODES['walk'] = NODES['walk'].set_index('walk2')
    NODES['ondemand'] = NODES['ondemand'].set_index('ondemand2')
    NODES['gtfs'] = NODES['gtfs'].set_index('gtfs2')

    # NODES['transit'] = NODES['transit'][~NODES['transit'].index.duplicated(keep='first')];
    # NODES['drive'] = NODES['drive'][~NODES['drive'].index.duplicated(keep='first')];
    # NODES['walk'] = NODES['walk'][~NODES['walk'].index.duplicated(keep='first')];
    # NODES['ondemand'] = NODES['ondemand'][~NODES['ondemand'].index.duplicated(keep='first')];
    # NODES['gtfs'] = NODES['gtfs'][~NODES['gtfs'].index.duplicated(keep='first')];

    NODES['transit'] = drop_duplicates(NODES['transit']); #[~NODES['transit'].index.duplicated(keep='first')];
    NODES['drive'] = drop_duplicates(NODES['drive']); #[~NODES['drive'].index.duplicated(keep='first')];
    NODES['walk'] = drop_duplicates(NODES['walk']); #[~NODES['walk'].index.duplicated(keep='first')];
    NODES['ondemand'] = drop_duplicates(NODES['ondemand']); #[~NODES['ondemand'].index.duplicated(keep='first')];
    NODES['gtfs'] = drop_duplicates(NODES['gtfs']); #[~NODES['gtfs'].index.duplicated(keep='first')];


#     NODES['transit'] = NODES['transit'].drop_duplicates(keep='first');
#     NODES['drive'] = NODES['drive'].drop_duplicates(keep='first');
#     NODES['walk'] = NODES['walk'].drop_duplicates(keep='first');
#     NODES['ondemand'] = NODES['ondemand'].drop_duplicates(keep='first');
    return NODES
  
# def updateNodesDF(NODES):
#     NODES['transit'] = NODES['all'].copy();
#     NODES['drive'] = NODES['all'].copy();
#     NODES['walk'] = NODES['all'].copy();
#     NODES['ondemand'] = NODES['all'].copy();

#     NODES['transit']['transit2'] = NODES['transit']['transit'].copy(); 
#     NODES['drive']['drive2'] = NODES['drive']['drive'].copy();
#     NODES['walk']['walk2'] = NODES['walk']['walk'].copy();
#     NODES['ondemand']['ondemand2'] = NODES['ondemand']['ondemand'].copy(); 

#     NODES['transit'] = NODES['transit'].set_index('transit2')
#     NODES['drive'] = NODES['drive'].set_index('drive2')
#     NODES['walk'] = NODES['walk'].set_index('walk2')
#     NODES['ondemand'] = NODES['ondemand'].set_index('ondemand2')
    
#     NODES['transit'] = NODES['transit'][~NODES['transit'].index.duplicated(keep='first')];
#     NODES['drive'] = NODES['drive'][~NODES['drive'].index.duplicated(keep='first')];
#     NODES['walk'] = NODES['walk'][~NODES['walk'].index.duplicated(keep='first')];
#     NODES['ondemand'] = NODES['ondemand'][~NODES['ondemand'].index.duplicated(keep='first')];

# #     NODES['transit'] = NODES['transit'].drop_duplicates(keep='first');
# #     NODES['drive'] = NODES['drive'].drop_duplicates(keep='first');
# #     NODES['walk'] = NODES['walk'].drop_duplicates(keep='first');
# #     NODES['ondemand'] = NODES['ondemand'].drop_duplicates(keep='first');
#     return NODES

#### CONVERTING BETWEEN NODE TYPES

def convertNode(node,from_type,to_type,NODES,verbose=False): ### ADDED TO CLASS
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

    out = None;
    if from_type == 'all':
        out = NODES['all'][to_type][node]
    # elif from_type == to_type:
    #     out = node
    else:
        mode =  from_type; # == 'drive':
        if node in NODES[mode].index:
            #print(node)
            out = NODES[mode][to_type][node]
            if isinstance(out,int) or isinstance(out,float):
                out = out;
            elif not(isinstance(out,str)):
                out = list(out)
                out = out[0]
            if not(isinstance(out,str)):
                if np.isnan(out):
                    
                    updateNodesDF(NODES);
                    out = NODES[mode][to_type][node]                    
                    if isinstance(out,int) or isinstance(out,float):
                        out = out;
                    elif not(isinstance(out,str)):
                        out = list(out)
                        out = out[0]
                    if verbose == True:
                        print('nan found...after conversion...',out)         
                    
            #print(out)
        else: 
            updateNodesDF(NODES);
            out = list(NODES[mode][to_type][node])[0]
            # if np.isnan(node2):
            #     print(node2)


    # if from_type == 'transit':
    #     out = NODES['transit'][to_type][node]
    # if from_type == 'walk':
    #     out = NODES['walk'][to_type][node]
    # if from_type == 'ondemand':
    #     out = NODES['ondemand'][to_type][node]
    # if isinstance(out,list):
    #     out = out[0]
    # print(out)
    return out


def drop_duplicates(df): ### ADDED TO CLASS
    # print(df.astype(str).drop_duplicates().index)
    # asdf
    out = df.loc[df.astype(str).drop_duplicates().index]
    return out 
#convert hte df to str type, drop duplicates and then select the rows from original df.


############ -------------- SUPPORT FUNCTIONS ---------------- ##################
############ -------------- SUPPORT FUNCTIONS ---------------- ##################


def findNode(node,from_type,to_type,NODES): ### ADDED TO CLASS
    out = None;
    if from_type == 'all':
        out = NODES['all'][to_type][node]
    if from_type == 'drive':
        out = NODES['drive'][to_type][node]
    if from_type == 'transit':
        out = NODES['transit'][to_type][node]
    if from_type == 'walk':
        out = NODES['walk'][to_type][node]
    if from_type == 'ondemand':
        out = NODES['ondemand'][to_type][node]
    return out


def find_closest_node(node,mode1,mode2,GRAPHS): ### ADDED TO CLASS
    # node must be in GRAPHS[mode1]
    GRAPH1 = GRAPHS[mode1]; feed1 = GRAPH1
    GRAPH2 = GRAPHS[mode2]; feed2 = GRAPH2
    stop = node;
    if mode1 == 'gtfs':
        lat = list(feed1.stops[feed1.stops['stop_id']==stop].stop_lat)
        lon = list(feed1.stops[feed1.stops['stop_id']==stop].stop_lon)
        lat = lat[0]
        lon = lon[0]
    else:
        lon = GRAPH1.nodes[node]['x'];
        lat = GRAPH1.nodes[node]['y'];
    if mode2 == 'gtfs':
        close = np.abs(feed2.stops.stop_lat - lat) + np.abs(feed2.stops.stop_lon - lon);
        close = close==np.min(close)
        found_stop = feed2.stops.stop_id[close];
        found_node = list(found_stop)[0]
    else:
        found_node = ox.distance.nearest_nodes(GRAPH2, lon,lat); #ORIG_LOC[i][0], ORIG_LOC[i][1]);
        xx = GRAPH2.nodes[found_node]['x'];
        yy = GRAPH2.nodes[found_node]['y'];
        # if not(np.abs(xx-lon) + np.abs(yy-lat) <= 0.1):
        #     found_node = None;
    return found_node


def find_close_node(node,graph,find_in_graph): ### ADDED TO CLASS
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


def find_close_node_gtfs_to_graph(stop,feed,graph): ### ADDED TO CLASS

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

def find_close_node_graph_to_gtfs(node,graph,feed): ### ADDED TO CLASS
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


def add_base_edge_masses(GRAPHS,WORLD,WORLD0): ### ADDED TO CLASS
    modes = []
    for i,mode in enumerate(WORLD):
        if not(mode=='main' or mode=='transit' or mode=='gtfs'):
            WORLD[mode]['base_edge_masses'] = {};
            for e,edge in enumerate(GRAPHS[mode].edges):
                if edge in WORLD0[mode]['current_edge_masses']:
                    WORLD[mode]['base_edge_masses'][edge] = WORLD0[mode]['current_edge_masses'][edge];



def generate_bnds(center_point): ### ADDED TO CLASS
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




def SETUP_GRAPHS_CHATTANOOGA(center_point,radius,time_window,bnds=[]): ### ADDED TO CLASS
    print('loading feed...')
    # Loading GTFS feed
    feed = gtfs.Feed('data/gtfs/carta_gtfs.zip', time_windows=[0, 6, 10, 12, 16, 19, 24])
    
    # # szz = 1.; radius = szz*5000;
    # # start = 8*60*60; end = 9*60*60;
    start = time_window[0]; end = time_window[1];
    print('constructing transit graph from feed...')
    ### turning the gtfs into a graph...
    graph_bus = load_feed_as_graph(feed);#,start,end) ## NEW
    GRAPHS = {};
    print('loading drive graph...')
    ### Loading in the driving graph
    GRAPHS['drive'] = ox.graph_from_place('chattanooga',network_type='drive'); #ox.graph_from_polygon(graph_boundary,network_type='drive')
    print('loading walk graph...')
    #### loading in the walking graph
    GRAPHS['walk'] = ox.graph_from_place('chattanooga',network_type='walk'); #ox.graph_from_polygon(graph_boundary,network_type='walk')
    GRAPHS['transit'] = graph_bus;


    #### ondemand graph is the same as driving
    GRAPHS['ondemand'] = GRAPHS['drive'].copy()    
    GRAPHS['gtfs'] = feed;

    print('composing graphs...')
    graphs = [GRAPHS['drive'],GRAPHS['walk'],GRAPHS['ondemand'],GRAPHS['transit']]
    zzgraph = nx.compose_all(graphs);
    GRAPHS['all'] = zzgraph


    ##### CHOPPING THE GRAPHS DOWN TO FIT IN A BOUNDING BOX #############
    ##### CHOPPING THE GRAPHS DOWN TO FIT IN A BOUNDING BOX #############

    if len(bnds)>0:
        mode = 'drive'; GRAPHS[mode] = subgraph_bnd_box(GRAPHS[mode],bnds[0],bnds[1]);
        mode = 'ondemand'; GRAPHS[mode] = subgraph_bnd_box(GRAPHS[mode],bnds[0],bnds[1]);
        mode = 'walk'; GRAPHS[mode] = subgraph_bnd_box(GRAPHS[mode],bnds[0],bnds[1]);
        mode = 'transit'; GRAPHS[mode] = subgraph_bnd_box(GRAPHS[mode],bnds[0],bnds[1]);
        mode = 'all'; GRAPHS[mode] = subgraph_bnd_box(GRAPHS[mode],bnds[0],bnds[1]);    


    #### COMPUTING GRAPHS WITH EDGES REVERESED....
    print('computing reverse graphs...')
    RGRAPHS = {};
    for i,mode in enumerate(GRAPHS):
        print('...reversing',mode,'graph...')
        if not(mode=='gtfs'):
            RGRAPHS[mode] = GRAPHS[mode].reverse()  


    #### OLD... CHANGING THE BUS GRAPH SOME....
    print('connecting close bus stops...')
    graph_bus_wt = GRAPHS['transit'].copy();
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
    GRAPHS['bus_graph_wt'] = graph_bus_wt
    GRAPHS['transit']= graph_bus_wt.copy()
    return {'GRAPHS':GRAPHS,'RGRAPHS':RGRAPHS,'feed':feed}


def generate_polygons(vers,center_point=[],path=''): ### ADDED TO CLASS
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




def generate_segtypes(vers): # reg1,reg2,bg  ### ADDED TO CLASS
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


def SETUP_NODESDF_CHATTANOOGA(GRAPHS,NODES,NDF=[],
    ndfs_to_rerun = ['gtfs','transit','delivery1','delivery2','source','target']):  ### ADDED TO CLASS

    start_time = time.time()
    
    if len(NDF) == 0:
        NDF = createEmptyNodesDF();

    feed = GRAPHS['gtfs']
    

    if 'gtfs' in ndfs_to_rerun: 
        print('starting gtfs...')
        for i,stop in enumerate(list(feed.stops.stop_id)):
            if np.mod(i,100)==0: print(i)
            NDF = addNodeToDF(stop,'gtfs',GRAPHS,NDF)

    if 'transit' in ndfs_to_rerun:
        print('starting transit nodes...')
        for i,node in enumerate(list(GRAPHS['transit'].nodes())):
            if np.mod(i,100)==0: print(i)
            NDF = addNodeToDF(node,'transit',GRAPHS,NDF)
    end_time = time.time()
    print('time to create nodes...: ',end_time-start_time)
            # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}        

    start_time = time.time()


    if 'delivery1' in ndfs_to_rerun:
        print('starting delivery1 sources...')
        for i,node in enumerate(NODES['delivery1']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)

    if 'delivery2' in ndfs_to_rerun:        
        print('starting delivery2 sources...')
        for i,node in enumerate(NODES['delivery2']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)
            

    if 'source' in ndfs_to_rerun:    
        print('starting source nodes...')
        for i,node in enumerate(NODES['orig']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)

    if 'target' in ndfs_to_rerun:        
        print('starting target nodes...')
        for i,node in enumerate(NODES['dest']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)
        
    end_time = time.time()
    print('time to create nodes...: ',end_time-start_time)
        # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}    
    updateNodesDF(NDF);
    return NDF    



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
        
            
    bus_graph = GRAPHS['bus_graph_wt'];
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





def SETUP_NODESDF_CHATTANOOGA(GRAPHS,NODES,NDF=[],
    ndfs_to_rerun = ['gtfs','transit','delivery1','delivery2','source','target']):  #### ADDED TO CLASS 

    start_time = time.time()
    
    if len(NDF) == 0:
        NDF = createEmptyNodesDF();

    feed = GRAPHS['gtfs']
    

    if 'gtfs' in ndfs_to_rerun: 
        print('starting gtfs...')
        for i,stop in enumerate(list(feed.stops.stop_id)):
            if np.mod(i,100)==0: print(i)
            NDF = addNodeToDF(stop,'gtfs',GRAPHS,NDF)

    if 'transit' in ndfs_to_rerun:
        print('starting transit nodes...')
        for i,node in enumerate(list(GRAPHS['transit'].nodes())):
            if np.mod(i,100)==0: print(i)
            NDF = addNodeToDF(node,'transit',GRAPHS,NDF)
    end_time = time.time()
    print('time to create nodes...: ',end_time-start_time)
            # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}        

    start_time = time.time()


    if 'delivery1' in ndfs_to_rerun:
        print('starting delivery1 sources...')
        for i,node in enumerate(NODES['delivery1']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)

    if 'delivery2' in ndfs_to_rerun:        
        print('starting delivery2 sources...')
        for i,node in enumerate(NODES['delivery2']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)
            

    if 'source' in ndfs_to_rerun:    
        print('starting source nodes...')
        for i,node in enumerate(NODES['orig']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)

    if 'target' in ndfs_to_rerun:        
        print('starting target nodes...')
        for i,node in enumerate(NODES['dest']):
            if np.mod(i,200)==0: print(i)
            NDF = addNodeToDF(node,'drive',GRAPHS,NDF)
        
    end_time = time.time()
    print('time to create nodes...: ',end_time-start_time)
        # NODES[node_walk] = {'transit':node}; #,'walk':node_walk,'drive':node_drive,'ondemand':node_ondemand}    
    updateNodesDF(NDF);
    return NDF

def INITIALIZING_BUSSTOPCONVERSION_CHATTANOOGA(GRAPHS): # NEED TO PHASE OUT
    # find the closest node in each network to bus stops... 
    ### COMPUTING BUS STOP NODES IN OTHER GRAPHS ####
    feed = GRAPHS['gtfs'];
    recompute_close_nodes = True;
    if (recompute_close_nodes):
        BUS_STOP_NODES = {};
        BUS_STOP_NODES['bus'] = {}; 
        start_time = time.time()

        BUS_STOP_NODES['drive2'] = bus_stop_nodes(feed,GRAPHS['drive'])    
        BUS_STOP_NODES['drive'] = bus_stop_nodes_wgraph(GRAPHS['transit'],GRAPHS['drive'])        
        end_time = time.time()
        print('time to compute drive bus nodes...', end_time - start_time)
        print('')
        start_time = time.time()
        #BUS_STOP_NODES['ondemand'] = bus_stop_nodes(feed,GRAPHS['ondemand'])
        BUS_STOP_NODES['ondemand'] = bus_stop_nodes_wgraph(GRAPHS['transit'],GRAPHS['ondemand'])            

        end_time = time.time()
        print('time to compute ondemand bus nodes...', end_time - start_time)
        print('')    
        start_time = time.time()
        #BUS_STOP_NODES['walk'] = bus_stop_nodes(feed,GRAPHS['walk'])
        BUS_STOP_NODES['walk'] = bus_stop_nodes_wgraph(GRAPHS['transit'],GRAPHS['walk'])    
        end_time = time.time()
        print('time to compute walk bus nodes...', end_time - start_time)
        print('')    
        recompute_close_nodes = False;
    return BUS_STOP_NODES



    
# def LOAD_POPULATION_CHATTANOOGA():


def clipping_bus_graph(): # OUTDATED 
    if False: 
        radius2 = szz*0.0000125*radius;
        nodes_to_keep = [];
        old_graph = graph_bus_wt.copy();
        for e,edge in enumerate(old_graph.edges):
            node1,node2,_ = edge;
            if not((node2,node1,0) in old_graph.edges):
                old_graph.add_edge(node2,node1)
        for i,node in enumerate(old_graph.nodes):
            NODE = old_graph.nodes[node];
            lon = NODE['x']; lat = NODE['y'];
            centerx = center_point[1]; centery = center_point[0];
            dist = np.sqrt((centerx-lon)*(centerx-lon) + (centery-lat)*(centery-lat))
            if dist < radius2:
                nodes_to_keep.append(node);
        #         print(center_point)
        #         print(lat)
        graph_new = old_graph.subgraph(nodes_to_keep)
        # graph_new = nx.compose(GRAPHS['drive'],graph_new)
        components = list(nx.strongly_connected_components(graph_new))
        # print(len(list(asdf)))
        large_components = [];
        for i,component in enumerate(components):
            #if len(component)>1:
            large_components.append(component)

        #graph_new = nx.compose(GRAPHS['drive'],graph_new)        
        sub_graphs = [];
        sampled_nodes = [];
        for i,component in enumerate(large_components):
            sub_graphs.append(graph_new.subgraph(component))
            sampled_nodes.append(sample(sub_graphs[-1].nodes,1)[0])
        graph_new = nx.compose_all(sub_graphs);

        # for i,node1 in enumerate(sampled_nodes):
        #     for j,node2 in enumerate(sampled_nodes):
        #         if j>i:
        #             graph_new.add_edge(node1,node2)
        #             graph_new.add_edge(node2,node1)
        for i,node1 in enumerate(sampled_nodes):
            if i< len(sampled_nodes)-1:
                node2 = sampled_nodes[i+1]
                graph_new.add_edge(node1,node2)
                graph_new.add_edge(node2,node1)            
        GRAPHS['transit'] = graph_new.copy()



########## ---------------- LOADING NETWORKS ---------------- ################
########## ---------------- LOADING NETWORKS ---------------- ################

########## GRAPHS....
def load_feed_as_graph(feed):  ###### ADDED TO CLASS 
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

    return graph

    # graph.graph ={'created_date': '2023-09-07 17:19:29','created_with': 'OSMnx 1.6.0','crs': 'epsg:4326','simplified': True}
    # FOR REFERENCE
    # {'osmid': 19496019, 'name': 'Benton Avenue',
    # 'highway': 'unclassified', 'oneway': False, 'reversed': True,
    # 'length': 377.384,
    # 'geometry': <LINESTRING (-85.21 35.083, -85.211 35.083, -85.211 35.083, -85.212 35.083, ...>}


########## ---------------- LOADING POPULATIONS ---------------- ################
########## ---------------- LOADING POPULATIONS ---------------- ################

########## GTFS FEEDS....




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





def runODdata():
    #### FROM RISHAV...
    lodes_od=pd.read_parquet('oddata/lodes_od_based_on_job.parquet')
    lodes_od['go_time']=pd.to_datetime(lodes_od.go_time_str)
    lodes_od['return_time']=pd.to_datetime(lodes_od.return_time_str)
    lodes_od['home_geo']=lodes_od.apply(lambda x: shp.geometry.Point(x.home_loc_lon,x.home_loc_lat),axis=1)
    lodes_od['work_geo']=lodes_od.apply(lambda x: shp.geometry.Point(x.work_loc_lon,x.work_loc_lat),axis=1)
    lodes_od=pd.read_parquet('oddata/lodes_od_based_on_job.parquet')
    lodes_od['go_time']=pd.to_datetime(lodes_od.go_time_str)
    lodes_od['return_time']=pd.to_datetime(lodes_od.return_time_str)
    lodes_od['home_geo']=lodes_od.apply(lambda x: shp.geometry.Point(x.home_loc_lon,x.home_loc_lat),axis=1)
    lodes_od['work_geo']=lodes_od.apply(lambda x: shp.geometry.Point(x.work_loc_lon,x.work_loc_lat),axis=1)
    lodes_od=lodes_od.drop(columns=['home_loc_lon','home_loc_lat','work_loc_lon','work_loc_lat','go_time_str','return_time_str'])
    lodes_od_morning=lodes_od.rename(columns={'h_geocode':'from_cbg','home_geo':'from_geo','w_geocode':'to_cbg','work_geo':'to_geo'}).drop(columns=['return_time'])[['from_cbg', 'to_cbg', 'total_jobs', 'go_time', 'from_geo', 'to_geo']]
    lodes_od_evening=lodes_od.rename(columns={'h_geocode':'to_cbg','home_geo':'to_geo','w_geocode':'from_cbg','work_geo':'from_geo'})\
                        .drop(columns=['go_time']).rename(columns={'return_time':'go_time'})[['from_cbg', 'to_cbg', 'total_jobs', 'go_time', 'from_geo', 'to_geo']]
    lodes_od_trips=pd.concat([lodes_od_morning,lodes_od_evening]).drop(columns=['total_jobs'])
    lodes_od_trips['go_hour']=lodes_od_trips.go_time.dt.hour

    # fig=px.histogram(lodes_od_trips,x='go_hour')
    # fig.show()
    cbg=gpd.read_file('oddata/Tennessee Census Block Groups/tl_2020_47_bg.shp')
    lodes_od_trips['from_cbg']=lodes_od_trips.from_cbg.astype('str')
    lodes_od_trips_from=lodes_od_trips.merge(cbg,left_on='from_cbg',right_on='GEOID',how='left')
    lodes_od_trips_group=lodes_od_trips_from.groupby(['from_cbg'],as_index=False).agg({'NAMELSAD':'first','go_time' : 'count', 'from_cbg' : 'first','geometry':'first'})\
                    .rename(columns={'go_time':'count','NAMELSAD':'name'})
    lodes_od_trips_group=gpd.GeoDataFrame(lodes_od_trips_group,geometry='geometry',crs=cbg.crs)
    # fig = px.choropleth(lodes_od_trips_group,
    #                    geojson=lodes_od_trips_group.geometry,
    #                    locations=lodes_od_trips_group.index,
    #                    color="count",
    #                    projection="mercator",title="from block group")
    # fig.update_geos(fitbounds="locations", visible=False)
    # fig.show()
    lodes_od_trips['to_cbg']=lodes_od_trips.to_cbg.astype('str')
    lodes_od_trips_to=lodes_od_trips.merge(cbg,left_on='to_cbg',right_on='GEOID',how='left')
    lodes_od_trips_group=lodes_od_trips_to.groupby(['to_cbg'],as_index=False).agg({'NAMELSAD':'first','go_time' : 'count', 'from_cbg' : 'first','geometry':'first'})\
                    .rename(columns={'go_time':'count','NAMELSAD':'name'})
    lodes_od_trips_group=gpd.GeoDataFrame(lodes_od_trips_group,geometry='geometry',crs=cbg.crs)
    fig = px.choropleth(lodes_od_trips_group,
                       geojson=lodes_od_trips_group.geometry,
                       locations=lodes_od_trips_group.index,
                       color="count",
                       projection="mercator",title="To Block Group",height =800,width=800)

    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()
    lodes_od_trips_group.head()


    

###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 
###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS ###### TRIPS 



####### ===================== TRIP COMPUTATION ======================== ################
####### ===================== TRIP COMPUTATION ======================== ################
####### ===================== TRIP COMPUTATION ======================== ################



def removeMassFromEdges(mode,WORLD,GRAPHS): ###### ADDED TO CLASS 
    NETWORK = WORLD[mode]
    if mode=='gtfs':
        GRAPH = GRAPHS['transit'];
    else: 
        GRAPH = GRAPHS[mode];
    # edges = list(NETWORK['current_edge_masses']);
    # NETWORK['current_edge_costs'] = {};
    NETWORK['current_edge_masses'] = {};    
    for e,edge in enumerate(GRAPH.edges):
        # NETWORK['current_edge_costs'][edge] = 0;
        NETWORK['current_edge_masses'][edge] = 0;        

def addTripMassToEdges(trip,NETWORK): ###### ADDED TO CLASS 
    if trip in NETWORK['trips']:
        TRIP = NETWORK['trips'][trip]
        mass = 1;
        path = TRIP['path']
        for j,node in enumerate(path):
            if j < len(path)-1:
                edge = (path[j],path[j+1],0)
                edge_mass = NETWORK['edge_masses'][edge][-1] + mass;
                edge_cost = 1.*edge_mass + 1.;
                NETWORK['edge_masses'][edge][-1] = edge_mass;
                NETWORK['edge_costs'][edge][-1] = edge_cost;
                current_mass = NETWORK['current_edge_masses'][edge];
                NETWORK['current_edge_masses'][edge] = edge_cost;            
                NETWORK['current_edge_costs'][edge] = edge_cost;            

def makeTrip_OLD(modes,nodes,node_types,NODES,deliveries = []): ###### ADDED TO CLASS 
    trip = {};
    segs = [];
    deliv_counter = 0;
    for i,mode in enumerate(modes):
        segs.append({});
        segs[-1]['mode'] = mode;
        segs[-1]['start_nodes'] = []; #nodes[i]
        segs[-1]['end_nodes'] = []; #nodes[i+1]
        segs[-1]['path'] = [];
        for j,node in enumerate(nodes[i]):
            node2 = convertNode(node,node_types[i],mode,NODES)
            segs[-1]['start_nodes'].append(node2)
        for j,node in enumerate(nodes[i+1]):
            node2 = convertNode(node,node_types[i+1],mode,NODES)
            segs[-1]['end_nodes'].append(convertNode(node,node_types[i+1],mode,NODES))
                
#         segs[-1]['start_types'] = node_types[i]
#         segs[-1]['end_types'] = node_types[i+1]
        if mode == 'ondemand':
            segs[-1]['delivery'] = deliveries[deliv_counter]
            deliv_counter = deliv_counter + 1;
            
    trip = {}
    trip['current'] = {};
    trip['structure'] = segs; 
    trip['current']['cost'] = 100000000;
    trip['current']['traj'] = []; 
    return trip


def makeTrip(modes,nodes,NODES,delivery_grps=[],deliveries=[]): #,node_types,NODES,deliveries = []): ###### ADDED TO CLASS 
    
    trip = {};
    segs = [];

    deliv_counter = 0;
    for i,mode in enumerate(modes):
        segs.append({});
        segs[-1]['mode'] = mode;
        segs[-1]['start_nodes'] = []; #nodes[i]
        segs[-1]['end_nodes'] = []; #nodes[i+1]
        segs[-1]['path'] = [];
        
        if len(delivery_grps)>0:
            if mode == 'ondemand': segs[-1]['delivery_group'] = delivery_grps[i];
            else: segs[-1]['delivery_group'] = None;

        for j,node in enumerate(nodes[i]['nodes']):
            node2 = convertNode(node,nodes[i]['type'],mode,NODES)

            segs[-1]['start_nodes'].append(node2)
        for j,node in enumerate(nodes[i+1]['nodes']):
            node2 = convertNode(node,nodes[i+1]['type'],mode,NODES)

            segs[-1]['end_nodes'].append(node2);
                
#         segs[-1]['start_types'] = node_types[i]
#         segs[-1]['end_types'] = node_types[i+1]

        if mode == 'ondemand':
            segs[-1]['delivery'] = deliveries[deliv_counter]
            deliv_counter = deliv_counter + 1;
            
    trip = {}
    trip['current'] = {};
    trip['structure'] = segs; 
    trip['current']['cost'] = 100000000;
    trip['current']['traj'] = [];
    trip['active'] = False;
    return trip


def addDelivSeg(seg_details,mode,GRAPHS,DELIVERIES,WORLD,group='group0',mass=1,track=False,PERSON={}): ###### ADDED TO CLASS 

    source = seg_details[0]
    target = seg_details[1]

    start1 = WORLD['main']['start_time']; #seg_details[2]
    end1 = WORLD['main']['end_time']; #seg_details[4]
    

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

    if not(trip in WORLD[mode]['trips'].keys()):

        # print('adding delivery segment...')

        WORLD[mode]['trips'][trip] = {};
        WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
        #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
        WORLD[mode]['trips'][trip]['path'] = [];
        WORLD[mode]['trips'][trip]['delivery_history'] = [];
        WORLD[mode]['trips'][trip]['mass'] = 0;
        WORLD[mode]['trips'][trip]['delivery'] = None; #delivery;

        ##### NEW
        WORLD[mode]['trips'][trip]['typ'] = None;  #shuttle or direct

        if len(PERSON)>0:
            if 'group_options' in PERSON: WORLD[mode]['trips'][trip]['group_options'] = PERSON['group_options'];

        WORLD[mode]['trips'][trip]['costs']['current_time'] = distance;
        WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
        WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
        WORLD[mode]['trips'][trip]['costs']['current_switches'] = 1; 
        WORLD[mode]['trips'][trip]['current_path'] = None; #path;        

        # WORLD[mode]['trips'][trip][''] asdfasdf

        

        WORLD[mode]['trips'][trip]['am'] = int(1);
        WORLD[mode]['trips'][trip]['wc'] = int(0);


        WORLD[mode]['trips'][trip]['group'] = group;

        booking_id = int(len(WORLD[mode]['booking_ids']));
        WORLD[mode]['trips'][trip]['booking_id'] = booking_id;
        WORLD[mode]['booking_ids'].append(booking_id);


        for _,group in enumerate(DELIVERIES['groups']):
            DELIVERY = DELIVERIES['groups'][group];
            time_matrix = DELIVERY['time_matrix'];
            shp = np.shape(time_matrix)
            time_matrix = np.block([[time_matrix,np.zeros([shp[0],2])],[np.zeros([2,shp[1]]),np.zeros([2,2]) ]]);
            DELIVERY['time_matrix'] = time_matrix;

            WORLD[mode]['trips'][trip]['pickup_node_id'] = shp[0];
            WORLD[mode]['trips'][trip]['dropoff_node_id'] = shp[0]+1;


        try: 
            ##### NEEDS TO BE UPDATED 
            dist = nx.shortest_path_length(GRAPHS['drive'], source=trip[0], target=trip[1], weight='c');
            maxtriptime = dist*4;
            pickup_start = np.random.uniform(low=start1, high=end1-maxtriptime, size=1)[0];
            pickup_end = pickup_start + 60*20;
            dropoff_start = pickup_start;
            dropoff_end = pickup_end + maxtriptime;

            WORLD[mode]['trips'][trip]['pickup_time_window_start'] = int(pickup_start);
            WORLD[mode]['trips'][trip]['pickup_time_window_end'] = int(pickup_end);
            WORLD[mode]['trips'][trip]['dropoff_time_window_start'] = int(dropoff_start);
            WORLD[mode]['trips'][trip]['dropoff_time_window_end'] = int(dropoff_end);

        except:
            print('balked on adding deliv seg...')
            WORLD[mode]['trips'][trip]['pickup_time_window_start'] = int(start1);
            WORLD[mode]['trips'][trip]['pickup_time_window_end'] = int(end1);
            WORLD[mode]['trips'][trip]['dropoff_time_window_start'] = int(start1);
            WORLD[mode]['trips'][trip]['dropoff_time_window_end'] = int(end1);






        
        # WORLD[mode]['active_trips'].append(trip)
    # print(np.shape(WORLD[mode]['time_matrix']))



    
    # if track==True:
    #     WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
    #     WORLD[mode]['trips'][trip]['costs']['money'].append(1)
    #     WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
    #     WORLD[mode]['trips'][trip]['costs']['switches'].append(1)
    #     WORLD[mode]['trips'][trip]['path'].append(path);

    


    # if mass > 0:
    #     for j,node in enumerate(path):
    #         if j < len(path)-1:
    #             edge = (path[j],path[j+1],0)
    #             edge_mass = WORLD[mode]['edge_masses'][edge][-1] + mass;
    #             # edge_cost = 1.*edge_mass + 1.;
    #             WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
    #             # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
    #             # WORLD[mode]['current_edge_costs'][edge] = edge_cost;
    #             WORLD[mode]['current_edge_masses'][edge] = edge_mass;    


def planDijkstraSeg(seg_details,mode,GRAPHS,WORLD,mass=1,track=False):  ###### ADDED TO CLASS 
    
    
    source = seg_details[0];
    target = seg_details[1];

    trip = (source,target);
    GRAPH = GRAPHS[mode];
    # if mode == 'transit' and mass > 0:
    #     print(mode)
    try:
        temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c'); #Dumbfunctions
        distance = temp[0];
        # if mode == 'walk':
        #     distance = distance/1000;
        path = temp[1]; 
    except: 
        #print('no path found for bus trip ',trip,'...')
        distance = 1000000000000;
        path = [];

    if not(trip in WORLD[mode]['trips'].keys()):
        WORLD[mode]['trips'][trip] = {};
        WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
        #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
        WORLD[mode]['trips'][trip]['path'] = [];
        WORLD[mode]['trips'][trip]['mass'] = 0;

    WORLD[mode]['trips'][trip]['costs']['current_time'] = distance
    WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
    WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
    WORLD[mode]['trips'][trip]['costs']['current_switches'] = 1; 
    WORLD[mode]['trips'][trip]['current_path'] = path;
    WORLD[mode]['trips'][trip]['mass'] = 0;
    
    if track==True:
        WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
        WORLD[mode]['trips'][trip]['costs']['money'].append(1)
        WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
        WORLD[mode]['trips'][trip]['costs']['switches'].append(1)
        WORLD[mode]['trips'][trip]['path'].append(path);

    if mass > 0:
        for j,node in enumerate(path):
            if j < len(path)-1:
                edge = (path[j],path[j+1],0)
                edge_mass = WORLD[mode]['edge_masses'][edge][-1] + mass;
                # edge_cost = 1.*edge_mass + 1.;
                WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
                # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                # WORLD[mode]['current_edge_costs'][edge] = edge_cost;
                WORLD[mode]['current_edge_masses'][edge] = edge_mass;



def planGTFSSeg(trip0,mode,GRAPHS,WORLD,NODES,mass=1,track=True,verbose=False):  ###### ADDED TO CLASS 
    
    source = trip0[0];
    target = trip0[1];
    source = str(source);
    target = str(target);
    trip = (source,target);
    FEED = GRAPHS['gtfs']
    GRAPH = GRAPHS['transit'];

    PRECOMPUTE = WORLD['gtfs']['precompute']
    REACHED = PRECOMPUTE['reached']
    PREV_NODES = PRECOMPUTE['prev_nodes'];
    PREV_TRIPS = PRECOMPUTE['prev_trips'];

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

    if not(trip in WORLD[mode]['trips'].keys()):
        WORLD[mode]['trips'][trip] = {};
        WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
        #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
        WORLD[mode]['trips'][trip]['path'] = [];
        WORLD[mode]['trips'][trip]['mass'] = 0;

    WORLD[mode]['trips'][trip]['costs']['current_time'] = 1.*distance
    WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
    WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
    WORLD[mode]['trips'][trip]['costs']['current_switches'] = num_segments - 1; 
    WORLD[mode]['trips'][trip]['current_path'] = path;
    WORLD[mode]['trips'][trip]['mass'] = mass;
    
    if track==True:
        WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
        WORLD[mode]['trips'][trip]['costs']['money'].append(1)
        WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
        WORLD[mode]['trips'][trip]['costs']['switches'].append(num_segments-1)
        WORLD[mode]['trips'][trip]['path'].append(path);

    num_missing_edges = 0;
    if mass > 0:
        #print(path)
        for j,node in enumerate(path):
            if j < len(path)-1:
                edge = (path[j],path[j+1],0)
                if edge in WORLD[mode]['edge_masses']:
                    edge_mass = WORLD[mode]['edge_masses'][edge][-1] + mass;
                    # edge_cost = 1.*edge_mass + 1.;
                    WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
                    # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                    # WORLD[mode]['current_edge_costs'][edge] = edge_cost;    
                    WORLD[mode]['current_edge_masses'][edge] = edge_mass;
                else:
                    num_missing_edges = num_missing_edges + 1
                    # if np.mod(num_missing_edges,10)==0:
                    #     print(num_missing_edges,'th missing edge...')
    if verbose:
        if num_missing_edges > 1:
            print('# edges missing in gtfs segment...',num_missing_edges)



def querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group='group0'):   ###### ADDED TO CLASS 
    """
    description -- queries the cost of travel (for a person) along a particular segment in a given mode
    inputs -- 
           start: start node
           end: end node
           mode: mode of travel
           PERSON: person object
           NODES: node conversion dict
           GRAPHS: graphs object
           WORLD: network information
    returns --
           cost: subjective cost of the trip
           path: optimal path (list of nodes)
    """
    
    start = seg_details[0]
    end = seg_details[1];
    if not((start,end) in WORLD[mode]['trips']):#.keys()):  
        if mode == 'gtfs':
            planGTFSSeg(seg_details,mode,GRAPHS,WORLD,NODES,mass=0);
        elif mode == 'ondemand':
            addDelivSeg((start,end),mode,GRAPHS,DELIVERY,WORLD,group=group,mass=0,PERSON=PERSON);

        elif (mode == 'drive') or (mode == 'walk'):
            planDijkstraSeg((start,end),mode,GRAPHS,WORLD,mass=0);
    # print(PERSON['prefs'].keys())
    # print('got here for mode...',mode)

    tot_cost = 0;
    weights = [];
    costs = [];
    COSTS = {};
    for l,factor in enumerate(PERSON['prefs'][mode]):
        weights.append(PERSON['weights'][mode][factor])
        costs.append(WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor])
        COSTS['factor'] = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor]
        # cost = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor] # possibly change for delivery
        # diff = cost; #-PERSON['prefs'][mode][factor]
        # tot_cost = tot_cost + PERSON['weights'][mode][factor]*diff;
    weights = np.array(weights);
    costs = np.array(costs);
    if 'logit_version' in PERSON: logit_version = PERSON['logit_version']
    else: logit_version = 'weighted_sum'
    tot_cost = applyLogitChoice(weights,costs,ver=logit_version);
    try: 
        path = WORLD[mode]['trips'][(start,end)]['current_path']
    except:
        path = None;

    return tot_cost,path,COSTS


def applyLogitChoice(weights,values,offsets = [],ver='weighted_sum'):  ###### ADDED TO CLASS 
    cost = 100000000.;
    if len(offsets) == 0: offsets = np.zeros(len(weights));
    if ver == 'weighted_sum':
        cost = weights@values;
    return cost


def queryTrip(TRIP,PERSON,NODES,GRAPHS,DELIVERY,WORLD):  ###### ADDED TO CLASS 
    trip_cost = 0;
    costs_to_go = [];
    # COSTS_to_go = [];
    next_inds = [];
    next_nodes = [];

    for k,SEG in enumerate(TRIP['structure'][::-1]):
        end_nodes = SEG['end_nodes'];
        start_nodes = SEG['start_nodes'];
        group = SEG['delivery_group']
#         print(SEG['end_types'])
#         end_types = SEG['end_types'];
#         start_types = SEG['start_types'];
        mode = SEG['mode'];
        NETWORK = WORLD[mode];
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
                seg_details = (start_node,end_node,)
                cost,path,_ = querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group=group)
                possible_leg_costs[i] = cost;
                # possible_leg_COSTS[i] = COSTS.copy();
            ind = np.argmax(possible_leg_costs)
            next_inds[-1].append(ind)
            next_nodes[-1].append(end_nodes[ind])
            costs_to_go[-1].append(possible_leg_costs[ind]);
            # COSTS_to_go[-1].append(possible_leg_COSTS[ind]);
        

    
    
    init_ind = np.argmin(costs_to_go[-1]);
    init_cost = costs_to_go[-1][init_ind]
    init_node = TRIP['structure'][0]['start_nodes'][init_ind];

    costs_to_go = costs_to_go[::-1];
    next_inds = next_inds[::-1]
    next_nodes = next_nodes[::-1]
    

    inds = [init_ind];
    nodes = [init_node];
    
    prev_mode = TRIP['structure'][0]['mode'];

    for k,SEG in enumerate(TRIP['structure']):
        mode = SEG['mode']
#         SEG['opt_start'] = 
#         SEG['opt_end'] = next_nodes[k][inds[-1]]
        next_ind = next_inds[k][inds[-1]];
        next_node = next_nodes[k][inds[-1]];
        
        SEG['opt_start'] = convertNode(nodes[-1],prev_mode,mode,NODES);
        SEG['opt_end'] = next_node;    
        inds.append(next_ind)
        nodes.append(next_node);
        prev_mode = SEG['mode']
    
    trip_cost = costs_to_go[init_ind][0];
    TRIP['current']['cost'] = trip_cost

    TRIP['current']['traj'] = nodes;

    





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
            new_node = convertNode(stop,'gtfs','transit',NODES)
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





###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 





def createGroupsDF(polygons,types0 = []):  ### ADDED TO CLASS
    ngrps = len(polygons);
    groups = ['group'+str(i) for i in range(len(polygons))];
    depotlocs = [];
    for polygon in polygons:
        depotlocs.append((1./np.shape(polygon)[0])*np.sum(polygon,0));    
    if len(types0)==0: types = [['direct','shuttle'] for i in range(ngrps)]
    else: types = types0;
    GROUPS = pd.DataFrame({'group':groups,'depot_loc':depotlocs,'type':types,'polygon':polygons}); #,index=list(range(ngrps))
    # new_nodes = pd.DataFrame(node_tags,index=[index_node])
    # NODES[mode] = pd.concat([NODES[mode],new_nodes]);
    return GROUPS

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
    


#### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS 
#### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS 
#### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS #### PAYLOAD FUNCTIONS 





def travel_time_matrix(nodes,GRAPH):   ### ADDED TO CLASS
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
    return MAT;




def updateTravelTimeMatrix(nodes,ids,GRAPH,DELIVERY,WORLD,group=[0]):  ### ADDED TO CLASS
    MAT = travel_time_matrix(nodes,GRAPH)
    for i,id1 in enumerate(ids):
        for j,id2 in enumerate(ids):
            if len(group)>0:
                DELIVERY['time_matrix'][id1][id2] = MAT[i,j];
            else:
                WORLD['ondemand']['time_matrix'][id1][id2] = MAT[i,j];


def getBookingIds(PAY):    ### ADDED TO CLASS
    booking_ids = [];
    for i,request in enumerate(PAY['requests']):
        booking_ids.append(request['booking_id'])
    return booking_ids

def filterPayloads(PAY,ids_to_keep):  ### ADDED TO CLASS
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




def constructPayload(active_trips,DELIVERY,WORLD,GRAPHS,group=[0]): #PAY,ids_to_keep):   ### ADDED TO CLASS
    payload = {};
    payload['date'] = DELIVERY['date']
    payload['depot'] = DELIVERY['depot']
    payload['driver_runs'] = DELIVERY['driver_runs']

    if len(group)>0:
        MAT = DELIVERY['time_matrix'];
    else:
        MAT = WORLD['ondemand']['time_matrix'];

    GRAPH = GRAPHS['ondemand']
    
    nodes = {};
    inds = [0];
    curr_node_id = 1;

    mode = 'ondemand'
    requests = [];
    for i,trip in enumerate(active_trips):
        requests.append({})
        node1 = trip[0];
        node2 = trip[1];
        TRIP = WORLD[mode]['trips'][trip];

        requests[i]['booking_id'] = TRIP['booking_id']
        requests[i]['pickup_node_id'] = curr_node_id; curr_node_id = curr_node_id + 1; #TRIP['pickup_node_id']
        requests[i]['dropoff_node_id'] = curr_node_id; curr_node_id = curr_node_id + 1; #TRIP['dropoff_node_id']

        requests[i]['am'] = int(TRIP['am'])
        requests[i]['wc'] = int(TRIP['wc'])
        requests[i]['pickup_time_window_start'] = TRIP['pickup_time_window_start']
        requests[i]['pickup_time_window_end'] = TRIP['pickup_time_window_end']
        requests[i]['dropoff_time_window_start'] = TRIP['dropoff_time_window_start']
        requests[i]['dropoff_time_window_end'] = TRIP['dropoff_time_window_end']

        requests[i]['pickup_pt'] = {'lat': float(GRAPH.nodes[node1]['y']), 'lon': float(GRAPH.nodes[node1]['x'])}
        requests[i]['dropoff_pt'] = {'lat': float(GRAPH.nodes[node2]['y']), 'lon': float(GRAPH.nodes[node2]['x'])}

        inds.append(requests[i]['pickup_node_id'])
        inds.append(requests[i]['dropoff_node_id'])

        nodes[node1] = {'main':TRIP['pickup_node_id'],'curr':requests[i]['pickup_node_id']}
        nodes[node2] = {'main':TRIP['dropoff_node_id'],'curr':requests[i]['dropoff_node_id']}

        # 'pickup_pt': {'lat': 35.0296296, 'lon': -85.2301767},
        # 'dropoff_pt': {'lat': 35.0734152, 'lon': -85.1315328},
        # 'booking_id': 39851211,
        # 'pickup_node_id': 1,
        # 'dropoff_node_id': 2},

    inds = np.array(inds);
    MAT = np.array(MAT)
    MAT = MAT[np.ix_(inds,inds)]
    MAT2 = []
    for i,row in enumerate(MAT):
        MAT2.append(list(row.astype('int')))
    payload['time_matrix'] = MAT2
    payload['requests'] = requests
    return payload,nodes;



def payloadDF(PAY,GRAPHS,include_drive_nodes = False):  ### ADDED TO CLASS
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

def fakeManifest(payload): ### ADDED TO CLASS
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


def manifestDF(MAN,PDF):   ### ADDED TO CLASS
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


def routeFromStops(nodes,GRAPH): ### ADDED TO CLASS
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


def routesFromManifest(MDF,GRAPHS): ### ADDED TO CLASS
    max_id_num = int(np.max(list(MDF['run_id'])))+1
    PLANS = {};
    for i in range(max_id_num):
        ZZ = MDF[MDF['run_id']==i]
        ZZ = list(ZZ.sort_values(['scheduled_time'])['drive_node'])
        out = routeFromStops(ZZ,GRAPHS['drive']);
        PLANS[i] = out.copy();
    return PLANS

def singleRoutesFromManifest(MDF,GRAPHS): ### ADDED TO CLASS
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


def assignCostsFromManifest(active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS
    mode = 'ondemand';

    times_to_average = [0.];

    actual_costs = {};
    runs_info = {};
    run_ids = {};
    for i,trip in enumerate(active_trips):
        node1 = trip[0]
        node2 = trip[1]
        booking_id = WORLD[mode]['trips'][trip]['booking_id']
        main_node_id1 = nodes[trip[0]]['main']
        main_node_id2 = nodes[trip[1]]['main']
        curr_node_id1 = nodes[trip[0]]['curr']
        curr_node_id2 = nodes[trip[1]]['curr']


        try: 

            # mask1 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
            # mask2 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');
            mask1 = (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
            mask2 = (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');

            time1 = MDF[mask1]['scheduled_time'][0];
            time2 = MDF[mask2]['scheduled_time'][0];

            run_id = int(MDF[mask1]['run_id'][0]);


            time_diff = np.abs(time2-time1)

            actual_costs[trip] = time_diff;
            run_ids[trip] = run_id;

            runs_info[trip] = {}
            runs_info[trip]['runid'] = run_id;
            
            MDF2 = MDF[MDF['run_id'] == run_id]
            ind1 = np.where(MDF2['node_id'] == curr_node_id1)[0][0];
            ind2 = np.where(MDF2['node_id'] == curr_node_id2)[0][0];
            MDF2 = MDF2.iloc[ind1:ind2+1]
            runs_info[trip]['num_passengers'] = MDF2['booking_id'].nunique();

            if time_diff < 7200:
                times_to_average.append(time_diff);

        except: 
            actual_costs[trip] = 1000000000.;
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
    for i,trip in enumerate(WORLD[mode]['trips']):#active_trips):
        #if not(trip in active_trips):

        WORLD[mode]['trips'][trip]['costs']['current_time'] = est_ave_time;
        if trip in active_trips:
            WORLD[mode]['trips'][trip]['costs']['time'].append(actual_costs[trip]);#est_ave_time);
            WORLD[mode]['trips'][trip]['costs']['current_time'] = actual_costs[trip];#est_ave_time);
            try:
                WORLD[mode]['trips'][trip]['run_id'] = run_ids[trip] 
                WORLD[mode]['trips'][trip]['num_passengers'] = runs_info[trip]['num_passengers'];
            except:
                pass
        else:
            if len(WORLD[mode]['trips'][trip]['costs']['time'])==0: prev_time = est_ave_time;
            else: prev_time = WORLD[mode]['trips'][trip]['costs']['time'][-1]
            WORLD[mode]['trips'][trip]['costs']['time'].append(prev_time);
            WORLD[mode]['trips'][trip]['costs']['current_time'] = prev_time;


        factors = ['time','money','conven','switches']
        for j,factor in enumerate(factors):#WORLD[mode]['trips'][trip]['costs']):
            if not(factor == 'time'): 
                # WORLD[mode]['trips']['costs'][factor] = 0. 
                WORLD[mode]['trips'][trip]['costs'][factor].append(0.);
                WORLD[mode]['trips'][trip]['costs']['current_'+factor] = 0. 

    return average_time,times_to_average





    # if len(MDF)==0:
    #     for i,trip in enumerate(active_trips):
    #         node1 = trip[0];
    #         node2 = trip[1];


    #         k = 0; 
    #         found=False
    #         while (k<len(active_trips)) & (found==False):
    #             k = k+1
    #             leg = MAN[k]
    #             node_id = leg['node_id'];
    #             if node == nodes[node_id];

    #         #trip in enumerate(active_trips):
    #         TRIP = WORLD[mode]['trips'][trip]

    #     run_id = leg['run_id']
    #     book_id = leg['booking_id'];
    #     action = leg['action']
    #     scheduled_time = leg['scheduled_time'];
    #     node_id = leg['node_id']

    #     if action == 'pickup': drive_node = PDF['pickup_drive_node'][book_id]
    #     if action == 'dropoff': drive_node = PDF['dropoff_drive_node'][book_id]



    #         node1 = trip[0]
    #         node2 = trip[1]
    #         mask1 = (MDF['drive_node'] == trip[0]) & (MDF['action'] == 'pickup');
    #         mask2 = (MDF['drive_node'] == trip[1]) & (MDF['action'] == 'dropoff');
    #         time1 = MDF[mask1]['scheduled_time'];
    #         time2 = MDF[mask2]['scheduled_time'];
    #         WORLD[mode]['trips']['costs']['time'] = time2-time1;


    #     MAN[]


    # else:



########## ====================== ONDEMAND FUNCTIONS =================== ###################
########## ====================== ONDEMAND FUNCTIONS =================== ###################
########## ====================== ONDEMAND FUNCTIONS =================== ###################


def listDeliveryNodes(GRAPHS,WORLD):
    TRIPS = WORLD['ondemand']['trips'];
    GRAPH = GRAPHS['ondemand'];
    DNODES = {};
    DLOCS = {};
    for _,trip in enumerate(TRIPS):
        TRIP = TRIPS[trip];
        if 'delivery' in TRIP:
            delivery = TRIP['delivery']
            if not(delivery in DNODES):
                DNODES[delivery] = [];
                DLOCS[delivery] = [];
            node = trip[0]
            DNODES[delivery].append(node);
            DLOCS[delivery].append([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']])
            node = trip[1]
            DNODES[delivery].append(node);
            DLOCS[delivery].append([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']])
    return [DNODES,DLOCS]

def divideDelivSegs(trips,DELIVERY,GRAPHS,WORLD,maxTrips = 1000,vers='ver1'):
    GRAPH = GRAPHS['ondemand'];
    deliverys_unsorted = list(DELIVERY.keys());

    # reseting active trips...
    for i,deliv in enumerate(DELIVERY):
        DELIVERY[deliv]['active_trips'] = []
        #DELIV = DELIVERY[deliv]['active_trip_history'].append([])

    #################################################
    if vers=='ver1':
        for i,trip in enumerate(trips):
            start = trip[0];
            xloc = GRAPH.nodes[start]['x'];
            yloc = GRAPH.nodes[start]['y'];
            end = trip[1];
            # max_dist = 100000000000;
            # dists = [100000000000000];
            # delivs = [None];
            # current_deliv = None;
            dists = [];
            for j,deliv in enumerate(DELIVERY):
                DELIV = DELIVERY[deliv];
                loc0 = DELIV['loc'];
                dist = np.sqrt((xloc-loc0[0])*(xloc-loc0[0])+(yloc-loc0[1])*(yloc-loc0[1]));
                dists.append(dist);
            dists = np.array(dists);

            inds = np.argsort(dists).astype(int)
            delivs = [deliverys_unsorted[ind] for i,ind in enumerate(inds)]
            dists = dists[inds];
            j = 0; spot_found = False;
            while (j<(len(delivs)-1)) & (spot_found==False):
                deliv = delivs[j];
                DELIV = DELIVERY[deliv];
                if len(DELIV['active_trips'])<maxTrips:
                    DELIV['active_trips'].append(trip)
                    spot_found = True;
                j = j+1

    if vers=='ver2':
        wts = np.ones(len(pairs));
        kmeans_pairs(pairs,wts,GRAPH); #fixed_nodes = [],fixed_wts = [],centers0=[],maxiter=10);







    for i,deliv in enumerate(DELIVERY):
            DELIVERY[deliv]['active_trip_history'].append(DELIVERY[deliv]['active_trips'])                


    #################################################
    if False: #vers = 'ver2':
        # Lloyd-Walsh


        for i,trip in enumerate(trips):
            start = trip[0];
            xloc = GRAPH.nodes[start]['x'];
            yloc = GRAPH.nodes[start]['y'];
            end = trip[1];
            # max_dist = 100000000000;
            # dists = [100000000000000];
            # delivs = [None];
            # current_deliv = None;
            dists = [];
            for j,deliv in enumerate(DELIVERY):
                DELIV = DELIVERY[deliv];
                loc0 = DELIV['loc'];
                dist = np.sqrt((xloc-loc0[0])*(xloc-loc0[0])+(yloc-loc0[1])*(yloc-loc0[1]));
                dists.append(dist);
            dists = np.array(dists);

            inds = np.argsort(dists).astype(int)
            delivs = [deliverys_unsorted[ind] for i,ind in enumerate(inds)]
            dists = dists[inds];
            j = 0; spot_found = False;
            while (j<(len(delivs)-1)) & (spot_found==False):
                deliv = delivs[j];
                DELIV = DELIVERY[deliv];
                if len(DELIV['active_trips'])<maxTrips:
                    DELIV['active_trips'].append(trip)
                    spot_found = True;
                j = j+1;        



            trips_to_plan = WORLD['ondemand']['active_trips']
            DELIVERY0 = DELIVERY['direct'];
            maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));
            divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD,maxTrips);

        #     print(len(dists))
        #     for ll in range(len(dists)):
        #         if dist < dists[ll] :
        #             dists.insert(ll,dist);
        #             delivs.insert(ll,deliv);
        #     # if dist < max_dist:
        #     #     current_deliv = deliv;
        #     #     max_dist = dist;
        # dists = dists[:-1]
        # delivs = delivs[:-1]
        # # print(len(delivs))
        # for j,deliv in enumerate(delivs):
        #     DELIV = DELIVERY[deliv];
        #     if len(DELIV['active_trips'])<maxTrips:
        #         DELIV['active_trips'].append(trip)



def planDelivSegs(sources,targets,start,delivery,DELIVERY,GRAPHS,WORLD,maxSegs=100,track=False):
    
    GRAPH = GRAPHS['ondemand'];

    #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP
    #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP
    #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP
    #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP #### NEEDS TO BE UPDATED WITH VRP


    DELIV = DELIVERY[delivery]
    # print('PLANNING FOR: ',delivery)
    # print('sources...',sources)
    # print('targets...',targets)
    # print('start...',start)
    maxind = np.min([len(sources),len(targets),maxSegs]);    
    pickups= orderpickups2(GRAPH,sources[:maxind],targets[:maxind],start);
    plan = traveling_salesman(GRAPHS,pickups[:-1],pickups[-1]);


    # return {'route':route,'cost':cost,'current_len':current_len}
    # print(delivery,'has cost of',plan['cost'],'...')
    # print('...and route: ',plan['route'])
    mode = 'ondemand';
    for i,source in enumerate(sources):
        target = targets[i];
        trip = (source,target);
        mass = 1;
        distance = plan['cost'];
        path = plan['route'];
        DELIV['current_path'] = path;
        if not(trip in WORLD[mode]['trips'].keys()):
            WORLD[mode]['trips'][trip] = {};
            WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
            WORLD[mode]['trips'][trip]['path'] = [];
            WORLD[mode]['trips'][trip]['delivery_history'] = [];
            WORLD[mode]['trips'][trip]['mass'] = 0;
            WORLD[mode]['trips'][trip]['delivery'] = delivery;
            #WORLD[mode]['trips'][trip]['tsp'] = tsp.copy();

        WORLD[mode]['trips'][trip]['costs']['current_time'] = distance
        WORLD[mode]['trips'][trip]['costs']['current_money'] = 1;
        WORLD[mode]['trips'][trip]['costs']['current_conven'] = 1; 
        WORLD[mode]['trips'][trip]['costs']['current_switches'] = 1; 
        WORLD[mode]['trips'][trip]['current_path'] = path;
        WORLD[mode]['trips'][trip]['delivery'] = delivery;
        WORLD[mode]['trips'][trip]['mass'] = 0;
    
        if track==True:
            WORLD[mode]['trips'][trip]['costs']['time'].append(distance)
            WORLD[mode]['trips'][trip]['costs']['money'].append(1)
            WORLD[mode]['trips'][trip]['costs']['conven'].append(1)
            WORLD[mode]['trips'][trip]['costs']['switches'].append(1)
            WORLD[mode]['trips'][trip]['path'].append(path);
            #WORLD[mode]['trips'][trip]['delivery_history'].append(delivery);





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

def kmeans_wfixed(nodes,num,GRAPH,wts=[],centers0=[],locs0=[],node_locs=[],maxiter=10):
    print('asdf')

def kmeans_equal(nodes,num,GRAPH,wts=[],centers0=[],locs0=[],node_locs=[],maxiter=10):

    if len(wts)==0:
        wts = np.ones(len(nodes));
    if len(node_locs)==0:
        for i,node in enumerate(nodes):
            node_locs.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));

    all_nodes = list(GRAPH.nodes)
    if (len(centers0)==0) and (len(locs0)==0):
        centers0 = sample(all_nodes,num);
        for i,node in enumerate(centers0):
            locs0.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));        
    elif len(centers0) == 0:
        for i,loc in enumerate(locs0):
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);
            centers0.append(center);
    elif len(locs0) == 0:
        for i,node in enumerate(centers0):
            locs0.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));
    print('finished finding nodes...')

    CENTERS = [centers0];
    CENTROIDS = [locs0];
    PENALS = [np.ones(len(centers0))];

    #locs = node_locs;

    INDS = [];
    NODES = [];
    FIXEDS = [];
    FINDS = [];
    DISTS = [];
    NODES = [];
    COSTS = [];
    PATHS = [];
    GROUPS = [];
    WEIGHTS = [];
    MATRIX = [];
    LOCS = [];


    for k in range(maxiter):
        if np.mod(k,1)==0: print('iteration',k,'...')
    
        MATRIX.append([]);
        COSTS.append([]);
        INDS.append([]);         
        DISTS.append([]);
        PATHS.append([]);
        PENALS.append([]);
        GROUPS.append([]);
        WEIGHTS.append([]);        
        LOCS.append([]);
        centers = CENTERS[k]
        
        #### SHORTEST PATHS 

        costs = [];

        for j,center in enumerate(centers):
            #dist,path = nx.multi_source_dijkstra(GRAPH, nodes, target=center); #, weight='c');
            dist,path = nx.single_source_dijkstra(GRAPH, source=center); #, weight='c');
            DISTS[k].append(dist);
            PATHS[k].append(path);
            GROUPS[k].append([])
            # # PENALS[k].append(np.ones(len(centers)));
            COSTS[k].append([])
            WEIGHTS[k].append([])
            LOCS[k].append([]);

        #### ASSIGNING NODES #### 
        for i,node in enumerate(nodes):
            MATRIX[k].append([])
            INDS[k].append([])

            loc = node_locs[i]; wt = wts[i];
            ind = 0; 
            current_cost = 100000000000000.;
            ticker = 0
            COSTS[k].append(current_cost)
            for j,center in enumerate(centers):
                #penal = PENALS[k][j]
                # try:
                #     new_cost = DISTS[k][j][node];#+100.*penal
                #     if  new_cost < current_cost:
                #         ind = j; 
                #         current_cost = new_cost;
                #         #costs[-1] = current_cost;
                # except:
                #     new_cost = 10000000000000000.;
                new_cost = DISTS[k][j][node];                    
                MATRIX[k][i].append(new_cost)
                INDS[k][i].append(j)

            # inds = np.argsort(MATRIX[k][i])
            # MATRIX[k][i] = MATRIX[i][inds]
            # INDS[k][i] = INDS[i][inds]


            # # if ticker > 0:
            # #     print('total cost faults for',node,'is',ticker,'...')
            # center = centers[ind];
            # COSTS[k].append(current_cost)
            # INDS[k].append(ind)
            # LOCS[k].append(loc);
            # NODES[k].append(node);
        # print(len(locs))
        maxsize = len(nodes)/len(centers)            
        MATRIX[k] = np.array(MATRIX[k])
        TEMP0 = MATRIX[k].copy();
        temp = np.array(list(range(len(nodes))));
        iter = 0;
        while len(temp)>0:
            #if np.mod(iter,20)==0: print('iters:',iter)
            # if np.mod(len(temp),20)==0:
            #     print('nodes left to classify:',len(temp))
            TEMP = TEMP0[temp];
            ind = np.where(TEMP == np.min(np.min(TEMP)));
            nod = ind[0][0]; cent = ind[1][0];

            if True: #len(GROUPS[k][cent])<=maxsize:
                GROUPS[k][cent].append(nodes[nod])
                LOCS[k][cent].append(node_locs[nod]);
                WEIGHTS[k][cent].append(wts[nod]);
                temp = np.delete(temp,nod,0)
            else: 
                TEMP0[nod,cent] = 10000000000000.
            iter = iter + 1;

        CENTERS.append([]);
        CENTROIDS.append([]);
        for i,tag in enumerate(GROUPS[k]):
            group = GROUPS[k][i];
            weights = WEIGHTS[k][i]
            LOCS[k][i] = np.array(LOCS[k][i]);
            locs = LOCS[k][i]
            loc = np.zeros(2);
            scale = 0;
            for j,node in enumerate(group):
                loc = loc + weights[j]*locs[j];
                scale = scale + weights[j];
            if scale == 0: scale = 1;
            loc = (1./scale)*loc
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);
            PENALS[k+1].append(len(LOCS[k][i]));
            CENTROIDS[k+1].append(loc);
            CENTERS[k+1].append(center);
        CENTROIDS[k+1] = np.array(CENTROIDS[k+1]);            

    return {'centroids':CENTROIDS,'centers':CENTERS,'locs':LOCS}


def kmeans_weighted(nodes,wts,GRAPH,centers0=[],locs0=[],node_locs=[],
                    fixed_nodes = [], fixed_locs = [], fixed_wts = [],
                    maxiter=10):

    if len(centers0) == 0:
        for i,loc in enumerate(locs0):
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);
            centers0.append(center);
    elif len(locs0) == 0:
        for i,node in enumerate(centers0):
            locs0.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));
    if len(node_locs)==0:
        for i,node in enumerate(nodes):
            node_locs.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));


    for i,node in enumerate(fixed_nodes):
        fixed_locs.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]))

    if (len(fixed_nodes) > 0)  and (len(fixed_wts) == 0):
        fixed_wts = np.ones(len(fixed_nodes))


    print('finished finding nodes...')

    CENTERS = [centers0];
    CENTROIDS = [locs0];
    PENALS = [np.ones(len(centers0))];
    
    DISTS = [];
    COSTS = [];
    PATHS = [];
    GROUPS = [];
    WEIGHTS = [];
    LOCS = [];
    
    
    for k in range(maxiter):
        if np.mod(k,1)==0: print('iteration',k,'...')
        
        COSTS.append([]);        
        DISTS.append([]);
        PATHS.append([]);
        PENALS.append([]);
        GROUPS.append([]);
        WEIGHTS.append([]);        
        LOCS.append([]);
        centers = CENTERS[k]
        
        #### SHORTEST PATHS 
        for j,center in enumerate(centers):
            #dist,path = nx.multi_source_dijkstra(GRAPH, nodes, target=center); #, weight='c');
            dist,path = nx.single_source_dijkstra(GRAPH, source=center); #, weight='c');
            DISTS[k].append(dist);
            PATHS[k].append(path);
            GROUPS[k].append([])
            # PENALS[k].append(np.ones(len(centers)));
            COSTS[k].append([])
            WEIGHTS[k].append([])
            LOCS[k].append([])

        #### ASSIGNING NODES #### 
        for i,node in enumerate(nodes):
            loc = node_locs[i]; wt = wts[i];
            ind = 0; current_cost = 100000000000000.;
            ticker = 0
            for j,center in enumerate(centers):
                penal = PENALS[k][j]
                try:
                    new_cost = DISTS[k][j][node];#+100.*penal
                    if  new_cost < current_cost:
                        ind = j; 
                        current_cost = new_cost;
                except:
                    ticker = ticker + 1;
                    continue
            if ticker > 0:
                print('total cost faults for',node,'is',ticker,'...')
            center = centers[ind];
            COSTS[k][ind].append(current_cost)
            LOCS[k][ind].append(loc);
            GROUPS[k][ind].append(node);
            WEIGHTS[k][ind].append(wt);

        for i,node in enumerate(fixed_nodes):
            loc = fixed_locs[i]; wt = fixed_wts[i];
            LOCS[k][i].append(loc);
            GROUPS[k][i].append(node);
            WEIGHTS[k][i].append(wt);            

#         CENTROIDS = {};
#         CENTERS.append({});

        CENTERS.append([]);
        CENTROIDS.append([]);
        for i,tag in enumerate(GROUPS[k]):
            group = GROUPS[k][i];
            weights = WEIGHTS[k][i]
            LOCS[k][i] = np.array(LOCS[k][i]);
            locs = LOCS[k][i]
            loc = np.zeros(2);
            scale = 0;
            for j,node in enumerate(group):
                loc = loc + weights[j]*locs[j];
                scale = scale + weights[j];
            if scale == 0: scale = 1;
            loc = (1./scale)*loc
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);
            PENALS[k+1].append(len(LOCS[k][i]));
            CENTROIDS[k+1].append(loc);
            CENTERS[k+1].append(center);
        CENTROIDS[k+1] = np.array(CENTROIDS[k+1]);            

    return {'centroids':CENTROIDS,'centers':CENTERS,'locs':LOCS}

def kmeans_pairs(pairs,wts,GRAPH,
                fixed_nodes = [],fixed_wts = [],
                centers0=[],maxiter=10):


    nodes1 = []; nodes2 = [];
    locs1 = []; locs2 = [];
    for i,pair in enumerate(pairs):
        node1 = pair[0]; node2 = pair[1];
        nodes1.append(node1); nodes2.append(node2);
        locs1.append(np.array([GRAPH.nodes[node1]['x'],GRAPH.nodes[node1]['y']]));        
        locs2.append(np.array([GRAPH.nodes[node2]['x'],GRAPH.nodes[node2]['y']]));

    locs0 = []
    for i,node in enumerate(centers0):
        locs0.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));

    fixed_locs = [];
    for i,node in enumerate(fixed_nodes):
        fixed_locs.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]))

    if (len(fixed_nodes) > 0)  and (len(fixed_wts) == 0):
        fixed_wts = np.ones(len(fixed_nodes))


    print('finished finding nodes...')

    CENTERS = [centers0];
    CENTROIDS = [locs0];
    PENALS = [np.ones(len(centers0))];
    
    DISTS = [];
    COSTS = [];
    PATHS = [];
    WEIGHTS = [];
    GROUPS1 = [];
    GROUPS2 = [];
    LOCS1 = [];
    LOCS2 = [];
    LOCS = [];
    
    
    for k in range(maxiter):
        if np.mod(k,1)==0: print('iteration',k,'...')
        
        COSTS.append([]);       
        DISTS.append([]);
        PATHS.append([]);
        PENALS.append([]);
        GROUPS1.append([]);
        GROUPS2.append([]);
        WEIGHTS.append([]);        
        LOCS1.append([]);
        LOCS2.append([]);
        LOCS.append([]);
        centers = CENTERS[k]
        
        #### SHORTEST PATHS 
        for j,center in enumerate(centers):
            #dist,path = nx.multi_source_dijkstra(GRAPH, nodes, target=center); #, weight='c');
            dist,path = nx.single_source_dijkstra(GRAPH, source=center); #, weight='c');
            DISTS[k].append(dist);
            PATHS[k].append(path);
            GROUPS1[k].append([])
            GROUPS2[k].append([])
            LOCS1[k].append([])
            LOCS2[k].append([])
            LOCS[k].append([])
            # PENALS[k].append(np.ones(len(centers)));
            COSTS[k].append([])
            WEIGHTS[k].append([])
            

        #### ASSIGNING NODES #### 
        for i,node1 in enumerate(nodes1):
            node2 = nodes2[i]
            loc1 = locs1[i]; loc2 = locs2[i];
            wt = wts[i];
            ind = 0; current_cost = 100000000000000.;
            ticker = 0
            for j,center in enumerate(centers):
                # penal = PENALS[k][j]
                try:
                    new_cost = DISTS[k][j][node1]+DISTS[k][j][node2];#+100.*penal
                    if  new_cost < current_cost:
                        ind = j; 
                        current_cost = new_cost;
                except:
                    ticker = ticker + 1;
                    continue
            if ticker > 0:
                print('total cost faults for',node,'is',ticker,'...')
            center = centers[ind];
            COSTS[k][ind].append(current_cost)
            LOCS1[k][ind].append(loc1);
            LOCS2[k][ind].append(loc2);
            LOCS[k][ind].append(loc1);
            LOCS[k][ind].append(loc2);
            GROUPS1[k][ind].append(node1)
            GROUPS2[k][ind].append(node2)
            WEIGHTS[k][ind].append(wt);

        # for i,node in enumerate(fixed_nodes):
        #     loc = fixed_locs[i]; wt = fixed_wts[i];
        #     FLOCS[k][i].append(loc);
        #     LOCS2[k][i].append(loc);
        #     GROUPS1[k][i].append(node);
        #     GROUPS2[k][i].append(node);
        #     WEIGHTS[k][i].append(wt);            

#         CENTROIDS = {};
#         CENTERS.append({});

        CENTERS.append([]);
        CENTROIDS.append([]);
        for i,tag in enumerate(GROUPS1[k]):
            LOCS1[k][i] = np.array(LOCS1[k][i]);
            LOCS2[k][i] = np.array(LOCS2[k][i]);
            LOCS[k][i] = np.array(LOCS[k][i]);
                        
            weights = WEIGHTS[k][i]

            loc = np.zeros(2);
            scale = 0;
            group = GROUPS1[k][i];
            locs = LOCS1[k][i]
            for j,node in enumerate(group):
                loc = loc + weights[j]*locs[j];
                scale = scale + weights[j];
            group = GROUPS2[k][i];
            locs = LOCS2[k][i]
            for j,node in enumerate(group):
                loc = loc + weights[j]*locs[j];
                scale = scale + weights[j];
            if scale == 0: scale = 1;
            loc = (1./scale)*loc
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);

            # PENALS[k+1].append(len(LOCS[k][i]));
            CENTROIDS[k+1].append(loc);
            CENTERS[k+1].append(center);
        CENTROIDS[k+1] = np.array(CENTROIDS[k+1]);            

    return {'centroids':CENTROIDS,'centers':CENTERS,'locs':LOCS}

def lloyd_walsh(nodes,wts,GRAPH,centers0=[],locs0=[],node_locs=[],maxiter=10):

    if len(centers0) == 0:
        for i,loc in enumerate(locs0):
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);
            centers0.append(center);
    elif len(locs0) == 0:
        for i,node in enumerate(centers0):
            locs0.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));
    if len(node_locs)==0:
        for i,node in enumerate(nodes):
            node_locs.append(np.array([GRAPH.nodes[node]['x'],GRAPH.nodes[node]['y']]));

    print('finished finding nodes...')

    CENTERS = [centers0];
    CENTROIDS = [locs0];
    PENALS = [np.ones(len(centers0))];
    
    DISTS = [];
    COSTS = [];
    PATHS = [];
    GROUPS = [];
    WEIGHTS = [];
    LOCS = [];
    
    
    for k in range(maxiter):
        if np.mod(k,1)==0: print('iteration',k,'...')
        
        COSTS.append([]);        
        DISTS.append([]);
        PATHS.append([]);
        PENALS.append([]);
        GROUPS.append([]);
        WEIGHTS.append([]);        
        LOCS.append([]);
        centers = CENTERS[k]
        
        #### SHORTEST PATHS 
        for j,center in enumerate(centers):
            #dist,path = nx.multi_source_dijkstra(GRAPH, nodes, target=center); #, weight='c');
            dist,path = nx.single_source_dijkstra(GRAPH, source=center); #, weight='c');
            DISTS[k].append(dist);
            PATHS[k].append(path);
            GROUPS[k].append([])
            # PENALS[k].append(np.ones(len(centers)));
            COSTS[k].append([])
            WEIGHTS[k].append([])
            LOCS[k].append([])

        #### ASSIGNING NODES #### 
        for i,node in enumerate(nodes):
            loc = node_locs[i]; wt = wts[i];
            ind = 0; current_cost = 100000000000000.;
            ticker = 0
            for j,center in enumerate(centers):
                penal = PENALS[k][j]
                try:
                    new_cost = DISTS[k][j][node];#+100.*penal
                    if  new_cost < current_cost:
                        ind = j; 
                        current_cost = new_cost;
                except:
                    ticker = ticker + 1;
                    continue
            if ticker > 0:
                print('total cost faults for',node,'is',ticker,'...')
            center = centers[ind];
            COSTS[k][ind].append(current_cost)
            LOCS[k][ind].append(loc);
            GROUPS[k][ind].append(node);
            WEIGHTS[k][ind].append(wt);

#         CENTROIDS = {};
#         CENTERS.append({});

        CENTERS.append([]);
        CENTROIDS.append([]);
        for i,tag in enumerate(GROUPS[k]):
            group = GROUPS[k][i];
            weights = WEIGHTS[k][i]
            LOCS[k][i] = np.array(LOCS[k][i]);
            locs = LOCS[k][i]
            loc = np.zeros(2);
            scale = 0;
            for j,node in enumerate(group):
                loc = loc + weights[j]*locs[j];
                scale = scale + weights[j];
            if scale == 0: scale = 1;
            loc = (1./scale)*loc
            center = ox.distance.nearest_nodes(GRAPH, loc[0],loc[1]);
            PENALS[k+1].append(len(LOCS[k][i]));
            CENTROIDS[k+1].append(loc);
            CENTERS[k+1].append(center);
        CENTROIDS[k+1] = np.array(CENTROIDS[k+1]);            

    return {'centroids':CENTROIDS,'centers':CENTERS,'locs':LOCS}


##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 





# def orderpickups2(graph,sources,targets,start,typ='straight'):
#     print('starting ordering nodes...');
#     sources2 = sources.copy();
#     targets2 = targets.copy();
#     nodes_left = sources2.copy();
#     path = [start];
#     current = start;
    
#     while len(nodes_left)>0:
#         print(len(nodes_left))
#         next_stop = next_node2(graph,nodes_left,current,typ=typ);
#         path.append(next_stop['node']);
#         nodes_left.pop(next_stop['ind'])
#         current = path[-1]
#         if current in sources:
#             ind = sources.index(current)
#             nodes_left.append(targets[ind]);
#     print('finished sorting.')
#     return path


# # DONE
# def traveling_salesman(graph,pickups,sink):
#     route = [];
#     cost = 0;
#     costs = []
#     current_len = []
#     for i in range(len(pickups)):
#         current_len.append(len(route))
#         start_node = pickups[i]
#         if i == (len(pickups)-1):
#             end_node = sink;
#         else: 
#             end_node = pickups[i+1]
#         try:
#             path = nx.shortest_path(graph, source=start_node, target=end_node,weight = 'c'); #, weight=None);
#             cost = cost + nx.shortest_path_length(graph, source=start_node, target=end_node, weight='c')            
#         except:
#             path =[];
#             cost = cost + 0;
#         costs.append(cost)
#         route = route + path
#     current_len.append(len(route))
#     return {'route':route,'cost':cost,'current_len':current_len}







def next_node(graph,nodes,current,sink,typ = 'straight'):
    remaining_nodes = nodes;
    sink_loc = np.array([graph.nodes[sink]['x'],graph.nodes[sink]['y']])
    current_loc = np.array([graph.nodes[current]['x'],graph.nodes[current]['y']])
    dists = [];
    
    for i,node in enumerate(remaining_nodes):
        if typ=='straight':        
            x = graph.nodes[node]['x'];
            y = graph.nodes[node]['y'];
            loc = np.array([x,y]);
            dist_to_sink = mat.norm(loc-sink_loc);
            dist_to_current = mat.norm(loc-current_loc);
            if i == 0:
                dist_total = dist_to_sink; # + dist_to_current;
            else:
                dist_total = dist_to_current
            dists.append(dist_total)
                    
    dists = np.array(dists);
    ind = np.where(dists==np.min(dists))[0][0]
    return {'ind':ind,'node':nodes[ind]}

# DONE 
def order_pickups(graph,nodes,sink,typ='straight'):
    nodes_left = nodes.copy();
    path = [];
    current = sink;
    for i in range(len(nodes)):
        next_stop = next_node(graph,nodes_left,current,sink,typ=typ);
        path.append(next_stop['node']);
        nodes_left.pop(next_stop['ind'])
        current = path[-1]
    return path


def next_node2(graph,nodes,current,typ = 'straight'):
    remaining_nodes = nodes;
    # sink_loc = np.array([graph.nodes[sink]['x'],graph.nodes[sink]['y']])
    current_loc = np.array([graph.nodes[current]['x'],graph.nodes[current]['y']])
    dists = [];
    
    for i,node in enumerate(remaining_nodes):
        if typ=='straight':        
            x = graph.nodes[node]['x'];
            y = graph.nodes[node]['y'];
            loc = np.array([x,y]);
            # dist_to_sink = mat.norm(loc-sink_loc);
            dist_to_current = mat.norm(loc-current_loc);
            # if i == 0:
            #     dist_total = dist_to_sink; # + dist_to_current;
            # else:
            dist_total = dist_to_current
            dists.append(dist_total)                    
    dists = np.array(dists);
    ind = np.where(dists==np.min(dists))[0][0]
    return {'ind':ind,'node':nodes[ind]}
    

def orderpickups2(graph,sources,targets,start,typ='straight'):
    sources2 = sources.copy();
    targets2 = targets.copy();
    nodes_left = sources2.copy();
    path = [start];
    current = start;
    while len(nodes_left)>0:
        next_stop = next_node2(graph,nodes_left,current,typ=typ);
        nodes_left.pop(next_stop['ind'])
        path.append(next_stop['node']);
        current = path[-1]
        if current in sources2:
            ind = sources2.index(current)
            nodes_left.append(targets2[ind]);
            sources2.pop(ind)
    return path

# DONE
def traveling_salesman(GRAPHS,pickups,sink):
    graph = GRAPHS['drive'];
    route = [];
    cost = 0;
    costs = []
    current_len = []
    for i in range(len(pickups)):
        current_len.append(len(route))
        start_node = pickups[i]
        if i == (len(pickups)-1):
            end_node = sink;
        else: 
            end_node = pickups[i+1]
        try:
            path = nx.shortest_path(graph, source=start_node, target=end_node,weight = 'c'); #, weight=None);
            cost = cost + nx.shortest_path_length(graph, source=start_node, target=end_node, weight='c')            
        except:
            path =[];
            cost = cost + 0;
        costs.append(cost)
        route = route + path
    current_len.append(len(route))
    return {'route':route,'cost':cost,'current_len':current_len}




#### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
#### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
#### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 

def traveling_salesman_groups(graph,pickups,dropoffs):
    route = [];
    cost = 0;
    costs = []
    current_len = []
    for i in range(len(pickups)):
        current_len.append(len(route))
        start_node = pickups[i]
        if i == (len(pickups)-1):
            end_node = sink;
        else: 
            end_node = pickups[i+1]
        path = nx.shortest_path(graph, source=start_node, target=end_node,weight = 'c'); #, weight=None);
        cost = cost + nx.shortest_path_length(graph, source=start_node, target=end_node, weight='c')
        costs.append(cost)
        route = route + path
    current_len.append(len(route))
    return {'route':route,'cost':cost,'current_len':current_len}


def compute_masses(cost):
    masses = {};
    for o,opt in enumerate(costs):
        cost = costs[opt];
        masses[opt] = 0;
        for i,person in enumerate(Mlocs):
            if Mlocs[person][opt] >= cost:
                masses[opt] = masses[opt] + 1;
    return masses    
    
# def compute_masses(cost):
#     masses = {};
#     for o,opt in enumerate(costs):
#         cost = costs[opt];
#         masses[opt] = 0;
#         for i,person in enumerate(Mlocs):
#             if Mlocs[person][opt] >= cost:
#                 masses[opt] = masses[opt] + 1;
#     return masses    

def current_pickups(lam,pickup_nodes):
    current = [];
    for i,pick in enumerate(pickup_nodes):
        if MLOCS[pick] >= lam:
            current.append(pick)
    return current

# def current_pickups(lam,pickup_nodes):
#     current = [];
#     for i,pick in enumerate(pickup_nodes):
#         if MLOCS[pick] >= lam:
#             current.append(pick)
#     return current

def tsp_dual(grads):  
    lam = grads['lam']
    pickups = current_pickups(lam,all_pickup_nodes)
    nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
    #tsp = nx.approximation.traveling_salesman_problem(graph, nodes=curr,cycle=True, method=None)
    ordered_pickups = order_pickups(graph,pickups,sink);
    ordered_pickups = ordered_pickups[::-1]
    tsp = traveling_salesman(graph,ordered_pickups,sink)
    xvals = np.ones(ne);
    lamval = tsp['cost'];
    return {'x':xvals,'lam':lamval,'tsp':tsp,'pickups':pickups} #,'v':vvals}     


##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP 
##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP 
##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP ##### TSP 


def segment_pickups(PEOPLE,WORLD,DELIVERY): #graph,pickups,num):
    clients = [];
    sources = [];
    targets = [];
    source_locations = [];
    target_locations = [];
    pickups = WORLD['ondemand']['people']; #+ WORLD['ondemand+transit']['people']
    for i,person in enumerate(pickups):
        person_found = False
        PERSON = PEOPLE[person]
        if PERSON['choice'] == 'ondemand+transit+walk':
            opt = 'ondemand_initial'
            person_found = True;
        if PERSON['choice'] == 'walk+transit+ondemand':
            #opt = 'ondemand+transit
            opt = 'ondemand_final'
            person_found = True;
        if PERSON['choice'] == 'ondemand+transit+ondemand':
            #opt = 'ondemand+transit
            opt = 'ondemand'
            person_found = True;            
            
        if person_found:
            source_node = PERSON['sources'][opt];
            target_node = PERSON['targets'][opt];            
            source_loc = PERSON['source_locs'][opt];            
            target_loc = PERSON['target_locs'][opt];
            
            clients.append(person);
            sources.append(source_node)
            targets.append(target_node)
            source_locations.append(np.array(source_loc))
            target_locations.append(np.array(target_loc))
    
    source_locations = np.array(source_locations);
    target_locations = np.array(target_locations);
    
    num = len(delivery1_tags);
    
    centroids,_ = sp.cluster.vq.kmeans(source_locations, num, iter=20); #numiters)
    splits = [[] for i in range(num)];
    for i,location in enumerate(source_locations):
        diffs = centroids - location
        mags = np.array([mat.norm(diff) for j,diff in enumerate(diffs)]);
        splits[np.where(np.min(mags)==mags)[0][0]].append(i)


#     client_groups = []; 
#     source_groups = [];
#     target_groups = []; 
    
#     for i,split in enumerate(splits):
#         client_groups.append([clients[kk] for kk in split]);
#         source_groups.append([sources[kk] for kk in split]);
#         target_groups.append([targets[kk] for kk in split]);
        
    for i, delivery in enumerate(delivery1_tags):
        split = splits[i];
        client_group = [clients[kk] for kk in split];

        for j,client in enumerate(client_group):
            PEOPLE[client]['delivery_grp'] = delivery;
        DELIVERY[delivery]['people'] = client_group; 
        DELIVERY[delivery]['sources'] = [sources[kk] for kk in split];    
        DELIVERY[delivery]['targets'] = [targets[kk] for kk in split];        
        
    #return {'clients':client_groups,'sources':source_groups,'targets':target_groups}        


#     delivery_dists = [DELIVERY[delivery]['loc'] for j, delivery in enumerate(delivery_tags)];
#     delivery_dists = np.array(delivery_dists);
#     diffs = delivery_dists - source_loc;
#     dists = [mat.norm(diff) for j, diff in enumerate(diffs)];
#     ind = np.where(dists == np.min(dists))[0][0];
#     delivery = delivery_tags[ind];
#     #for i, delivery in enumerate(delivery_tags):   
#     PEOPLE[person]['delivery'] = delivery;
#     DELIVERY[delivery]['clients'].append(person)
#     DELIVERY[delivery]['sources'].append(PEOPLE[person]['sources']['ondemand'])    
#     DELIVERY[delivery]['targets'].append(PEOPLE[person]['targets']['ondemand'])  







###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 
###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE ###### CHOICE 


########### =================== UPDATE POPULATION CHOICES ================== #################
########### =================== UPDATE POPULATION CHOICES ================== #################
########### =================== UPDATE POPULATION CHOICES ================== #################


def update_choices(PEOPLE, DELIVERY, NODES, GRAPHS, WORLD, version=1,verbose=False,takeall=False):  ##### ADDED TO CLASS
    if verbose: print('updating choices')
    ## clear options
#     for o,opt in enumerate(WORLD):
#         WORLD[opt]['people'] = [];
        
    people_chose = {};
    for i,person in enumerate(PEOPLE):
        if np.mod(i,200)==0: print(person,'...')
        PERSON = PEOPLE[person];
        
        delivery_grp = PERSON['delivery_grp'];        
        delivery_grp_inital = PERSON['delivery_grp_initial'];
        delivery_grp_final = PERSON['delivery_grp_final'];        
        COMPARISON = [];

        trips_to_consider = PERSON['trips_to_consider'];

        for k,trip in enumerate(trips_to_consider):
            TRIP = PERSON['trips'][trip];
            try: 
                queryTrip(TRIP,PERSON,NODES,GRAPHS,DELIVERY,WORLD)
                COMPARISON.append(TRIP['current']['cost']);
                cost = TRIP['current']['cost'];
                # print('cost of',trip,'...',cost)
            except:
                COMPARISON.append(10000000000000000);

        ind = np.argmin(COMPARISON);
        trip_opt = trips_to_consider[ind];

        current_choice = ind;
        current_cost = COMPARISON[ind]

        PERSON['current_choice'] = current_choice;
        PERSON['current_cost'] = current_cost;

        PERSON['choice_traj'].append(current_choice);
        PERSON['cost_traj'].append(current_cost);
        # updating world choice...






        if takeall==False:
            tripstotake = [PERSON['trips'][trip_opt]];
        else:
            tripstotake = [PERSON['trips'][zz] for _,zz in enumerate(trips_to_consider)];


        for _,CHOSEN_TRIP in enumerate(tripstotake):
            num_segs = len(CHOSEN_TRIP['structure'])
            for k,SEG in enumerate(CHOSEN_TRIP['structure']):
                try:
                    mode = SEG['mode'];

                    # if mode == 'gtfs': print(SEG) 
            #             start = SEG['start_nodes'][SEG['opt_start']]
            #             end = SEG['end_nodes'][SEG['opt_end']
                    start = SEG['opt_start']
                    end = SEG['opt_end']
                    #print(WORLD[mode]['trips'][(start,end)])
                    #if not(mode in ['transit']):

                    # print(mode)
                    # print((start,end))
                    # if mode=='transit':
                    #     print(WORLD[mode]['trips'])
                    # try: 
                    trip = (start,end)

                    if (start,end) in WORLD[mode]['trips']:
                        WORLD[mode]['trips'][(start,end)]['mass'] = WORLD[mode]['trips'][(start,end)]['mass'] + PERSON['mass'];
                        WORLD[mode]['trips'][(start,end)]['active'] = True; 

                        if not((start,end) in WORLD[mode]['active_trips']):
                            WORLD[mode]['active_trips'].append((start,end));

                        if (num_segs == 3) & (mode == 'ondemand'):
                            WORLD[mode]['active_trips_shuttle'].append((start,end));
                        if (num_segs == 1) & (mode == 'ondemand'):
                            WORLD[mode]['active_trips_direct'].append((start,end));                            

                    else:
                        continue#print('missing segment...')
                except:
                    continue #print('trip balked for mode ',mode)
                #     continue

    

#         WORLD[current_choice]['people'].append(person);
        # WORLD[current_choice]['sources'].append(PERSON['sources'][current_choice])
        # WORLD[current_choice]['targets'].append(PERSON['targets'][current_choice])        
            
#         WORLD[current_choice]['sources'][''.append(node);
#         WORLD[current_choice]['targets'].append(PERSON['target']);    
            
########### WORLD[current_choice]['sources'][''.append(node); ####################
########### WORLD[current_choice]['targets'].append(PERSON['target']); ###########


def update_choices_OLD(PEOPLE, DELIVERY, WORLD, version=1):
    print('updating choices')
    ## clear options
    for o,opt in enumerate(WORLD):
        WORLD[opt]['people'] = [];
        
    people_chose = {};
    for i,person in enumerate(PEOPLE):
        PERSON = PEOPLE[person];
        
        delivery_grp = PERSON['delivery_grp'];        
        delivery_grp_inital = PERSON['delivery_grp_initial'];
        delivery_grp_final = PERSON['delivery_grp_final'];        
        COMPARISON = {};
        
        for o,opt in enumerate(PERSON['opts']):
            ## GO THROUGH OPTIONS: NEW more complicated ==> (opt1,opt2,opt3)ipy
#             DECISION[opt] = 0;
            COMPARISON[opt] = 0;
    
            if opt=='drive':
                for j,factor in enumerate(PERSON['prefs'][opt]):
                    mode = 'drive'
                    start_node = PERSON['nodes']['source'][mode]
                    end_node = PERSON['nodes']['target'][mode]
                    cost = WORLD[mode]['trips'][(start_node,end_node)]['costs'][factor][-1]
                    diff = cost-PERSON['prefs'][opt][factor]
                    COMPARISON[opt] = COMPARISON[opt] + PERSON['weights'][mode][factor]*diff
            elif opt=='ondemand':
                for j,factor in enumerate(PERSON['prefs'][opt]):
                    mode = 'ondemand';
                    start_node = PERSON['nodes']['source'][mode]
                    end_node = PERSON['nodes']['target'][mode]
                    # print(start_node)
                    # print(end_node)
                    # print(WORLD[mode]['trips'][(start_node,end_node)]['costs'])
                    cost = WORLD[mode]['trips'][(start_node,end_node)]['costs'][factor][-1]
                    diff = cost-PERSON['prefs'][opt][factor]
                    COMPARISON[opt] = COMPARISON[opt] + PERSON['weights'][mode][factor]*diff
            elif opt=='walk':
                for j,factor in enumerate(PERSON['prefs'][opt]):
                    mode = 'walk';
                    start_node = PERSON['nodes']['source'][mode]
                    end_node = PERSON['nodes']['target'][mode]
                    cost = WORLD[mode]['trips'][(start_node,end_node)]['costs'][factor][-1]
                    diff = cost-PERSON['prefs'][opt][factor]
                    COMPARISON[opt] = COMPARISON[opt] + PERSON['weights'][mode][factor]*diff
            else: 
                mode1 = PERSON['opts2'][o][0]
                mode2 = PERSON['opts2'][o][1]
                mode3 = PERSON['opts2'][o][2]

                for j,factor in enumerate(PERSON['prefs'][opt]):

                    if mode1=='walk': temp_tag1 = 'transit_walk1';
                    if mode1=='ondemand': temp_tag1 = 'transit_ondemand1';
                    if mode1=='drive': temp_tag1 = 'transit_drive1';
                    ######### 
                    start_node = PERSON['nodes']['source'][mode1]
                    end_node = PERSON['nodes'][temp_tag1][mode1]
                    cost = WORLD[mode1]['trips'][(start_node,end_node)]['costs'][factor][-1]
                    diff = cost-PERSON['prefs'][mode1][factor]
                    COMPARISON[opt] = COMPARISON[opt] + PERSON['weights'][mode1][factor]*diff
                    ######### 
                    if mode3=='walk': temp_tag2 = 'transit_walk2';
                    if mode3=='ondemand': temp_tag1 = 'transit_ondemand2';
                    if mode3=='drive': temp_tag1 = 'transit_drive2';
                    start_node = PERSON['nodes'][temp_tag1]['transit']
                    end_node = PERSON['nodes'][temp_tag2]['transit']
                    cost = WORLD['transit']['trips'][(start_node,end_node)]['costs'][factor][-1]
                    diff = cost-PERSON['prefs']['transit'][factor]
                    COMPARISON[opt] = COMPARISON[opt] + PERSON['weights']['transit'][factor]*diff
                    ######### 
                    start_node = PERSON['nodes'][temp_tag2][mode3]
                    end_node = PERSON['nodes']['target'][mode3]
                    cost = WORLD[mode3]['trips'][(start_node,end_node)]['costs'][factor][-1]
                    diff = cost-PERSON['prefs'][mode3][factor]
                    COMPARISON[opt] = COMPARISON[opt] + PERSON['weights'][mode3][factor]*diff

                    
        current_choice = None;
        current_cost = 0;
        for o,opt in enumerate(COMPARISON):
            if (COMPARISON[opt] < 0) & (COMPARISON[opt] < current_cost):
                current_choice = opt
                current_cost = COMPARISON[opt];
                
        PERSON['current_choice'] = current_choice;
        PERSON['current_cost'] = current_cost;
        PERSON['choice_traj'].append(current_choice);
        PERSON['cost_traj'].append(current_cost);
        # updating world choice...
        
        WORLD[current_choice]['people'].append(person);
        # WORLD[current_choice]['sources'].append(PERSON['sources'][current_choice])
        # WORLD[current_choice]['targets'].append(PERSON['targets'][current_choice])        
            
#         WORLD[current_choice]['sources'][''.append(node);
#         WORLD[current_choice]['targets'].append(PERSON['target']);    











### GENERATES MAIN WORLD OBJECT
def generateWorld(GRAPHS,params):  ##### ADDED TO CLASS 1
    modes = params['modes'];
    graphs = params['graphs'];
    people_tags = params['people_tags'];
    factors = params['factors'];    
    nodes = params['nodes'];
    WORLD = {};

    WORLD['main'] = {}
    WORLD['main']['iter'] = 1.;
    WORLD['main']['alpha'] = 1./WORLD['main']['iter'];

    WORLD['main']['start_time'] = 0;
    WORLD['main']['end_time'] = 3600*4.;    

    for m,mode in enumerate(modes):
        WORLD[mode] = {}
        WORLD[mode]['trips'] = {}
        WORLD[mode]['graph'] = graphs[m];
        WORLD[mode]['costs'] = {};
        WORLD[mode]['sources'] = [];
        WORLD[mode]['targets'] = [];
        WORLD[mode]['edge_masses'] = {};
        WORLD[mode]['edge_costs'] = {};
        WORLD[mode]['current_edge_masses'] = {};
        WORLD[mode]['current_edge_costs'] = {};

        WORLD[mode]['edge_cost_poly'] = {};

        if mode == 'ondemand':
            WORLD[mode]['current_edge_masses1'] = {};    
            WORLD[mode]['current_edge_masses2'] = {};            

        
        WORLD[mode]['people'] = people_tags.copy();
        WORLD[mode]['active_trips'] = [];

        if mode == 'ondemand':
            WORLD[mode]['active_trips'] = [];
            WORLD[mode]['active_trips_shuttle'] = [];
            WORLD[mode]['active_trips_direct'] = [];
            WORLD[mode]['booking_ids'] = [];
            WORLD[mode]['time_matrix'] = np.zeros([1,1]);
            WORLD[mode]['expected_cost'] = [0.];
            WORLD[mode]['current_expected_cost'] = 0.;
            WORLD[mode]['actual_average_cost'] = [];





            # WORLD[mode]['depot'] = np.zeros([1,1]);


        for j,node in enumerate(nodes):
            WORLD[mode]['costs'][node] = dict(zip(factors,np.ones(len(factors))));

        if graphs[m]=='gtfs':
            GRAPH = GRAPHS['transit']
            edges = GRAPH.edges;
        else:
            GRAPH = GRAPHS[graphs[m]]
            edges = GRAPH.edges;

        if (mode=='drive') or (mode=='walk'): #############  
            POLYS = createEdgeCosts(mode,GRAPHS) ##################
            WORLD[mode]['cost_fx'] = POLYS.copy();

        for j,edge in enumerate(edges):
            WORLD[mode]['edge_masses'][edge] = [0];
            WORLD[mode]['edge_costs'][edge]=[1];
            WORLD[mode]['current_edge_masses'][edge]=0;
            if (mode=='drive') & (mode=='walk'): #############  
                WORLD[mode]['edge_cost_poly'][edge]=POLYS[edge]; ##############
            if mode == 'ondemand':
                WORLD[mode]['current_edge_masses1'][edge]=0;
                WORLD[mode]['current_edge_masses2'][edge]=0;
            WORLD[mode]['current_edge_costs'][edge]=1;        

    WORLD['ondemand']['people'] = people_tags.copy();
     #people_tags.copy();
    # WORLD['ondemand+transit']['people'] = people_tags.copy();
    WORLD['ondemand']['trips'] = {};

    WORLD['gtfs']['precompute'] = {};
    WORLD['gtfs']['precompute']['reached'] = {};
    WORLD['gtfs']['precompute']['prev_nodes'] = {};
    WORLD['gtfs']['precompute']['prev_trips'] = {};
    return WORLD 



# graph = GRAPHS['drive']

#     npickups = 20;
#     mean = 500; stdev = 25;
#     Mlocs = np.random.normal(mean,stdev,npickups);

#     num_people = len(ORIG_LOC);
#     people_tags = [];
#     for i in range(len(ORIG_LOC)):
#         tag = 'person' + str(i);
#         people_tags.append('person'+str(i));    


#     # opts = ['drive','ondemand','walk+transit','ondemand+transit'];
#     opts = ['drive','ondemand',
#             'walk+transit+walk',
#             'ondemand+transit+walk',
#             'walk+transit+ondemand',        
#             'ondemand+transit+ondemand'];
#     orders = [['drive'],
#               ['ondemand'],
#               ['walk','transit','walk'],
#               ['ondemand','transit','walk'],
#               ['walk','transit','ondemand'],          
#               ['ondemand','transit','ondemand']];
              
#     modes = ['drive','ondemand','walk','transit','gtfs'];
#     graphs = ['drive','ondemand','walk','transit','gtfs'];
#     factors = ['time','money','conven','switches'];
#     means = [mean,mean,mean,mean,mean];
#     stdevs = [stdev,stdev,stdev,stdev,stdev];


#     print(modes)
#     STATS = {}
#     for m,mode in enumerate(modes):
#         STATS[mode] = {};
#         STATS[mode]['mean'] = dict(zip(factors,means));
#         STATS[mode]['stdev'] = dict(zip(factors,stdevs));

        
        
#     WORLD = {};
#     for m,mode in enumerate(modes):
#         WORLD[mode] = {}
#         WORLD[mode]['trips'] = {}
#         WORLD[mode]['graph'] = graphs[m];
#         WORLD[mode]['costs'] = {};
#         WORLD[mode]['sources'] = [];
#         WORLD[mode]['targets'] = [];
#         WORLD[mode]['edge_masses'] = {};
#         WORLD[mode]['edge_costs'] = {};
#         WORLD[mode]['current_edge_masses'] = {};    
#         if mode == 'ondemand':
#             WORLD[mode]['current_edge_masses1'] = {};    
#             WORLD[mode]['current_edge_masses2'] = {};            
        
#         WORLD[mode]['current_edge_costs'] = {};
#         WORLD[mode]['people'] = people_tags.copy();
#         WORLD[mode]['active_trips'] = [];
#         for j,node in enumerate(nodes):
#             WORLD[mode]['costs'][node] = dict(zip(factors,np.ones(len(factors))));
            
            
#         if graphs[m]=='gtfs':
#             GRAPH = GRAPHS['transit']
#             edges = GRAPH.edges;
#         else:
#             GRAPH = GRAPHS[graphs[m]]
#             edges = GRAPH.edges;
#         for j,edge in enumerate(edges):
#             WORLD[mode]['edge_masses'][edge] = [0];
#             WORLD[mode]['edge_costs'][edge]=[1];
#             WORLD[mode]['current_edge_masses'][edge]=0;
#             if mode == 'ondemand':
#                 WORLD[mode]['current_edge_masses1'][edge]=0;
#                 WORLD[mode]['current_edge_masses2'][edge]=0;
#             WORLD[mode]['current_edge_costs'][edge]=1;        
            
#     WORLD['ondemand']['people'] = people_tags.copy();
#      #people_tags.copy();
#     # WORLD['ondemand+transit']['people'] = people_tags.copy();
#     WORLD['ondemand']['trips'] = {};

#     WORLD['gtfs']['precompute'] = {};
#     WORLD['gtfs']['precompute']['reached'] = {};
#     WORLD['gtfs']['precompute']['prev_nodes'] = {};
#     WORLD['gtfs']['precompute']['prev_trips'] = {};

#     import pickle
#     import os.path

#     filename = 'gtfs_trips.obj'
#     if os.path.isfile(filename):
#         # open a file, where you stored the pickled data
#         file = open('gtfs_trips.obj', 'rb')
#         # dump information to that file
#         data = pickle.load(file)    
#         # close the file
#         file.close()
#         WORLD['gtfs']['precompute'] = {};
#         WORLD['gtfs']['precompute']['reached'] = data['REACHED_NODES'];
#         WORLD['gtfs']['precompute']['prev_nodes'] = data['PREV_NODES'];
#         WORLD['gtfs']['precompute']['prev_trips'] = data['PREV_TRIPS'];

#     maxMloc = np.max(Mlocs);
#     Mmass = np.ones(npickups);



def selectDeliveryGroup(trip,grpsDF,GRAPHS,typ='direct'):
    GRAPH = GRAPHS['drive'];
    node1 = trip[0]; node2 = trip[1];
    loc1 = np.array([GRAPH.nodes[node1]['x'],GRAPH.nodes[node1]['y']]);
    loc2 = np.array([GRAPH.nodes[node2]['x'],GRAPH.nodes[node2]['y']]);
    pt1_series = gpd.GeoSeries([Point(loc1)])
    pt2_series = gpd.GeoSeries([Point(loc2)])

    ###### types are in priority order 
    possible_group_inds = [];
    riders_to_drivers = [];
    for i in range(len(grpsDF)):
        ROW = grpsDF.iloc[i]
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
        group = grpsDF.iloc[group_ind]['group']
    else:
        group = []; group_ind = None
    return group,group_ind





def generateDeliveries(GRAPHS,WORLD,NDF,params,DELIVERYDF={}):   ##### ADDED TO CLASS 

    if 'direct_locs' in params: direct_locs = params['direct_locs'];
    else: direct_locs = [];
    if 'shuttle_locs' in params: shuttle_locs = params['shuttle_locs'];
    else: shuttle_locs = [];



    if 'driver_info' in params: driver_info = params['driver_info'];

    if 'center_point' in params: center_point = params['center_point'];
    else: center_point: center_point = (-85.3094,35.0458)
 
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


    ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS 
    ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS 
    ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS ##### NEW VERSION 2 W/GROUPS 

    # GROUPS = 

    DELIVERY['groups'] = {};


    if len(DELIVERYDF)>0:
        grpsDF = DELIVERYDF['grps']
        driversDF = DELIVERYDF['drivers']
        DELIVERY['grpsDF'] = grpsDF
        DELIVERY['driversDF'] = driversDF
        DELIVERY['grpsDF']['num_drivers'] = list(np.zeros(len(grpsDF)))


        num_groups = len(grpsDF);
        for k in range(num_groups):
            ROW = grpsDF.iloc[k];
            group = ROW['group']
            DELIVERY['groups'][group] = {};

            DELIVERY['groups'][group]['driver_runs'] = [];
            DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);

            if 'date' in ROW: DELIVERY['groups'][group]['date'] = ROW['date'];
            else: DELIVERY['groups'][group]['date'] = '2023-07-31'

            num_drivers = 8; 
            loc = ROW['depot_loc']
            DELIVERY['groups'][group]['depot'] = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
            DELIVERY['groups'][group]['depot_node'] = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

            DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);
            DELIVERY['groups'][group]['expected_cost'] = [0.];
            DELIVERY['groups'][group]['current_expected_cost'] = 0.;
            DELIVERY['groups'][group]['actual_average_cost'] = [];

            DRVDF = driversDF[group]
            if len(DRVDF) > 0:
                num_drivers = len(DRVDF);
                for i in range(num_drivers):
                    ROW2 = DRVDF.iloc[i];
                    driver_start_time = ROW2['start_time'];
                    driver_end_time = ROW2['end_time']; 
                    am_capacity = ROW2['am_capacity'];
                    wc_capacity = ROW2['wc_capacity'];

                    DELIVERY['groups'][group]['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
                        'am_capacity': am_capacity,'wc_capacity': wc_capacity})
            else: 
                for i in range(num_drivers):
                    driver_start_time = WORLD['main']['start_time'];
                    driver_end_time = WORLD['main']['end_time'];
                    am_capacity = 8
                    wc_capacity = 2
                    DELIVERY['groups'][group]['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
                        'am_capacity': am_capacity,'wc_capacity': wc_capacity})

            DELIVERY['grpsDF']['num_drivers'].iloc[k] = num_drivers;

    else: 
        if 'num_groups' in params: num_groups = params['num_groups'];
        else: num_groups = 2;

        for i  in range(num_groups):
            group = 'group'+str(i);
            DELIVERY['groups'][group] = {};
            num_drivers = 8; 
            driver_start_time = WORLD['main']['start_time'];
            driver_end_time = WORLD['main']['end_time'];
            am_capacity = 8
            wc_capacity = 2
            DELIVERY['groups'][group]['driver_runs'] = [];
            DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);

            DELIVERY['groups'][group]['date'] = '2023-07-31'
            loc = center_point
            DELIVERY['groups'][group]['depot'] = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
            DELIVERY['groups'][group]['depot_node'] = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

            DELIVERY['groups'][group]['time_matrix'] = np.zeros([1,1]);
            DELIVERY['groups'][group]['expected_cost'] = [0.];
            DELIVERY['groups'][group]['current_expected_cost'] = 0.;
            DELIVERY['groups'][group]['actual_average_cost'] = [];

            for j in range(num_drivers):
                DELIVERY['groups'][group]['driver_runs'].append({'run_id': j,'start_time': driver_start_time,'end_time': driver_end_time,
                    'am_capacity': am_capacity,'wc_capacity': wc_capacity})


    if len(DELIVERYDF)>0:
        DELIVERY['grpsDF']['num_possible_trips'] = list(np.zeros(len(DELIVERY['grpsDF'])))




    ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 
    ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 ##### NEW VERSION 1 

    num_drivers = 8; 
    driver_start_time = WORLD['main']['start_time'];
    driver_end_time = WORLD['main']['end_time'];
    am_capacity = 8
    wc_capacity = 2
    DELIVERY['driver_runs'] = [];
    DELIVERY['time_matrix'] = np.zeros([1,1]);

    DELIVERY['date'] = '2023-07-31'
    loc = center_point
    DELIVERY['depot'] = {'pt': {'lat': loc[1], 'lon': loc[0]}, 'node_id': 0}
    DELIVERY['depot_node'] = ox.distance.nearest_nodes(GRAPHS['drive'], loc[0], loc[1]);

    for i in range(num_drivers):
        DELIVERY['driver_runs'].append({'run_id': i,'start_time': driver_start_time,'end_time': driver_end_time,
            'am_capacity': am_capacity,'wc_capacity': wc_capacity})





    ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 
    ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 
    ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION ####### OLD VERSION 

    # nopts = len(opts);
    # people_tags = [];
    delivery1_tags = [];
    delivery2_tags = [];

    delivery_nodes = [];
    for i,loc in enumerate(direct_locs):
        tag = 'delivery1_' + str(i);
        delivery1_tags.append(tag);
        DELIVERY['direct'][tag] = {};
        DELIVERY['direct'][tag]['active_trips'] = [];
        DELIVERY['direct'][tag]['active_trip_history'] = [];    
        DELIVERY['direct'][tag]['loc'] = loc; #DELIVERY1_LOC[i]
        DELIVERY['direct'][tag]['current_path'] = []    

        DELIVERY['direct'][tag]['nodes'] = {};
        node = ox.distance.nearest_nodes(GRAPHS['ondemand'], loc[0], loc[1]);
        DELIVERY['direct'][tag]['nodes']['source'] = node
        direct_node_lists['source'].append(node)


        # node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
        # node = BUS_STOP_NODES['ondemand'][node0];
        node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
        node = int(convertNode(node0,'transit','ondemand',NDF))

        DELIVERY['direct'][tag]['nodes']['transit'] = node
        DELIVERY['direct'][tag]['nodes']['transit2'] = node0
        direct_node_lists['transit'].append(node)    

        DELIVERY['direct'][tag]['people'] = [];
        DELIVERY['direct'][tag]['sources'] = [];
        DELIVERY['direct'][tag]['targets'] = [];


    delivery2_nodes = [];
    for i,loc in enumerate(shuttle_locs):
        tag = 'delivery2_' + str(i);
        delivery2_tags.append(tag);
        DELIVERY['shuttle'][tag] = {};
        DELIVERY['shuttle'][tag]['active_trips'] = [];    
        DELIVERY['shuttle'][tag]['active_trip_history'] = [];    
        DELIVERY['shuttle'][tag]['loc'] = loc;
        DELIVERY['shuttle'][tag]['current_path'] = []

        DELIVERY['shuttle'][tag]['nodes'] = {};
        node = ox.distance.nearest_nodes(GRAPHS['ondemand'], loc[0], loc[1]);
        DELIVERY['shuttle'][tag]['nodes']['source'] = node
        shuttle_node_lists['source'].append(node)    

        # node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
        # node = BUS_STOP_NODES['ondemand'][node0];
        node0 = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0], loc[1]);
        node = int(convertNode(node0,'transit','ondemand',NDF))



        
        DELIVERY['shuttle'][tag]['nodes']['transit'] = node
        DELIVERY['shuttle'][tag]['nodes']['transit2'] = node0
        shuttle_node_lists['transit'].append(node)    

        DELIVERY['shuttle'][tag]['people'] = [];
        DELIVERY['shuttle'][tag]['sources'] = [];
        DELIVERY['shuttle'][tag]['targets'] = [];
        
    return DELIVERY




def generatePopulation(GRAPHS,DELIVERY,WORLD,NODES,VEHS,LOCS,PRE,params,verbose=True):  ##### ADDED TO CLASS

    people_tags = list(PRE); #params['people_tags']
    ORIG_LOC = LOCS['orig'] #params['ORIG_LOC'];
    DEST_LOC = LOCS['dest'] #params['DEST_LOC'];
    modes = params['modes'];
    graphs = params['graphs'];
    nodes = params['nodes'];
    factors = params['factors'];
    mass_scale = params['mass_scale']

    PEOPLE = {};
    start_time3 = time.time()
    print_interval = 10;


    mean = 500; stdev = 25;
    means = [mean,mean,mean,mean,mean];
    stdevs = [stdev,stdev,stdev,stdev,stdev];
    STATS = {}
    for m,mode in enumerate(modes):
        STATS[mode] = {};
        STATS[mode]['mean'] = dict(zip(factors,means));
        STATS[mode]['stdev'] = dict(zip(factors,stdevs));

    person_str = 'person';
    print('GENERATING POPULATION OF',len(people_tags),'...')

    DELIVERY['grpsDF']['num_possible_trips'] = list(np.zeros(len(DELIVERY['grpsDF'])));

    grpsDF = DELIVERY['grpsDF']
    for i,person in enumerate(people_tags):


        if np.mod(i,print_interval)==0:
            end_time3 = time.time()
            if verbose:
                print(person)
                print('time to add',print_interval, 'people: ',end_time3-start_time3)
            start_time3 = time.time()

        PEOPLE[person] = {
                        'mass': None,
                        'current_choice': None,
                        'current_cost': 0,
                        'choice_traj': [],
                        'delivery_grps': {'straight':None,'initial':None,'final':None},
                        'delivery_grp': None,        
                        'delivery_grp_initial': None,
                        'delivery_grp_final': None,
                        'logit_version': 'weighted_sum',
                        'trips':[],
                        'opts': ['drive','ondemand','walk','transit'],
                        'opts2': [random.choice(['walk','ondemand','drive']),
                                  'transit',
                                  random.choice(['walk','ondemand','drive'])],
                        'cost_traj': [],
                        'costs':{}, #opt, factor
                        'prefs':{}, #opt, factor                    
                        'weights':{}, #opt, factor

                        'am': int(1),
                        'wc': int(0),
                        'pickup_time_window_start': 18000,
                        'pickup_time_window_end': 19800,
                        'dropoff_time_window_start': 18900,
                        'dropoff_time_window_end': 22200,

                        'time_windows': {},
                        'booking_id': 39851211,
                        'pickup_node_id': 1,
                        'dropoff_node_id': 2
                        }



        PERSON = PEOPLE[person];
        PERSON['mass_total'] = PRE[person]['pop']
        PERSON['mass'] = PRE[person]['pop']*mass_scale


        if False:
            orig_loc = ORIG_LOC[i];
            dest_loc = DEST_LOC[i];
        else:
            orig_loc = PRE[person]['orig_loc'];
            dest_loc = PRE[person]['dest_loc'];


        PERSON['pickup_pt'] = {'lat': 35.0296296, 'lon': -85.2301767};
        PERSON['dropoff_pt'] = {'lat': 35.0734152, 'lon': -85.1315328};

        # PRE[person]['weight'][mode][factor]

        #### PREFERENCES #### PREFERENCES #### PREFERENCES #### PREFERENCES 
        ###################################################################
        if 'logit_version' in PRE[person]: PERSON['logit_version'] = PRE[person]['logit_version']; #OVERWRITES ABOVE

        for m,mode in enumerate(modes):
            PERSON['costs'][mode] = {};
            PERSON['prefs'][mode] = {};
            PERSON['weights'][mode] = {};
            for j,factor in enumerate(factors):
                sample_pt = STATS[mode]['mean'][factor] + STATS[mode]['stdev'][factor]*(np.random.rand()-0.5)
                PERSON['prefs'][mode][factor] = 0; #sample_pt
                PERSON['costs'][mode][factor] = 0.

                try:
                    PERSON['weights'][mode][factor] = PRE[person]['weight'][mode][factor]
                except:
                    if factor == 'time': PERSON['weights'][mode][factor] = 1.
                    else: PERSON['weights'][mode][factor] = 1.
            # PERSON['weights'][mode] = dict(zip(factors,np.ones(len(factors))));

        ###### DELIVERY ###### DELIVERY ###### DELIVERY ###### DELIVERY ###### DELIVERY 
        ###############################################################################

        picked_deliveries = {'direct':None,'initial':None,'final':None}
        person_loc = orig_loc
        dist = 10000000;
        picked_delivery = None;    
        for k,delivery in enumerate(DELIVERY['direct']):
            DELIV = DELIVERY['direct'][delivery]
            loc = DELIV['loc']

            diff = np.array(list(person_loc))-np.array(list(loc));
            if mat.norm(diff)<dist:
                PERSON['delivery_grps']['direct'] = delivery
                dist = mat.norm(diff);
                picked_deliveries['direct'] = delivery;

        
        person_loc = orig_loc
        dist = 10000000;
        picked_delivery = None;    
        for k,delivery in enumerate(DELIVERY['shuttle']):
            DELIV = DELIVERY['shuttle'][delivery]
            loc = DELIV['loc']
            diff = np.array(list(person_loc))-np.array(list(loc));
            if mat.norm(diff)<dist:
                PERSON['delivery_grps']['initial'] = delivery
                dist = mat.norm(diff);
                picked_deliveries['initial'] = delivery;            


        person_loc = dest_loc;
        dist = 10000000;
        picked_delivery = None;    
        for k,delivery in enumerate(DELIVERY['shuttle']):
            DELIV = DELIVERY['shuttle'][delivery]
            loc = DELIV['loc']
            diff = np.array(list(person_loc))-np.array(list(loc));
            if mat.norm(diff)<dist:
                PERSON['delivery_grps']['final'] = delivery
                dist = mat.norm(diff);
                picked_deliveries['final'] = delivery;              

        if not(picked_deliveries['direct']==None):
            DELIVERY['direct'][picked_deliveries['direct']]['people'].append(person)
        if not(picked_deliveries['initial']==None):        
            DELIVERY['shuttle'][picked_deliveries['initial']]['people'].append(person)    
        if not(picked_deliveries['final']==None):        
            DELIVERY['shuttle'][picked_deliveries['final']]['people'].append(person)             


        ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 
        ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 
        ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1 ##### VERSION1     

        
        PERSON['trips'] = {}; 


        if 'seg_types' in PRE[person]:
            seg_types = PRE[person]['seg_types']
            PERSON['trips_to_consider'] = PRE[person]['seg_types']

        elif False: 
            samp = np.random.rand(1);
            if samp<0.15:
                seg_types = [('drive',),
                             ('ondemand',),
                             ('walk','gtfs','walk'),
                             ('walk','gtfs','ondemand'),
                             ('ondemand','gtfs','walk'),
                             ('ondemand','gtfs','ondemand')
                            ];
                PERSON['trips_to_consider'] = seg_types
            elif (samp >= 0.15) & (samp <= 0.3):
                seg_types = [('walk','gtfs','walk'),
                             ('walk','gtfs','ondemand'),
                             ('ondemand','gtfs','walk'),
                             ('ondemand','gtfs','ondemand')
                            ];
                PERSON['trips_to_consider'] = seg_types                        
            else:
                seg_types = [('drive',)]
                PERSON['trips_to_consider'] = seg_types

        else: 
            samp = np.random.rand(1);
            if PRE[person]['take_car'] == 0.:
                seg_types = [('ondemand',),
                             ('walk','gtfs','walk'),
                             ('walk','gtfs','ondemand'),
                             ('ondemand','gtfs','walk'),
                             ('ondemand','gtfs','ondemand')
                            ];              
                PERSON['trips_to_consider'] = seg_types
            elif (PRE[person]['take_car']==1) & (samp < 0.3):
                seg_types = [('drive',),
                             ('ondemand',),
                             ('walk','gtfs','walk'),
                             ('walk','gtfs','ondemand'),
                             ('ondemand','gtfs','walk'),
                             ('ondemand','gtfs','ondemand')
                            ];
                PERSON['trips_to_consider'] = seg_types                        
            else:
                seg_types = [('drive',)]
                PERSON['trips_to_consider'] = seg_types



        start_time4 = time.time()


        new_trips_to_consider = [];
        for _,segs in enumerate(seg_types):


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

                NODES = addNodeToDF(start_node,mode1,GRAPHS,NODES);
                NODES = addNodeToDF(end_node,mode1,GRAPHS,NODES);
                #updateNodesDF(NODES);

                nodes_temp = [{'nodes':[start_node],'type':mode1},
                              {'nodes':[end_node],'type':mode1}]


                ##### SELECTING GROUP ######
                if mode1 == 'ondemand':
                    grouptag,groupind = selectDeliveryGroup((start_node,end_node),grpsDF,GRAPHS,typ='direct')
                    if len(grouptag)==0: add_new_trip = False;
                    else:  grpsDF.iloc[groupind]['num_possible_trips'] = grpsDF.iloc[groupind]['num_possible_trips'] + 1;



                ############################

                deliveries_temp = [];
                deliveries_temp2 = [];
                if mode1=='ondemand':
                    if not(picked_deliveries['direct']==None):
                        DELIVERY['direct'][picked_deliveries['direct']]['sources'].append(start_node);
                        DELIVERY['direct'][picked_deliveries['direct']]['targets'].append(end_node);
                    deliveries_temp.append(picked_deliveries['direct'])


                    # deliveries_temp2.append('group0'); #### EDITTING 
                    deliveries_temp2.append(grouptag); #### EDITTING 

                end_time = time.time();
                #print("segment1: ",np.round(end_time-start_time,4))

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
                    transit1_walk,transit2_walk = nearest_applicable_gtfs_node(mode2,GRAPHS,WORLD,NODES,
                                                                 orig_loc[0],orig_loc[1],
                                                                 dest_loc[0],dest_loc[1]);
                                                                 # rad1=rad1,rad2=rad2);  ####HERE 
                    # print('transit node 1 is type...',type(transit2_walk))

                if mode1 == 'ondemand':
                    initial_delivery = PERSON['delivery_grps']['initial'];
                    transit_node1 = DELIVERY['shuttle'][initial_delivery]['nodes']['transit2'];
                    if mode2=='gtfs':
                        transit_node1 = convertNode(transit_node1,'transit','gtfs',NODES)                
                elif mode1 == 'walk':
                    transit_node1 = transit1_walk;
                    # transit_node1 = nearest_nodes(mode2,GRAPHS,NODES,orig_loc[0],orig_loc[1]);  ####HERE 
                    # print('old transit node 1 is type...',type(transit_node1))
                    #transit_node1 = ox.distance.nearest_nodes(GRAPHS[mode2], orig_loc[0],orig_loc[1]);

                if mode3 == 'ondemand':
                    final_delivery = PERSON['delivery_grps']['final'];
                    transit_node2 = DELIVERY['shuttle'][final_delivery]['nodes']['transit2'];                                
                    if mode2=='gtfs':
                        transit_node2 = convertNode(transit_node2,'transit','gtfs',NODES)

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
                NODES = addNodeToDF(start_node,mode1,GRAPHS,NODES);
                NODES = addNodeToDF(end_node,mode3,GRAPHS,NODES);            
                NODES = addNodeToDF(transit_node1,mode2,GRAPHS,NODES);
                NODES = addNodeToDF(transit_node2,mode2,GRAPHS,NODES);
                

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


                # transit_node1 = convertNode(transit_node1,'transit','gtfs',NODES)

                if mode1 == 'ondemand':
                    if mode2=='gtfs':
                        transit_node_converted = convertNode(transit_node1,'gtfs','ondemand',NODES)
                    grouptag1,groupind1 = selectDeliveryGroup((start_node,transit_node_converted),grpsDF,GRAPHS,typ='shuttle')
                    if len(grouptag1)==0: add_new_trip = False;
                    else:  grpsDF.iloc[groupind1]['num_possible_trips'] = grpsDF.iloc[groupind1]['num_possible_trips'] + 1;

                if mode3=='ondemand':
                    if mode2=='gtfs':
                        transit_node_converted = convertNode(transit_node2,'gtfs','ondemand',NODES)
                    grouptag3,groupind3 = selectDeliveryGroup((transit_node_converted,end_node),grpsDF,GRAPHS,typ='shuttle')
                    if len(grouptag3)==0: add_new_trip = False;
                    else:  grpsDF.iloc[groupind3]['num_possible_trips'] = grpsDF.iloc[groupind3]['num_possible_trips'] + 1;





                deliveries_temp = [None,None,None];
                deliveries_temp2 = [None,None,None];
                if mode1=='ondemand':  
                    if not(picked_deliveries['initial']==None):                
                        DELIVERY['shuttle'][picked_deliveries['initial']]['sources'].append(start_node);
                        deliveries_temp.append(picked_deliveries['initial'])
                    # deliveries_temp2[0] = 'group1' #### EDITTING 
                    deliveries_temp2[0] = grouptag1 #### EDITTING 
                if mode3=='ondemand':
                    if not(picked_deliveries['final']==None):
                        DELIVERY['shuttle'][picked_deliveries['final']]['targets'].append(end_node);
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
                TRIP = makeTrip(segs,nodes_temp,NODES,deliveries_temp2,deliveries_temp)
                PERSON['trips'][segs] = TRIP;
                end_time = time.time()
            #print('time to make trip: ',np.round(end_time-start_time,4))

        PERSON['trips_to_consider'] = new_trips_to_consider 
    DELIVERY['grpsDF'] = grpsDF.copy();
        ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------  
        ###### ------ ###### ------ ###### ------ ###### ------ ###### ------ ###### ------            
    return PEOPLE




def createEdgeCosts(mode,GRAPHS):   ##### ADDED TO CLASS 
    POLYS = {};
    if (mode == 'drive') or (mode == 'ondemand'):
        # keys '25 mph'
        GRAPH = GRAPHS[mode];
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


    if (mode == 'walk'):
        # keys '25 mph'
        GRAPH = GRAPHS[mode];
        for i,edge in enumerate(GRAPH.edges):
            EDGE = GRAPH.edges[edge];
            #maxspeed = EDGE['maxspeed'];
            maxspeed = 3.; # in mph
            length = EDGE['length'] ; # assumed in meters? 1 mph = 0.447 m/s
            t0 = length/(maxspeed * 0.447)
            EDGE['cost_fx'] = [t0];
            POLYS[edge] = [t0];

    return POLYS


def compute_UncongestedEdgeCosts(WORLD,GRAPHS):  #### ADDED TO CLASS 
    modes = ['drive','walk'];
    for mode in modes:
        GRAPH = GRAPHS[mode]
        edge_costs0 = {}
        for e,edge in enumerate(GRAPH.edges):
            poly = WORLD[mode]['cost_fx'][edge];
            edge_costs0[edge] = poly[0]
        nx.set_edge_attributes(GRAPH,edge_costs0,'c0')


def compute_UncongestedTripCosts(WORLD,GRAPHS):   ### ADDED TO CLASS 
    modes = ['drive','walk','gtfs','ondemand'];
    for mode in modes:
        if mode == 'ondemand': GRAPH = GRAPHS['drive']
        else: GRAPH = GRAPHS[mode]
        for t,trip in enumerate(WORLD[mode]['trips']):
            source = trip[0]; target = trip[1];
            try:
                temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c0'); #Dumbfunctions
                distance = temp[0];
                path = temp[1]; 
            except: 
                #print('no path found for bus trip ',trip,'...')
                distance = 1000000000000;
                path = [];
            WORLD[mode]['trips'][trip]['uncongested'] = {};
            WORLD[mode]['trips'][trip]['uncongested']['path'] = path;
            WORLD[mode]['trips'][trip]['uncongested']['costs'] = {}; 
            WORLD[mode]['trips'][trip]['uncongested']['costs']['time'] = distance;
            WORLD[mode]['trips'][trip]['uncongested']['costs']['money'] = 0;
            WORLD[mode]['trips'][trip]['uncongested']['costs']['conven'] = 0;
            WORLD[mode]['trips'][trip]['uncongested']['costs']['switches'] = 0;        


    # for i,person in enumerate(WALK['people']):
    #     PERSON = PEOPLE[person];

    #     trips = [];
    #     source = PERSON['nodes']['source']['walk'];
    #     target = PERSON['nodes']['target']['walk'];
    #     trips.append((source,target));
    #     source = PERSON['nodes']['source']['walk'];
    #     target = PERSON['nodes']['transit_walk1']['walk'];
    #     trips.append((source,target));        
    #     source = PERSON['nodes']['transit_walk2']['walk'];
    #     target = PERSON['nodes']['target']['walk'];
    #     trips.append((source,target));

    #     seg_types = [['walk'],
    #                  ['walk+transit+drive','walk+transit+walk','walk+transit+ondemand'],
    #                  ['drive+transit+walk','walk+transit+walk','ondemand+transit+walk']]
    #     for tt,trip in enumerate(trips):
    #         mass = 1;
    #         source,target = trip;
    #         try:
    #             temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c');
    #             distance = temp[0];
    #             path = temp[1]; 
    #         except: 
    #             print('no path found for walking trip ',trip,'...')
    #             distance = 1000000;
    #             path = [];      
        
    #         if not(trip in WALK['trips'].keys()):
    #             WALK['trips'][trip] = {};
    #             WALK['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
    #             WALK['trips'][trip]['path'] = [];
    #             WALK['trips'][trip]['mass'] = 1;
            
            
    #         WALK['trips'][trip]['costs']['time'].append(distance)
    #         WALK['trips'][trip]['costs']['money'].append(1)
    #         WALK['trips'][trip]['costs']['conven'].append(1)
    #         WALK['trips'][trip]['costs']['switches'].append(1)
    #         WALK['trips'][trip]['path'].append(path);
        
    #         if True: #PERSON['current_choice'] in seg_types[tt]:
    #             for j,node in enumerate(path):
    #                 if j<len(path)-1:
    #                     edge = (path[j],path[j+1],0)            
    #                     edge_mass = WALK['edge_masses'][edge][-1] + mass;
    #                     edge_cost = 1.*edge_mass + 1.;

    #                     WALK['edge_masses'][edge][-1] = edge_mass;
    #                     WALK['edge_costs'][edge][-1] = edge_cost;
    #                     WALK['current_edge_costs'][edge] = edge_cost;
    #                     WALK['current_edge_masses'][edge] = edge_mass;    
    #         WALK['trips'][trip]['active'] = False;

    # WALK['active_trips']  = [];            
        


###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD 
###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD 
###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD 
###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD 
###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD ###### END OF WORLD 






###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 
###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE ###### SIMULATE 


########### =================== RUNNING SIMULATION ======================= ##############
########### =================== RUNNING SIMULATION ======================= ##############
########### =================== RUNNING SIMULATION ======================= ##############
########### =================== RUNNING SIMULATION ======================= ##############
########### =================== RUNNING SIMULATION ======================= ##############

####### world of drive ####### world of drive ####### world of drive ####### world of drive ####### world of drive 
####### world of drive ####### world of drive ####### world of drive ####### world of drive ####### world of drive 








def world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=False,clear_active=True): ##### ADDED TO CLASS 
    #graph,costs,sources, targets):
    if verbose: print('starting driving computations...')
    mode = 'drive'
    kk = WORLD['main']['iter']
    alpha = WORLD['main']['alpha']
    # DRIVE = WORLD['drive'];
    GRAPH = GRAPHS[WORLD[mode]['graph']];
    if (not('edge_masses' in WORLD[mode].keys()) or (kk==0)):
        WORLD[mode]['edge_masses'] = {};
        WORLD[mode]['edge_costs'] = {};
        WORLD[mode]['current_edge_costs'] = {};
        WORLD[mode]['current_edge_masses'] = {};
        WORLD[mode]['edge_a0'] = {};
        WORLD[mode]['edge_a1'] = {};
        for e,edge in enumerate(GRAPH.edges):
            WORLD[mode]['edge_masses'][edge] = [0]
            current_edge_cost = WORLD[mode]['cost_fx'][edge][0];
            WORLD[mode]['edge_costs'][edge] = [current_edge_cost]; #WORLD[mode]['cost_fx'][edge][0]];
            WORLD[mode]['current_edge_costs'][edge] = current_edge_cost; 
            WORLD[mode]['current_edge_masses'][edge] = 0. 
            # WORLD[mode]['edge_a0'][edge] = 1;
            # WORLD[mode]['edge_a1'][edge] = 1;


    else: #### GRADIENT UPDATE STEP.... 
        for e,edge in enumerate(GRAPH.edges):
            # WORLD[mode]['edge_masses'][edge].append(0)
            current_cost = WORLD[mode]['current_edge_costs'][edge]
            poly = WORLD[mode]['cost_fx'][edge]


            current_edge_mass = WORLD[mode]['current_edge_masses'][edge];
            if 'base_edge_masses' in WORLD[mode]:
                current_edge_mass = current_edge_mass + WORLD[mode]['base_edge_masses'][edge];
            expected_mass = invDriveCostFx(current_cost,poly)
            #diff = WORLD[mode]['current_edge_masses'][edge] - expected_mass;
            diff = current_edge_mass - expected_mass; 

            new_edge_cost = WORLD[mode]['current_edge_costs'][edge] + alpha * diff
            min_edge_cost = poly[0];
            max_edge_cost = 10000000.;
            new_edge_cost = np.min([np.max([min_edge_cost,new_edge_cost]),100000.])
            WORLD[mode]['edge_costs'][edge].append(new_edge_cost)
            WORLD[mode]['current_edge_costs'][edge] = new_edge_cost;            
        
    # if True: #'current_edge_costs' in WORLD[mode].keys():
    current_costs = WORLD[mode]['current_edge_costs'];
    # else: # INITIALIZE 
    #     current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}


    # print(current_costs)
    nx.set_edge_attributes(GRAPH,current_costs,'c');
    mode = 'drive'
    removeMassFromEdges(mode,WORLD,GRAPHS)
    segs = WORLD[mode]['active_trips']
    print('...with ',len(segs),' active trips...')    
    for i,seg in enumerate(WORLD[mode]['trips']):
        if seg in segs: 
            if np.mod(i,500)==0: print('>>> segment',i,'...')
            source = seg[0];
            target = seg[1];
            trip = (source,target)
            mass = WORLD[mode]['trips'][trip]['mass']
            planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=mass,track=True);
            WORLD[mode]['trips'][trip]['active'] = False;
        else:
            if len(WORLD[mode]['trips'][seg]['costs']['time'])==0:
                WORLD[mode]['trips'][seg]['costs']['time'].append(0)
            else:
                recent_value = WORLD[mode]['trips'][seg]['costs']['time'][-1]
                WORLD[mode]['trips'][seg]['costs']['time'].append(recent_value)

    if clear_active:
        WORLD[mode]['active_trips']  = [];

    ######## REMOVING MASS ON TRIPS ##########
    ######## REMOVING MASS ON TRIPS ##########
    for i,trip in enumerate(WORLD[mode]['trips']):
        TRIP = WORLD[mode]['trips'][trip];
        TRIP['mass'] = 0;



####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
                        
def world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=False,clear_active=True):  ### ADDED TO CLASS 
        #graph,costs,sources, targets):
    if verbose: print('starting walking computations...')
    mode = 'walk'
    kk = WORLD['main']['iter']
    alpha = WORLD['main']['alpha']

    WALK = WORLD[mode];
    GRAPH = GRAPHS[WALK['graph']];    
    if (not('edge_masses' in WALK.keys()) or (kk==0)):
        WALK['edge_masses'] = {};
        WALK['edge_costs'] = {};
        WALK['current_edge_costs'] = {};
        WALK['edge_a0'] = {};
        WALK['edge_a1'] = {};
        for e,edge in enumerate(GRAPH.edges):
            # WALK['edge_masses'][edge] = [0]
            # WALK['edge_costs'][edge] = [1]
            # WALK['current_edge_costs'][edge] = 1.;#GRAPH.edges[edge]['cost_fx'][0];
            # WALK['edge_a0'][edge] = 1;
            # WALK['edge_a1'][edge] = 1;               
            WORLD[mode]['edge_masses'][edge] = [0]
            current_edge_cost = WORLD[mode]['cost_fx'][edge][0];
            WORLD[mode]['edge_costs'][edge] = [current_edge_cost]; #WORLD[mode]['cost_fx'][edge][0]];
            WORLD[mode]['current_edge_costs'][edge] = current_edge_cost; 
            WORLD[mode]['current_edge_masses'][edge] = 0. 
            # WORLD[mode]['edge_a0'][edge] = 1;
            # WORLD[mode]['edge_a1'][edge] = 1;

    else: 
        for e,edge in enumerate(GRAPH.edges):
            # WALK['edge_masses'][edge].append(0)
            # WALK['edge_costs'][edge].append(0)
            # WALK['current_edge_costs'][edge] = 1;
            # WORLD[mode]['edge_masses'][edge].append(0)

            current_edge_cost = WORLD[mode]['current_edge_costs'][edge];# + alpha * WORLD[mode]['current_edge_masses'][edge] 
            WORLD[mode]['edge_costs'][edge].append(current_edge_cost)
            WORLD[mode]['current_edge_costs'][edge] = current_edge_cost;            
        
    # if True: #'current_edge_costs' in WORLD[mode].keys():
    current_costs = WORLD[mode]['current_edge_costs'];
    # else: # INITIALIZE 
    #     current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}        


    nx.set_edge_attributes(GRAPH,current_costs,'c');     

    mode = 'walk'
    removeMassFromEdges(mode,WORLD,GRAPHS)  
    segs = WORLD[mode]['active_trips']
    print('...with ',len(segs),' active trips...')        
    for i,seg in enumerate(WORLD[mode]['trips']):
        if seg in segs:
            if np.mod(i,500)==0: print('>>> segment',i,'...')
            source = seg[0];
            target = seg[1];
            trip = (source,target)
            mass = WORLD[mode]['trips'][trip]['mass']
            planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=mass,track=True);
            WORLD[mode]['trips'][trip]['active'] = False;
        else:
            if len(WORLD[mode]['trips'][seg]['costs']['time'])==0:
                WORLD[mode]['trips'][seg]['costs']['time'].append(0)
            else:
                recent_value = WORLD[mode]['trips'][seg]['costs']['time'][-1]
                WORLD[mode]['trips'][seg]['costs']['time'].append(recent_value)



    if clear_active:    
        WORLD[mode]['active_trips']  = [];

    ######## REMOVING MASS ON TRIPS ##########
    ######## REMOVING MASS ON TRIPS ##########
    for i,trip in enumerate(WORLD[mode]['trips']):
        TRIP = WORLD[mode]['trips'][trip];
        TRIP['mass'] = 0;


def generateOndemandCongestion(WORLD,PEOPLE,DELIVERIES,GRAPHS,num_counts=3,num_per_count=1,verbose=True,show_delivs='all',clear_active=True):  #### ADDED TO CLASS 

    if 'groups' in DELIVERIES:
        mode = 'ondemand'
        all_possible_trips = list(WORLD[mode]['trips']);
        if verbose: print('starting on-demand curve computations for a total of',len(all_possible_trips),'trips...')

        trips_by_group = {group:[] for _,group in enumerate(DELIVERIES['groups'])}
        for _,trip in enumerate(all_possible_trips):
            group = WORLD[mode]['trips'][trip]['group'];
            trips_by_group[group].append(trip)

        for k, group in enumerate(DELIVERIES['groups']):

            possible_trips = trips_by_group[group]

            if len(possible_trips)>0:
                # counts = [20,30,40,50,60,70,80];
                num_pts = num_counts; num_per_count = num_per_count; 

                counts = np.linspace(1,len(possible_trips)-1,num_pts)
                counts = [int(count) for _,count in enumerate(counts)];
                
                if verbose: print('starting on-demand curve computation for',group) #'with',len(possible_trips),'...')
                
                GRAPH = GRAPHS['ondemand'];
                ONDEMAND = WORLD['ondemand'];
                DELIVERY = DELIVERIES['groups'][group]

                print('...for a total number of trips of',len(possible_trips))
                print('counts to compute: ',counts)

                pts = np.zeros([len(counts),num_per_count]);


                ptsx = {};
                for i,count in enumerate(counts):
                    print('...computing averages for',count,'active ondemand trips in',group,'...')
                    for j in range(num_per_count):

                        try: 
                            trips_to_plan = sample(possible_trips,count)
                            nodes_to_update = [DELIVERY['depot_node']];
                            nodeids_to_update = [0];
                            for _,trip in enumerate(possible_trips):
                                TRIP = WORLD[mode]['trips'][trip]
                                nodes_to_update.append(trip[0]);
                                nodes_to_update.append(trip[1]);
                                nodeids_to_update.append(TRIP['pickup_node_id'])
                                nodeids_to_update.append(TRIP['dropoff_node_id'])

                            updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                            payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);


                            manifests = optimizer.offline_solver(payload) 
                            PDF = payloadDF(payload,GRAPHS)
                            MDF = manifestDF(manifests,PDF)
                            average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,0)
                            pts[i,j] = average_time
                        except:
                            pass


                        # ptsx[(i,j)] = times_to_average

                DELIVERY['fit'] = {}
                DELIVERY['fit']['counts'] = np.outer(counts,np.ones(num_per_count))
                DELIVERY['fit']['pts'] = pts.copy();

                shp = np.shape(pts);
                xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
                yy = np.reshape(pts,shp[0]*shp[1])
                order = 1;
                coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
                Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
                DELIVERY['fit']['poly'] = coeffs[::-1]
                DELIVERY['fit']['Yh'] = Yh.copy();

            else:
                DELIVERY = DELIVERIES['groups'][group]
                counts = np.linspace(1,100,num_counts)
                DELIVERY['fit'] = {}
                DELIVERY['fit']['counts'] = np.outer(counts,np.ones(num_per_count))
                pts = 1000.*np.ones([num_counts,num_per_count]);
                DELIVERY['fit']['pts'] = pts; #1000.*np.ones([num_counts,num_per_count]);

                shp = np.shape(pts);
                xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
                yy = np.reshape(pts,shp[0]*shp[1])
                order = 1;
                coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
                Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
                DELIVERY['fit']['poly'] = coeffs[::-1]
                DELIVERY['fit']['Yh'] = Yh.copy();
        out = None


    else: 
        DELIVERY = DELIVERIES;
        if verbose: print('starting on-demand curve computation...')    
        mode = 'ondemand'
        GRAPH = GRAPHS['ondemand'];
        ONDEMAND = WORLD['ondemand'];
        possible_trips = list(WORLD[mode]['trips']) ;

        print('...for a total number of trips of',len(possible_trips))
        print('counts to compute: ',counts)

        pts = np.zeros([len(counts),num_per_count]);

        ptsx = {};
        for i,count in enumerate(counts):
            print('...computing averages for',count,'active ondemand trips...')
            for j in range(num_per_count):
                trips_to_plan = sample(possible_trips,count)

                nodes_to_update = [DELIVERY['depot_node']];
                nodeids_to_update = [0];
                for _,trip in enumerate(trips_to_plan):
                    TRIP = WORLD[mode]['trips'][trip]
                    nodes_to_update.append(trip[0]);
                    nodes_to_update.append(trip[1]);
                    nodeids_to_update.append(TRIP['pickup_node_id'])
                    nodeids_to_update.append(TRIP['dropoff_node_id'])

                updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

                manifests = optimizer.offline_solver(payload) 
                PDF = payloadDF(payload,GRAPHS)
                MDF = manifestDF(manifests,PDF)
                average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,0)

                pts[i,j] = average_time
                # ptsx[(i,j)] = times_to_average
        out = pts

# # counts = [20,30,40,50,60,70,80];
# num_pts = 5; num_per_count = 1; 
# trips = list(WORLD['ondemand']['trips'])
# counts = np.linspace(1,len(trips)-1,num_pts)
# counts = [int(count) for _,count in enumerate(counts)];
# pts = generateOndemandCongestion(counts,WORLD,PEOPLE,DELIVERY,GRAPHS,num_per_count=num_per_count)

# WORLD['ondemand']['fit'] = {}
# WORLD['ondemand']['fit']['counts'] = np.outer(counts,np.ones(4))
# WORLD['ondemand']['fit']['pts'] = pts.copy();

# shp = np.shape(pts);
# xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
# yy = np.reshape(pts,shp[0]*shp[1])
# order = 1;
# coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
# Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
# WORLD['ondemand']['fit']['poly'] = coeffs[::-1]

# for j in range(np.shape(pts)[1]):
#     plt.plot(counts,pts[:,j],'o',color='blue')
# plt.plot(xx,Yh,'--',color='red')
# print(coeffs[::-1])

# # data_points = {'x':xx,'y':yy}
# # fileObj = open('data/fits/demandcurve153','wb')
# # pickle.dump(data_points,fileObj)
# # fileObj.close()


    return out#,ptsx


# def assignGroupToTrips(vers='region') 



def world_of_ondemand(WORLD,PEOPLE,DELIVERIES,GRAPHS,verbose=False,show_delivs='all',clear_active=True): ##### ADDED TO CLASSA 
    ##### ADDED TO CLASSA 
    if verbose: print('starting on-demand computations...')    

    #### PREV: tsp_wgrps
    #lam = grads['lam']
    #pickups = current_pickups(lam,all_pickup_nodes) ####
    #nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
    kk = WORLD['main']['iter']
    alpha = WORLD['main']['alpha']

    mode = 'ondemand'
    GRAPH = GRAPHS['ondemand'];
    ONDEMAND = WORLD['ondemand'];

    total_trips_to_plan = WORLD[mode]['active_trips'];#_direct'] + WORLD['ondemand']['active_trips_shuttle'];
    num_total_trips = len(total_trips_to_plan);

    ##### SORTING TRIPS BY GROUPS: 
    if 'groups' in DELIVERIES:
        GROUPS_OF_TRIPS = {};
        for _,group in enumerate(DELIVERIES['groups']):
            GROUPS_OF_TRIPS[group] = [];

        for i,trip in enumerate(WORLD[mode]['active_trips']):
            TRIP = WORLD[mode]['trips'][trip];
            if 'group' in TRIP:
                group = TRIP['group'];
                GROUPS_OF_TRIPS[group].append(trip);

        #### TO ADD: SORT TRIPS BY GROUP
        for _,group in enumerate(GROUPS_OF_TRIPS):

            trips_to_plan = GROUPS_OF_TRIPS[group]
            DELIVERY = DELIVERIES['groups'][group]



            ########### -- XX START HERE ---- ####################
            ########### -- XX START HERE ---- ####################
            ########### -- XX START HERE ---- ####################
            print('...with',len(trips_to_plan),'active ondemand trips...')

            # poly = WORLD[mode]['cost_poly'];
            # poly = np.array([-6120.8676711, 306.5130127]) # 1st order
            # poly = np.array([-8205.87778054,   342.32193064]) # 1st order 
            # poly = np.array([5047.38255623,-288.78570445,6.31107635]); # 2nd order


            if 'fit' in DELIVERY:
                poly  = DELIVERY['fit']['poly'];
            elif 'fit' in WORLD['ondemand']:
                poly = WORLD['ondemand']['fit']['poly'];
            else:
                poly = WORLD['ondemand']['poly'];

            # poly = np.array([6.31107635, -288.78570445, 5047.38255623]) # 2nd order 


            num_trips = len(trips_to_plan);

            ### dumby values...
            MDF = []; nodes = [];
            if len(trips_to_plan)>0:
                nodes_to_update = [DELIVERY['depot_node']];
                nodeids_to_update = [0];
                for j,trip in enumerate(trips_to_plan):
                    TRIP = WORLD[mode]['trips'][trip]
                    nodes_to_update.append(trip[0]);
                    nodes_to_update.append(trip[1]);
                    nodeids_to_update.append(TRIP['pickup_node_id'])
                    nodeids_to_update.append(TRIP['dropoff_node_id'])


                updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
                payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

                print('')
                print(payload.keys())
                print('')
                print(payload['date'])
                print(payload['depot'])
                print('')
                print(payload['driver_runs'])
                print('')
                print(payload['time_matrix'][0][:10])
                print('')
                print(payload['requests'])
                asdfasdf


                # import pickle
                # fileObj = open('payload1', 'wb')
                # pickle.dump(payload,fileObj)
                # fileObj.close()

                manifests = optimizer.offline_solver(payload) 
                # manifest = optimizer(payload) ####
                # manifest = fakeManifest(payload)

                PDF = payloadDF(payload,GRAPHS)
                MDF = manifestDF(manifests,PDF)

                DELIVERY['current_PDF'] = PDF;
                DELIVERY['current_MDF'] = MDF;
                DELIVERY['current_payload'] = payload;
                DELIVERY['current_manifest'] = manifests;


            average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,WORLD[mode]['current_expected_cost'])

            # print(poly)
            # print(WORLD[mode]['current_expected_cost'])

            # expected_num_trips = invDriveCostFx(WORLD[mode]['current_expected_cost'],poly)
            expected_num_trips = invDriveCostFx(DELIVERY['current_expected_cost'],poly)            


            print('...expected num of trips given cost:',expected_num_trips)
            print('...actual num trips:',num_trips)

            diff = num_trips - expected_num_trips

            print('...adjusting cost estimate by',alpha*diff);
            # new_ave_cost = WORLD[mode]['current_expected_cost'] + alpha * diff
            new_ave_cost = DELIVERY['current_expected_cost'] + alpha * diff

            min_ave_cost = poly[0];
            max_ave_cost = 100000000.;
            new_ave_cost = np.min([np.max([min_ave_cost,new_ave_cost]),max_ave_cost])

            print('...changing cost est from',DELIVERY['current_expected_cost'],'to',new_ave_cost )

            DELIVERY['expected_cost'].append(new_ave_cost)
            DELIVERY['current_expected_cost'] = new_ave_cost

            DELIVERY['actual_average_cost'].append(average_time)

        for i,trip in enumerate(WORLD[mode]['trips']):        
            if not(trip in total_trips_to_plan):
                if len(WORLD[mode]['trips'][trip]['costs']['time'])==0:
                    WORLD[mode]['trips'][trip]['costs']['time'].append(0)
                    WORLD[mode]['trips'][trip]['costs']['current_time']= 0;
                else:
                    recent_value = WORLD[mode]['trips'][trip]['costs']['time'][-1]
                    WORLD[mode]['trips'][trip]['costs']['time'].append(recent_value)
                    WORLD[mode]['trips'][trip]['costs']['current_time'] = recent_value

        if clear_active: 
            WORLD[mode]['active_trips']  = [];



    else:
        ########### -- XX START HERE ---- ####################
        ########### -- XX START HERE ---- ####################
        ########### -- XX START HERE ---- ####################
        print('...with',len(trips_to_plan),'active ondemand trips...')

        # poly = WORLD[mode]['cost_poly'];
        # poly = np.array([-6120.8676711, 306.5130127]) # 1st order
        # poly = np.array([-8205.87778054,   342.32193064]) # 1st order 
        # poly = np.array([5047.38255623,-288.78570445,6.31107635]); # 2nd order


        if 'fit' in WORLD['ondemand']:
            poly = WORLD['ondemand']['fit']['poly'];
        else:
            poly = WORLD['ondemand']['poly'];

        # poly = np.array([6.31107635, -288.78570445, 5047.38255623]) # 2nd order 


        num_trips = len(trips_to_plan);

        ### dumby values...
        MDF = []; nodes = [];
        if len(trips_to_plan)>0:
            nodes_to_update = [DELIVERY['depot_node']];
            nodeids_to_update = [0];
            for j,trip in enumerate(trips_to_plan):
                TRIP = WORLD[mode]['trips'][trip]
                nodes_to_update.append(trip[0]);
                nodes_to_update.append(trip[1]);
                nodeids_to_update.append(TRIP['pickup_node_id'])
                nodeids_to_update.append(TRIP['dropoff_node_id'])

            updateTravelTimeMatrix(nodes_to_update,nodeids_to_update,GRAPHS['ondemand'],DELIVERY,WORLD);
            payload,nodes = constructPayload(trips_to_plan,DELIVERY,WORLD,GRAPHS);

            # import pickle
            # fileObj = open('payload1', 'wb')
            # pickle.dump(payload,fileObj)
            # fileObj.close()

            manifests = optimizer.offline_solver(payload) 
            # manifest = optimizer(payload) ####
            # manifest = fakeManifest(payload)


            PDF = payloadDF(payload,GRAPHS)
            MDF = manifestDF(manifests,PDF)

            DELIVERY['current_PDF'] = PDF;
            DELIVERY['current_MDF'] = MDF;
            DELIVERY['current_payload'] = payload;
            DELIVERY['current_manifest'] = manifests;



        average_time,times_to_average = assignCostsFromManifest(trips_to_plan,nodes,MDF,WORLD,WORLD[mode]['current_expected_cost'])

        # print(poly)
        # print(WORLD[mode]['current_expected_cost'])

        expected_num_trips = invDriveCostFx(WORLD[mode]['current_expected_cost'],poly)





        print('...expected num of trips given cost:',expected_num_trips)
        print('...actual num trips:',num_trips)

        diff = num_trips - expected_num_trips

        print('...adjusting cost estimate by',alpha*diff);
        new_ave_cost = WORLD[mode]['current_expected_cost'] + alpha * diff



        min_ave_cost = poly[0];
        max_ave_cost = 100000000.;
        new_ave_cost = np.min([np.max([min_ave_cost,new_ave_cost]),max_ave_cost])

        print('...changing cost est from',WORLD[mode]['current_expected_cost'],'to',new_ave_cost )

        WORLD[mode]['expected_cost'].append(new_ave_cost)
        WORLD[mode]['current_expected_cost'] = new_ave_cost

        WORLD[mode]['actual_average_cost'].append(average_time)

        for i,trip in enumerate(WORLD[mode]['trips']):        
            if not(trip in trips_to_plan):
                if len(WORLD[mode]['trips'][trip]['costs']['time'])==0:
                    WORLD[mode]['trips'][trip]['costs']['time'].append(0)
                else:
                    recent_value = WORLD[mode]['trips'][trip]['costs']['time'][-1]
                    WORLD[mode]['trips'][trip]['costs']['time'].append(recent_value)

        if clear_active: 
            WORLD[mode]['active_trips']  = [];
    # WORLD[mode]['active_trips_shuttle'] = [];
    # WORLD[mode]['active_trips_direct'] = [];

    # else: 
    #     ########### -- XX START HERE ---- ###################

    #     DELIVERY['active_trips'] = [];


    #     for i,trip in enumerate(trips):
    #         start = trip[0];
    #         xloc = GRAPH.nodes[start]['x'];
    #         yloc = GRAPH.nodes[start]['y'];
    #         end = trip[1];



    #     for i,deliv in enumerate(DELIVERY):
    #         DELIVERY[deliv]['active_trip_history'].append(DELIVERY[deliv]['active_trips'])                


    # #################################################
    # elif False: #vers = 'ver2':
    #     # Lloyd-Walsh


    #     for i,trip in enumerate(trips):
    #         start = trip[0];
    #         xloc = GRAPH.nodes[start]['x'];
    #         yloc = GRAPH.nodes[start]['y'];
    #         end = trip[1];
    #         # max_dist = 100000000000;
    #         # dists = [100000000000000];
    #         # delivs = [None];
    #         # current_deliv = None;
    #         dists = [];
    #         for j,deliv in enumerate(DELIVERY):
    #             DELIV = DELIVERY[deliv];
    #             loc0 = DELIV['loc'];
    #             dist = np.sqrt((xloc-loc0[0])*(xloc-loc0[0])+(yloc-loc0[1])*(yloc-loc0[1]));
    #             dists.append(dist);
    #         dists = np.array(dists);

    #         inds = np.argsort(dists).astype(int)
    #         delivs = [deliverys_unsorted[ind] for i,ind in enumerate(inds)]
    #         dists = dists[inds];
    #         j = 0; spot_found = False;
    #         while (j<(len(delivs)-1)) & (spot_found==False):
    #             deliv = delivs[j];
    #             DELIV = DELIVERY[deliv];
    #             if len(DELIV['active_trips'])<maxTrips:
    #                 DELIV['active_trips'].append(trip)
    #                 spot_found = True;
    #             j = j+1;        



    #         trips_to_plan = WORLD['ondemand']['active_trips']
    #         DELIVERY0 = DELIVERY['direct'];
    #         maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));
    #         divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD,maxTrips);

    #     #     print(len(dists))
    #     #     for ll in range(len(dists)):
    #     #         if dist < dists[ll] :
    #     #             dists.insert(ll,dist);
    #     #             delivs.insert(ll,deliv);
    #     #     # if dist < max_dist:
    #     #     #     current_deliv = deliv;
    #     #     #     max_dist = dist;
    #     # dists = dists[:-1]
    #     # delivs = delivs[:-1]
    #     # # print(len(delivs))
    #     # for j,deliv in enumerate(delivs):
    #     #     DELIV = DELIVERY[deliv];
    #     #     if len(DELIV['active_trips'])<maxTrips:
    #     #         DELIV['active_trips'].append(trip)







    #     divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD,maxTrips);
    #     #######################################################

    #     if show_delivs=='all': delivs_to_show = list(range(len(list(DELIVERY0.keys()))));
    #     elif isinstance(show_delivs,int): delivs_to_show = list(range(show_delivs));
    #     else: delivs_to_show = show_delivs;

    #     for i,delivery in enumerate(DELIVERY0):
    #         if np.mod(i,200)==0: print('direct',i,'...')
    #         DELIV = DELIVERY0[delivery];
    #         active_trips = DELIV['active_trips'];
    #         sources = []; targets = [];
    #         start = DELIV['nodes']['source'];        
    #         [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
    #         [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
    #         planDelivSegs(sources,targets,start,delivery,DELIVERY0,GRAPHS,WORLD,track=True);
    #         path = DELIV['current_path'];
    #         if i in delivs_to_show:
    #             for j,node in enumerate(path):
    #                 if j < len(path)-1:
    #                     edge = (path[j],path[j+1],0)
    #                     edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
    #                     ONDEMAND['current_edge_masses'][edge] = edge_mass;
    #                     ONDEMAND['current_edge_masses1'][edge] = edge_mass;



    #     ################################# 


    #         # DELIV['active_trips']  = [];

    #     trips_to_plan = WORLD['ondemand']['active_trips_shuttle']
    #     print('...with',len(trips_to_plan),'active shuttle trips...')
    #     DELIVERY0 = DELIVERY['shuttle'];
    #     maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));
    #     divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD);

    #     if show_delivs=='all': delivs_to_show = list(range(len(list(DELIVERY0.keys()))));
    #     elif isinstance(show_delivs,int): delivs_to_show = list(range(show_delivs));
    #     else: delivs_to_show = show_delivs;


    #     for i,delivery in enumerate(DELIVERY0):
    #         if np.mod(i,200)==0: print('shuttle',i,'...')
    #         DELIV = DELIVERY0[delivery];
    #         active_trips = DELIV['active_trips'];
    #         sources = []; targets = [];
    #         start = DELIV['nodes']['source'];        
    #         [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
    #         [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
    #         planDelivSegs(sources,targets,start,delivery,DELIVERY0,GRAPHS,WORLD,track=True);
    #         # DELIV['active_trips']  = [];
    #         path = DELIV['current_path'];          
    #         if i in delivs_to_show:        
    #             for j,node in enumerate(path):
    #                 if j < len(path)-1:
    #                     edge = (path[j],path[j+1],0)
    #                     edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
    #                     ONDEMAND['current_edge_masses'][edge] = edge_mass;
    #                     ONDEMAND['current_edge_masses2'][edge] = edge_mass;
    #     WORLD[mode]['active_trips']  = [];
    #     WORLD[mode]['active_trips_shuttle'] = [];
    #     WORLD[mode]['active_trips_direct'] = [];

    #     ######## REMOVING MASS ON TRIPS ##########
    #     ######## REMOVING MASS ON TRIPS ##########
    #     for i,trip in enumerate(WORLD[mode]['trips']):
    #         TRIP = WORLD[mode]['trips'][trip];
    #         TRIP['mass'] = 0;




    if False:
        ########### -- XX START HERE ---- ####################
        trips_to_plan = WORLD['ondemand']['active_trips_direct']    
        print('...with',len(trips_to_plan),'active direct trips...')
        DELIVERY0 = DELIVERY['direct'];
        maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));

        divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD,maxTrips);
        #######################################################

        if show_delivs=='all': delivs_to_show = list(range(len(list(DELIVERY0.keys()))));
        elif isinstance(show_delivs,int): delivs_to_show = list(range(show_delivs));
        else: delivs_to_show = show_delivs;

        for i,delivery in enumerate(DELIVERY0):
            if np.mod(i,200)==0: print('direct',i,'...')
            DELIV = DELIVERY0[delivery];
            active_trips = DELIV['active_trips'];
            sources = []; targets = [];
            start = DELIV['nodes']['source'];        
            [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
            [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
            planDelivSegs(sources,targets,start,delivery,DELIVERY0,GRAPHS,WORLD,track=True);
            path = DELIV['current_path'];
            if i in delivs_to_show:
                for j,node in enumerate(path):
                    if j < len(path)-1:
                        edge = (path[j],path[j+1],0)
                        edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
                        ONDEMAND['current_edge_masses'][edge] = edge_mass;
                        ONDEMAND['current_edge_masses1'][edge] = edge_mass;



        ################################# 


            # DELIV['active_trips']  = [];

        trips_to_plan = WORLD['ondemand']['active_trips_shuttle']
        print('...with',len(trips_to_plan),'active shuttle trips...')
        DELIVERY0 = DELIVERY['shuttle'];
        maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));
        divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD);

        if show_delivs=='all': delivs_to_show = list(range(len(list(DELIVERY0.keys()))));
        elif isinstance(show_delivs,int): delivs_to_show = list(range(show_delivs));
        else: delivs_to_show = show_delivs;


        for i,delivery in enumerate(DELIVERY0):
            if np.mod(i,200)==0: print('shuttle',i,'...')
            DELIV = DELIVERY0[delivery];
            active_trips = DELIV['active_trips'];
            sources = []; targets = [];
            start = DELIV['nodes']['source'];        
            [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
            [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
            planDelivSegs(sources,targets,start,delivery,DELIVERY0,GRAPHS,WORLD,track=True);
            # DELIV['active_trips']  = [];
            path = DELIV['current_path'];          
            if i in delivs_to_show:        
                for j,node in enumerate(path):
                    if j < len(path)-1:
                        edge = (path[j],path[j+1],0)
                        edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
                        ONDEMAND['current_edge_masses'][edge] = edge_mass;
                        ONDEMAND['current_edge_masses2'][edge] = edge_mass;
        WORLD[mode]['active_trips']  = [];
        WORLD[mode]['active_trips_shuttle'] = [];
        WORLD[mode]['active_trips_direct'] = [];

        ######## REMOVING MASS ON TRIPS ##########
        ######## REMOVING MASS ON TRIPS ##########
        for i,trip in enumerate(WORLD[mode]['trips']):
            TRIP = WORLD[mode]['trips'][trip];
            TRIP['mass'] = 0;



####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 
####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 


def world_of_gtfs(WORLD,PEOPLE,GRAPHS,NODES,verbose=False,clear_active=True):  ##### ADDED TO CLASS 
    if verbose: print('starting gtfs computations...')
    #raptor_gtfs
    kk = WORLD['main']['iter']
    alpha = WORLD['main']['alpha']

    GTFS = WORLD['gtfs'];
    GRAPH =GRAPHS['transit']; 
    FEED = GRAPHS['gtfs'];
    if (not('edge_masses_gtfs' in GTFS.keys()) or (kk==0)):
        GTFS['edge_costs'] = {};
        GTFS['edge_masses'] = {};
        GTFS['current_edge_costs'] = {};
        GTFS['edge_a0'] = {};
        GTFS['edge_a1'] = {};        
        for e,edge in enumerate(GRAPH.edges):
            GTFS['edge_masses'][edge] = [0]
            GTFS['edge_costs'][edge] = [1]
            GTFS['current_edge_costs'][edge] = 1;
            GTFS['edge_a0'][edge] = 1;
            GTFS['edge_a1'][edge] = 1;               
        
    if 'current_edge_costs' in GTFS.keys():
        current_costs = GTFS['current_edge_costs'];
    else: 
        current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}
        
    #nx.set_edge_attributes(GRAPH,current_costs,'c');

    mode = 'gtfs'
    removeMassFromEdges(mode,WORLD,GRAPHS) 
    segs = WORLD[mode]['active_trips']
    print('...with ',len(segs),' active trips...')        

    for i,seg in enumerate(WORLD[mode]['trips']):
    # for i,seg in enumerate(segs):
        if seg in segs:
            source = seg[0];
            target = seg[1];
            trip = (source,target)
            mass = WORLD[mode]['trips'][trip]['mass']
            planGTFSSeg((source,target),mode,GRAPHS,WORLD,NODES,mass=mass,track=True);
            WORLD[mode]['trips'][trip]['active'] = False;
        else:
            if len(WORLD[mode]['trips'][seg]['costs']['time'])==0:
                WORLD[mode]['trips'][seg]['costs']['time'].append(0)
            else:
                recent_value = WORLD[mode]['trips'][seg]['costs']['time'][-1]
                WORLD[mode]['trips'][seg]['costs']['time'].append(recent_value)


    if clear_active:
        WORLD[mode]['active_trips']  = [];

        ######## REMOVING MASS ON TRIPS ##########
        ######## REMOVING MASS ON TRIPS ##########
        print('REMOVING MASS FROM GTFS TRIPS...')
        for i,trip in enumerate(WORLD[mode]['trips']):
            TRIP = WORLD[mode]['trips'][trip];
            TRIP['mass'] = 0;



def world_of_transit(WORLD,PEOPLE,GRAPHS,verbose=False):   ###### ADDED TO CLASS 
    # uses GTFS feed
    if verbose: print('starting transit computations...')
    #raptor_gtfs
    TRANSIT = WORLD['transit'];
    GRAPH = GRAPHS[TRANSIT['graph']];

    ########################################################################################
    #### --- NOT SURE WE NEED THIS>>>> 
    if not('edge_masses' in TRANSIT.keys()):
        TRANSIT['edge_masses'] = {};
        TRANSIT['edge_costs'] = {};
        TRANSIT['current_edge_costs'] = {};
        TRANSIT['edge_a0'] = {};
        TRANSIT['edge_a1'] = {};        
        for e,edge in enumerate(GRAPH.edges):
            TRANSIT['edge_masses'][edge] = [0]
            TRANSIT['edge_costs'][edge] = [1]
            TRANSIT['current_edge_costs'][edge] = 1;
            TRANSIT['edge_a0'][edge] = 1;
            TRANSIT['edge_a1'][edge] = 1;               

    if 'current_edge_costs' in TRANSIT.keys():
        current_costs = TRANSIT['current_edge_costs'];
    else: 
        current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}        
    nx.set_edge_attributes(GRAPH,current_costs,'c');
    ########################################################################################
    

    mode = 'transit'
    removeMassFromEdges(mode,WORLD,GRAPHS) 
    segs = WORLD[mode]['active_trips']
    for i,seg in enumerate(segs):
        source = seg[0];
        target = seg[1];
        trip = (source,target)
        planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=1,track=True);
        WORLD[mode]['trips'][trip]['active'] = False;    
    WORLD[mode]['active_trips']  = [];    

def world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=False):  ### ADDED TO CLASS 
    if verbose: print('starting transit computations...')
    #raptor_gtfs
    TRANSIT = WORLD['transit'];
    GRAPH = GRAPHS[TRANSIT['graph']];
    if not('edge_masses' in TRANSIT.keys()):
        TRANSIT['edge_masses'] = {};
        TRANSIT['edge_costs'] = {};
        TRANSIT['current_edge_costs'] = {};
        TRANSIT['edge_a0'] = {};
        TRANSIT['edge_a1'] = {};        
        for e,edge in enumerate(GRAPH.edges):
            TRANSIT['edge_masses'][edge] = [0]
            TRANSIT['edge_costs'][edge] = [1]
            TRANSIT['current_edge_costs'][edge] = 1;
            TRANSIT['edge_a0'][edge] = 1;
            TRANSIT['edge_a1'][edge] = 1;               
        
    if 'current_edge_costs' in TRANSIT.keys():
        current_costs = TRANSIT['current_edge_costs'];
    else: 
        current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}
        
    nx.set_edge_attributes(GRAPH,current_costs,'c');
    

    mode = 'transit'
    removeMassFromEdges(mode,WORLD,GRAPHS) 
    segs = WORLD[mode]['active_trips']
    print('...with ',len(segs),' active trips...')        
    for i,seg in enumerate(segs):
        source = seg[0];
        target = seg[1];
        trip = (source,target)
        mass = WORLD[mode]['trips'][trip]['mass']
        planDijkstraSeg((source,target),mode,GRAPHS,WORLD,mass=mass,track=True);
        WORLD[mode]['trips'][trip]['active'] = False;    
    WORLD[mode]['active_trips']  = [];




###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand 
###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand 


####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 
####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS ####### OLD BUS FUNCTIONS 


#### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES 
#### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES 
#### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES #### (OLD) BUS STOP NODES 

#### OLD FUNCTIONS.... 

def bus_stop_nodes(feed, graph):
    BUS_STOP_NODES = {};
    stop_ids = list(feed.stops.stop_id)
    df = feed.stops.set_index('stop_id');
    for i,stop_id in enumerate(stop_ids):
        lat = df.loc[stop_id].stop_lat
        lon = df.loc[stop_id].stop_lon
        orig_node = ox.distance.nearest_nodes(graph, lon,lat); # ORIG_LOC[i][0], ORIG_LOC[i][1]);
        xx = graph.nodes[orig_node]['x']
        yy = graph.nodes[orig_node]['y']
        if (np.abs(xx-lon) + np.abs(yy-lat) <= 0.0001):
            BUS_STOP_NODES[stop_id] = orig_node;
    return BUS_STOP_NODES


def bus_stop_nodes_wgraph(bus_graph, graph):
    BUS_STOP_NODES = {};
    nodes = list(bus_graph.nodes)
    for i,node in enumerate(bus_graph.nodes):
        lon = bus_graph.nodes[node]['x'];
        lat = bus_graph.nodes[node]['y'];
        orig_node = ox.distance.nearest_nodes(graph, lon,lat); #ORIG_LOC[i][0], ORIG_LOC[i][1]);
        xx = graph.nodes[orig_node]['x']
        yy = graph.nodes[orig_node]['y']
        if (np.abs(xx-lon) + np.abs(yy-lat) <= 0.1):
            BUS_STOP_NODES[node] = orig_node;
        else:
            continue; #print('no node found in other graph for bus stop ',node)
    return BUS_STOP_NODES


##### COMPUTING CLOSEST NODES ON BUS LINES 
def bus_connection_nodes(graph,node,bus_stop_nodes,eps=10):
    connection_nodes = {}
    all_bus_stops = [];
    dists = [];
    tags = [];
    for i,bus_stop_tag in enumerate(bus_stop_nodes):
        dest = bus_stop_nodes[bus_stop_tag]
        orig = node
        try:
            bus_stop_added = False;
            path = nx.shortest_path(graph, source=orig, target=dest, weight=None);
            dist = nx.shortest_path_length(graph, source=orig, target=dest, weight=None);
            if dist < eps:
               connection_nodes[bus_stop_tag] = {}
               connection_nodes[bus_stop_tag]['dist'] = dist;
               connection_nodes[bus_stop_tag]['path'] = path;
               if not(bus_stop_tag in all_bus_stops):
                    all_bus_stops.append(bus_stop_tag);
                    
        except:
            continue
    return connection_nodes,ordering #,all_bus_stops

recompute_close_nodes = True;
recompute_close_nodes = True;
recompute_connection_nodes = True;
recompute_foot_transfers = True;





###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 
###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT ###### PLOT 



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
    

def plotODs(GRAPHS,SIZES,NODES,scale=1.,figsize=(10,10)):
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
    
        fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=bgcolor,  
                                node_color=node_colors,
                                node_size = node_sizes,
                                edge_color=edge_colors,
                                edge_linewidth=edge_widths,
                                figsize=figsize,
                                show=False,); #file_format='svg')


def plotShapesOnGraph(GRAPHS,shapes,figsize=(10,10)):
    fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=[0.8,0.8,0.8,0.8],
                             node_color = [1,1,1,1],node_size = 1,
                             edge_color = [1,1,1,1],
                             edge_linewidth = 2,
                             figsize = figsize,
                             show = False);
    for shape in shapes:
        plotCvxHull(ax,shape,params = {});
        # group_polygons = [np.random.rand(10,2) for i in range(5)];


def sortList(list1,inds):
    outs = [list1[ind] for ind in inds];
    return outs

def sortListByList(sortlist,bylist):
    inds = np.argsort(bylist)
    sorted_list = [];
    sorted_bylist = [];
    for ind in inds:
        sorted_list.append(sortlist[ind])
        sorted_bylist.append(bylist[ind])
    return inds,sorted_list,sorted_bylist



def getSortedEdgeData(typ,params,sortby=[],mode='drive',use_all_trips=False): pass 

def getSortedPersonData(factor,params,segtypes=[],use_uncong=False,use_all=False,use_current=True):
    WORLD = params['WORLD']
    GRAPHS = params['GRAPHS']
    DELIVERY = params['DELIVERY']
    PEOPLE = params['PEOPLE'];
    OUTS = {};        
    for person in PEOPLE:
        PERSON = PEOPLE[person]
        ind = PERSON['current_choice']
        if len(segtypes)>0: trip = segtypes;
        else: trip = PERSON['trips_to_consider'][ind]
        if trip in PERSON['trips_to_consider']:
            STRUCTURE = PERSON['trips'][trip]['structure'];
            cost = 0;
            for SEG in STRUCTURE:
                mode = SEG['mode']
                start = SEG['opt_start']; end = SEG['opt_end']
                seg = (start,end);
                try: 
                    if use_uncong: new_cost = WORLD[mode]['trips'][seg]['uncongested']['costs'][factor];
                    else: new_cost = WORLD[mode]['trips'][seg]['costs']['current_' + factor];
                    if new_cost < 10000000: cost = cost + new_cost 
                except: pass;
            OUTS[person] = cost
    return OUTS


def getSortedTripData(mode,factor,params,sortby=[],use_all=False,use_current=True):
    WORLD = params['WORLD']
    GRAPHS = params['GRAPHS']
    DELIVERY = params['DELIVERY']

    if use_all: active_trips = list(WORLD[mode]['trips'])
    else: active_trips = list(WORLD[mode]['active_trips'])

    TRIPS = WORLD[mode]['trips'];
    trips  = []; costs = []; sortbyvals = [];
    for i,trip in enumerate(active_trips):
        TRIP = TRIPS[trip]
        if use_current: cost = TRIP['costs']['current_'+factor];
        else: cost = TRIP['costs'][factor][-1];

        if len(sortby)>0:
            if use_current: sortbyval = TRIP['costs']['current_'+sortby];
            else: sortbyval = TRIP['costs'][factor][-1];

        if cost < 10000000:
            trips.append(trip)
            costs.append(cost)
            if len(sortby)>0: sortbyvals.append(sortbyval)

    if len(sortby)>0:
        inds = np.argsort(sortbyvals)
        sorted_trips = [trips[ind] for ind in inds];
        sorted_costs = [costs[ind] for ind in inds];
        sorted_sortbyvals = [sortbyvals[ind] for ind in inds];
    else: 
        sorted_trips = trips
        sorted_costs = costs;
        sorted_sortbyvals = [];
        inds = [];
    return sorted_trips,sorted_costs,sorted_sortbyvals,inds



# def computeData(WORLD,GRAPHS,DELIVERY,factor='time',use_all_trips=False):
#     DATA = {'drive':''};
#     loop through people
#     loop through drive_trips
#     loop through walk__trips
#     loop through ondemand_trips
#     loop through gtfs_trips
#     return None


def loadDataRuns(folder,filenames,GRAPHS):
    DATA = {};
    for dataname in filenames:
        filename = filenames[dataname]
        file = open(folder+filename,'rb');
        data = pickle.load(file)
        file.close();
        DATA[dataname] = computeData(data['WORLD'],GRAPHS,data['DELIVERY']);
    return DATA;

def computeData(WORLD,GRAPHS,DELIVERY):

    colheaders = ['people']
    colheaders = colheaders + ['time','money','switches','conven'];
    colheaders = colheaders + ['time0','money0','switches0','conven0'];
    colheaders = colheaders + ['start_node','end_node'];
    colheaders = colheaders + ['active']
    
    colheaders2 = colheaders + ['runid','group','num_passengers']
    COLHEADERS = {tag:[] for tag in colheaders};
    COLHEADERS2 = {tag:[] for tag in colheaders2};

    DATA = {}; #'people':{},'drive':{},'walk':{},'ondemand':{},'gtfs':{}}
    DATA['drive'] = {'DF': pd.DataFrame(COLHEADERS,index=[])}
    DATA['walk'] = {'DF': pd.DataFrame(COLHEADERS,index=[])}
    DATA['gtfs'] = {'DF': pd.DataFrame(COLHEADERS,index=[])}    
    DATA['ondemand'] = {'DF' : pd.DataFrame(COLHEADERS2,index=[])}

    for _,mode in enumerate(['drive','walk','ondemand','gtfs']):
        DF = DATA[mode]['DF'];
        active_trips = WORLD[mode]['active_trips'];
        all_trips = list(WORLD[mode]['trips']);
        TRIPS = WORLD[mode]['trips']

        for trip in TRIPS:
            TRIP = TRIPS[trip];
            
            NEWDATA = {};
            if 'people' in TRIP: NEWDATA['people'] = TRIP['people']
            else: NEWDATA['people'] = [None];
                
            NEWDATA['start_node'] = [trip[0]];
            NEWDATA['end_node'] = [trip[1]];

            if trip in active_trips: NEWDATA['active'] = True;
            else: NEWDATA['active'] = False;

            NEWDATA['time'] = [TRIP['costs']['current_time']]
            NEWDATA['money'] = [TRIP['costs']['current_money']]
            NEWDATA['switches'] = [TRIP['costs']['current_switches']]
            NEWDATA['conven'] = [TRIP['costs']['current_conven']]
            if 'uncongested' in TRIP:
                NEWDATA['time0'] = [TRIP['uncongested']['costs']['time']]
                NEWDATA['money0'] = [TRIP['uncongested']['costs']['money']]
            else: 
                NEWDATA['time0'] = NEWDATA['time']
                NEWDATA['money0'] = NEWDATA['money']
            
            if mode == 'ondemand':
                if 'group' in TRIP: NEWDATA['group'] = [TRIP['group']];
                if 'run_id' in TRIP: NEWDATA['runid'] = [TRIP['run_id']];
                if 'num_passengers' in TRIP: NEWDATA['num_passengers'] = [TRIP['num_passengers']];

            NEWDF = pd.DataFrame(NEWDATA)
            DF = pd.concat([DF,NEWDF], ignore_index = True)            
        DATA[mode]['DF'] = DF;
    return DATA




def sortingData(WORLD,GRAPHS,DELIVERY,factor='time',use_all_trips=False):
    modes = ['drive','walk','gtfs','ondemand']
    maxcosts = {'drive': 10000,'walk': 10000,'gtfs': 1000000000,'ondemand': 10000}
    tripsSorted = {}
    total_costs = [];
    total_costs0 = [];

    try: 
        for _,group in enumerate(DELIVERY['groups']):
            payload = DELIVERY['groups'][group]['current_payload'];
            manifest = DELIVERY['groups'][group]['current_manifest'];
            PDF = payloadDF(payload,GRAPHS,include_drive_nodes = True);
            MDF = manifestDF(manifest,PDF)
            # ZZ = MDF[MDF['run_id']==1]
            # zz = list(ZZ.sort_values(['scheduled_time'])['drive_node'])
            PLANS = routesFromManifest(MDF,GRAPHS)
            # PATHS = routeFromStops(zz,GRAPHS['drive'])
            ONDEMAND_EDGES[group] = [PLANS[plan]['edges'] for _,plan in enumerate(PLANS)]
            # DELIVERY_TAGS[group] = []
            for i,plan in enumerate(PLANS):
                tags.append(group+'_delivery'+str(i))
    except:
        pass

    ONDEMAND_TRIPS = {};
    for _,group in enumerate(DELIVERY['groups']):
        if 'current_payload' in DELIVERY['groups'][group]:
            payload = DELIVERY['groups'][group]['current_payload'];
            manifest = DELIVERY['groups'][group]['current_manifest'];
            PDF = payloadDF(payload,GRAPHS,include_drive_nodes = True);
            MDF = manifestDF(manifest,PDF)
            PLANS = singleRoutesFromManifest(MDF,GRAPHS)
            ONDEMAND_TRIPS[group] = [];
            for runid in PLANS:
                PLAN = PLANS[runid];
                for trip in PLAN:
                    ONDEMAND_TRIPS[group].append(trip)


    ########################################################################################
    ########################################################################################
    ########################################################################################
    ########################################################################################


    factors = ['time','money','conven','switches']

    for m,mode in enumerate(modes):
        tripsSorted[mode] = {}

        if use_all_trips:
            active_trips = list(WORLD[mode]['trips'])
        else: 
            active_trips = list(WORLD[mode]['active_trips'])

        if mode == 'ondemand':
            tripsSorted[mode]['groups'] = {};
    
        active_costs = [];
        active_costs0 = [];


        ACTIVE_COSTS = {'time':[],'money':[],'conven':[],'switches':[]};

        for i,trip in enumerate(active_trips):
            TRIP = WORLD[mode]['trips'][trip]
            # print(TRIP.keys())
            # cost0 = 100000000;

            if mode == 'ondemand':
                cost = TRIP['costs'][factor][-1]
                if 'uncongested' in TRIP: cost0 = TRIP['uncongested']['costs'][factor]
                else: cost0 = 10000000;
                if cost < 100000 and cost0 < 100000:
                    active_costs.append(TRIP['costs'][factor][-1])
                    active_costs0.append(cost0)

            elif mode == 'gtfs':
                cost = TRIP['costs'][factor][-1]
                if cost < 100000:
                    active_costs.append(cost)
                    active_costs0.append(cost)

            else:
                cost = TRIP['costs']['current_'+factor];
                if 'uncongested' in TRIP: cost0 = TRIP['uncongested']['costs'][factor]
                else: cost0 = 10000000;
                if cost < 1000000 and cost0 < 1000000:
                    active_costs.append(cost)
                    active_costs0.append(cost0)
    
        total_costs = total_costs + active_costs;    
        total_costs0 = total_costs0 + active_costs0;
        active_costs = np.array(active_costs);
        active_costs0 = np.array(active_costs0);
        inds = np.argsort(active_costs);
        active_costs = active_costs[inds]
        active_costs0 = active_costs0[inds]
        active_trips = [active_trips[ind] for i,ind in enumerate(inds)];

        if mode == 'ondemand':
            for _,group in enumerate(DELIVERY['groups']):
                tripsSorted[mode]['groups'][group] = {'costs': [], 'trips':[],'overall_counts':[],'group_counts':[],'runs':{},
                                                      'costs0':[]}
            # for i,trip in enumerate(active_trips):
            for i,trip in enumerate(active_trips): #ONDEMAND_TRIPS[group]):
                # try: 
                TRIP = WORLD[mode]['trips'][trip]
                group = TRIP['group'];
                cost = active_costs[i]
                cost0 = active_costs0[i];
                tripsSorted[mode]['groups'][group]['costs'].append(cost)
                tripsSorted[mode]['groups'][group]['costs0'].append(cost0)
                tripsSorted[mode]['groups'][group]['trips'].append(trip)
                tripsSorted[mode]['groups'][group]['overall_counts'].append(i)

                len1 = len(tripsSorted[mode]['groups'][group]['group_counts']);
                tripsSorted[mode]['groups'][group]['group_counts'].append(len1)

                if 'run_id' in TRIP:
                    runid = TRIP['run_id'];
                    if not(runid in tripsSorted[mode]['groups'][group]['runs']):
                        tripsSorted[mode]['groups'][group]['runs'][runid] = {'costs0':[],'costs': [], 'trips':[],'overall_counts':[],'run_counts':[]};

                    tripsSorted[mode]['groups'][group]['runs'][runid]['costs'].append(cost)
                    tripsSorted[mode]['groups'][group]['runs'][runid]['costs0'].append(cost0)
                    tripsSorted[mode]['groups'][group]['runs'][runid]['trips'].append(trip)
                    tripsSorted[mode]['groups'][group]['runs'][runid]['overall_counts'].append(len1)
                    len1 = len(tripsSorted[mode]['groups'][group]['runs'][runid]['run_counts']);
                    tripsSorted[mode]['groups'][group]['runs'][runid]['run_counts'].append(len1)
                # except:
                #     pass;

    
        trips = WORLD[mode]['trips'];
        not_active_trips = []
        costs = [];
        for i,trip in enumerate(trips):
            if not(trip in active_trips):
                not_active_trips.append(trip)
                TRIP = WORLD[mode]['trips'][trip]
                if mode == 'ondemand':
                    cost = TRIP['costs'][factor][-1]
                    if cost < 100000:
                        costs.append(TRIP['costs'][factor][-1])
                else:
                    cost = TRIP['costs']['current_'+factor]
                    if cost < 100000:
                        costs.append(cost)
        #total_costs = total_costs + costs;            
        costs = np.array(costs);
        inds = np.argsort(costs);
        costs = costs[inds]
        not_active_trips = [not_active_trips[ind] for i,ind in enumerate(inds)];
    
        tripsSorted[mode]['costs'] = list(active_costs);# + list(costs);
        tripsSorted[mode]['costs0'] = list(active_costs0);# + list(costs);
        tripsSorted[mode]['counts'] = list(range(len(active_costs)));# + list(costs);

        tripsSorted[mode]['trips'] = active_trips; # + not_active_trips;
        
    maxcost = np.max(total_costs)
    

    ###############################################################
    ###############################################################
    ###############################################################
    ###############################################################


    convergeSorted = {};
    itr = int(WORLD['main']['iter'])+1
    for m,mode in enumerate(modes):
    
        convergeSorted[mode] = {}
        convergeSorted[mode]['cost_trajs'] = [];
        for tt,trip in enumerate(WORLD[mode]['trips']):#active_trips):
            
            
            
            costs = WORLD[mode]['trips'][trip]['costs'][factor]
            costs = np.array(costs)
            costs = costs[np.where(costs<maxcosts[mode])]  
    
            if mode == 'ondemand':
                costs = costs[-itr:]
            
            try:
                if len(costs)>0:
                    if costs[-1]<1000000.:
                        convergeSorted[mode]['cost_trajs'].append(costs.copy())
            except:
                pass
        if mode == 'ondemand':
            convergeSorted[mode]['expected_cost'] = WORLD[mode]['expected_cost'];


    OUT = {};
    OUT['trips_sorted'] = tripsSorted;
    OUT['converge_sorted'] = convergeSorted;
    return OUT

def generateLayers(GRAPHS,NODES,SIZES,DELIVERY,WORLD,params,use_all_trips=False):

    print('starting to create plotting layers...')
    start_time = time.time()
    bgcolor = params['bgcolor'];

    node_scale = 100.;    
    
    lims = [0,1,0,1]
    

    shows = {'drive': False,
             'walk': False,
             'transit':False,
             'ondemand':False,
             'direct':False,
             'shuttle':False,
             'ondemand_indiv':False,
             'lines':False,
             'gtfs':False,
             'source':False,
             'target':True,
             'legend':False,
             'base':False}


    maxwids = {'drive': 4.,'walk':2,'transit':6,'lines':4,'gtfs':8,'ondemand':2,'direct':1,'ondemand_indiv':10,'base':4.}
    minwids = {'drive': 4.,'walk':2,'transit':6,'lines':4,'gtfs':8,'ondemand':2,'direct':1,'ondemand_indiv':10,'base':4.}
    # mxpop1 = 1.
    mxpops = {'drive': 0.3,'walk':0.01,'transit':0.001,'lines':0.00001,'gtfs':0.001,'ondemand':0.001}


    
    # params['colors']['shuttle'] = [0.,0.,1.]
    # params['set_alphas'] = {'direct':0.6,'shuttle':0.6}
    # params['set_wids'] = {'direct':4,'shuttle':4}    


    colors = {'drive':[1,0,0],'walk':[1.,1.,0.],#[0.7,0.7,0.7],
              'lines':[0.,0.,0.],'transit':[1.,0.,1.],'gtfs': [1.,0.5,0.],   
              'ondemand':[0.,0.,1.],'direct':[0.6,0.,1.],'shuttle':[0.,0.,1.],          
              'source':[0.,0.,1.,0.5],'target':[1.,0.,0.,0.5], #[0.8,0.8,0.8],
              'groups':[[0.,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5]],
              'shuttle_nodes':[1.,0.5,0.,0.5],
              'ondemand_indiv':[1.,0.,1.0],
              'ondemand1':[0.,0.,1.0],
              'ondemand2':[1.,0.,1.],
              'ondemand_trips1':[0.,0.,0.,1.],
              'default_edge':[1,1,1,1]}

    colors2 = {'drive_all':[1,0,0,0.3],'walk_all':[1,1.,0,1.],'gtfs_all':[1,0.5,0,0.8],'ondemand_all':[0.,0.,1.,0.5],
               'drive_trips':[0,0,0,1.],'walk_trips':[1.,1.,0,1.],'gtfs_trips':[1.,0.5,0.,1.],'ondemand_trips':[0.,0.,0.,1.0]}

    colorB = np.array([0.,0.,1.])
    colorA = np.array([0.6,0.,1.]); 
    if len(list(DELIVERY['groups']))>0:groups = list(DELIVERY['groups'])
    else: groups = ['group0']
    colors2['groups'] = {};
    if len(groups)<=1: ngrps = 1;
    else: ngrps = len(groups)-1;
    for i,group in enumerate(groups):
        colorAB = (float(i)/ngrps)*colorA + (1.-(float(i)/ngrps))*colorB
        colors2['groups'][group] = [colorAB[0],colorAB[1],colorAB[2],0.5];



    sizes = {'source':100,'target':100,'shuttle':300,'direct':300,'gtfs':5}
    node_rads = {'drive':100.,'walk':100.,'ondemand':100.,'gtfs':100.};
    node_edgecolors = {'lines':[0,0,0,1],'source':[0,0,0,1],'target':[0,0,0],'shuttle':'k','direct':'k','gtfs':'k'};
    # mxpop1 = 1.; #num_sources/10;
    threshs = {'drive': [0,0,mxpops['drive'],1],'walk': [0,0,mxpops['walk'],1],
               'transit': [0,0,mxpops['transit'],1],'gtfs': [0,0,mxpops['gtfs'],1],'ondemand': [0,0,mxpops['ondemand'],1]}


    #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP 
    #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP 
    #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP #### SETUP 

    tags = ['base','lines','source','target','drive','walk','ondemand','gtfs'];


    

    # def edgesFromPath(path):
    # """ computes edges in a path from a list of nodes """
    # edges = []; # initializing list
    # for i,node in enumerate(path): # loops through nodes in path
    #     if i<len(path)-1: # if not the last node
    #         node1 = path[i]; node2 = path[i+1]; 
    #     edges.append((node1,node2)) # add an edge tag defined as (node1,node2) which is the standard networkx structure
    # return edges


    ONDEMAND_EDGES = {};
    ONDEMAND_RUNS = {};
    # DELIVERY_TAGS = {};
    ondemand_run_tags = [];
    try: 
        for _,group in enumerate(DELIVERY['groups']):
            tags.append(group);
            payload = DELIVERY['groups'][group]['current_payload'];
            manifest = DELIVERY['groups'][group]['current_manifest'];
            PDF = payloadDF(payload,GRAPHS,include_drive_nodes = True);
            MDF = manifestDF(manifest,PDF)
            ZZ = MDF[MDF['run_id']==1]
            zz = list(ZZ.sort_values(['scheduled_time'])['drive_node'])
            PLANS = routesFromManifest(MDF,GRAPHS)
            # PATHS = routeFromStops(zz,GRAPHS['drive'])
            ONDEMAND_EDGES[group] = [PLANS[plan]['edges'] for _,plan in enumerate(PLANS)]
            # DELIVERY_TAGS[group] = []
            ONDEMAND_RUNS[group] = {};
            for _,plan in enumerate(PLANS):
                PLAN = PLANS[plan]
                ONDEMAND_RUNS[group][plan] = PLAN['edges']
                runtag = group + '_run' + str(plan);
                ondemand_run_tags.append(runtag)
            for i,plan in enumerate(PLANS):
                tags.append(group+'_delivery'+str(i))

    except:
        pass


    ONDEMAND_TRIPS = {};
    ondemand_trip_tags = [];

    try: 
        for _,group in enumerate(DELIVERY['groups']):
            tags.append(group);
            payload = DELIVERY['groups'][group]['current_payload'];
            manifest = DELIVERY['groups'][group]['current_manifest'];
            PDF = payloadDF(payload,GRAPHS,include_drive_nodes = True);
            MDF = manifestDF(manifest,PDF)
            PLANS = singleRoutesFromManifest(MDF,GRAPHS)
            # PATHS = routeFromStops(zz,GRAPHS['drive'])
            # ONDEMAND_RUNS[group] = {};
            for runid in PLANS:
                PLAN = PLANS[runid];
                for trip in PLAN:
                    TRIP = PLAN[trip];
                    node1 = trip[0]; node2 = trip[1];
                    # tag = 'ondemand'+'_'+group+'_'+'run'+str(runid)+'_'+str(node1)+'_'+str(node2);
                    # ONDEMAND_TRIPS[tag] = TRIP['edges'];
                    # ondemand_trip_tags.append(tag);
                    # tags.append(tag)
                    tag = 'ondemand'+'_'+group+'_'+str(node1)+'_'+str(node2);
                    ONDEMAND_TRIPS[tag] = TRIP['edges'];
                    ondemand_trip_tags.append(tag);

    except:
        pass        


    print(ondemand_run_tags)

    ACTIVE_TRIPS = {}
    for mode in ['drive','walk','gtfs','ondemand']:
        if use_all_trips: active_trips = list(WORLD[mode]['trips'])
        else: active_trips = WORLD[mode]['active_trips']
        ACTIVE_TRIPS[mode] = active_trips.copy();

    walk_trip_tags = []; drive_trip_tags = []; gtfs_trip_tags = [];
    TRIP_EDGES = {};
    for mode in ['drive','walk','gtfs']:
        # print(mode)
        TRIP_EDGES[mode] = {}
        active_trips = ACTIVE_TRIPS[mode]
        for trip in active_trips:
            tag = mode + '_' + str(int(trip[0])) + '_' + str(int(trip[1]))
            TRIP = WORLD[mode]['trips'][trip];
            edges = edgesFromPath(TRIP['current_path'])
            TRIP_EDGES[mode][tag] = edges;
            tags.append(tag);
            if mode == 'walk': walk_trip_tags.append(tag);
            if mode == 'drive': drive_trip_tags.append(tag);
            if mode == 'gtfs': gtfs_trip_tags.append(tag);



    all_tags = tags + ondemand_trip_tags + ondemand_run_tags;

    nodes = {tag:[] for i,tag in enumerate(all_tags)}
    edges = {tag:[] for i,tag in enumerate(all_tags)}
    node_colors = {tag:[] for i,tag in enumerate(all_tags)}
    edge_colors = {tag:[] for i,tag in enumerate(all_tags)}
    node_sizes = {tag:[] for i,tag in enumerate(all_tags)}
    edge_widths = {tag:[] for i,tag in enumerate(all_tags)}
    node_edge_colors = {tag:[] for i,tag in enumerate(all_tags)};


    #### BASE LAYERS #####
    basic_layer = {'graph':'all','bgcolor':[0.8,0.8,0.9,0.7],
                  'node_colors':[1,1,1,1],'node_sizes':10,'node_edge_colors':[0,0,0,1],
                  'edge_colors':[1,1,1,1],'edge_widths':10}

    #### ACTIVE TRIPS 
    start_nodes = {'drive':[],'walk':[],'ondemand':[],'gtfs':[]}
    end_nodes = {'drive':[],'walk':[],'ondemand':[],'gtfs':[]}

    for m,mode in enumerate(start_nodes):
        for i,trip in enumerate(ACTIVE_TRIPS[mode]):
            start_nodes[mode].append(trip[0])
            end_nodes[mode].append(trip[1]) 

    FGRAPH = GRAPHS['all']


    # for i,name in enumerate(['drive','walk','ondemand','gtfs']):
    #     nodes[name] = start_nodes[name]+end_nodes[name];


    
    # # for i,tag in enumerate(['lines','walk','ondemand','gtfs','sources','targets']):
    # for i,tag in enumerate(['lines','source','target','drive','walk','ondemand','gtfs']):
    #     nodes[tag] = list(FGRAPH.nodes)
    #     edges[tag] = list(FGRAPH.edges)
    #     nn = len(nodes[tag])
    #     ne = len(edges[tag])
    #     node_colors[tag] = np.outer(np.ones(nn),np.array([1,1,1,1]))
    #     edge_colors[tag] = np.outer(np.ones(ne),np.array([1,1,1,1]))
    #     node_edge_colors[tag] = np.outer(np.ones(ne),np.array([1,1,1,1]))
    #     node_sizes[tag] = 10.*np.ones(nn);
    #     edge_widths[tag] = 10.*np.ones(ne);

    #### PRESETS #### PRESETS #### PRESETS #### PRESETS #### PRESETS 
    #### PRESETS #### PRESETS #### PRESETS #### PRESETS #### PRESETS 
    preset_tags = []
    for i,tag in enumerate(preset_tags):
        nodes[tag] = list(FGRAPH.nodes)
        nn = len(nodes[tag])
        node_colors[tag] = np.outer(np.ones(nn),np.array([1,1,1,1]))
        node_edge_colors[tag] = np.outer(np.ones(nn),np.array([1,1,1,1]))
        node_sizes[tag] = 1.*np.ones(nn);
        
    preset_tags = []
    for i,tag in enumerate(preset_tags):
        edges[tag] = list(FGRAPH.edges)
        ne = len(edges[tag])        
        edge_colors[tag] = np.outer(np.ones(ne),np.array([1,1,1,1]))
        edge_widths[tag] = 1.*np.ones(ne);





    ################ ------ ################ ------ ################ ------ ################ ------ ################ ------ 
    ################ ------ ################ ------ ################ ------ ################ ------ ################ ------ 
    ################ ------ ################ ------ ################ ------ ################ ------ ################ ------ 
    ################ ------ ################ ------ ################ ------ ################ ------ ################ ------ 

    edge_color = [];
    edge_width = [];
    drive_edges = list(GRAPHS['drive'].edges())
    ondemand_edges = list(GRAPHS['ondemand'].edges())
    walk_edges = list(GRAPHS['walk'].edges())
    transit_edges = list(GRAPHS['transit'].edges())

    DRIVE = WORLD['drive'];
    TRANSIT = WORLD['transit'];
    ONDEMAND = WORLD['ondemand'];
    WALK = WORLD['walk'];

    if 'other' in params:
        other_nodes = params['other']['nodes'];
        other_sizes = params['other']['sizes'];
        other_color = params['other']['color'];
    else:
        other_nodes = []
        other_sizes = [];
        other_color = [0,0,0,1];

    
    # #include_graphs = {'ondemand':False,'drive':False,'transit':False,'walk':True};
    # plot_graphs = [];
    # for _,mode in enumerate(include_graphs):
    #     if include_graphs[mode]==True:
    #         plot_graphs.append(GRAPHS[mode])
    # full_graph = nx.compose_all(plot_graphs);

    ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES 
    ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES 
    ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES 
    ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES ###### ADDING NODES 


    for i,node in enumerate(FGRAPH.nodes):

        if node in NODES['orig']:
            tag = 'source'
            nodes[tag].append(node)
            node_sizes[tag].append(SIZES['home_sizes'][node]*node_scale)
            node_colors[tag].append(colors[tag])
            node_edge_colors[tag].append(node_edgecolors[tag])

        if node in NODES['dest']:
            tag = 'target'
            nodes[tag].append(node)
            node_sizes[tag].append(SIZES['work_sizes'][node]*node_scale)
            node_colors[tag].append(colors[tag])
            node_edge_colors[tag].append(node_edgecolors[tag])

        for m,mode in enumerate(start_nodes):
            if node in start_nodes[mode]:
                tag = mode;
                nodes[tag].append(node)
                node_sizes[tag].append(node_rads[tag])
                node_colors[tag].append(colors[tag])
                node_edge_colors[tag].append([0,0,0,1])


        for m,mode in enumerate(end_nodes):
            if node in end_nodes[mode]:
                tag = mode;
                nodes[tag].append(node)
                node_sizes[tag].append(node_rads[tag])
                node_colors[tag].append(colors[tag])
                node_edge_colors[tag].append([0,0,0,1])


        if node in GRAPHS['transit'].nodes:
            tag = 'lines'
            nodes[tag].append(node)
            node_sizes[tag].append(1);#SIZES['work_sizes'][node]*node_scale)
            node_colors[tag].append(colors[tag])
            node_edge_colors[tag].append(node_edgecolors[tag])

                    
    #         if shows['drive'] or shows['ondemand'] or shows['ondemand_indiv'] or shows['shuttle'] or shows['direct']:
    #             tag = (edge[0],edge[1],0)
    #             edge_mass = WORLD['drive']['current_edge_masses'][tag]
    #             ondemand_mass = WORLD['ondemand']['current_edge_masses'][tag]
    #             ondemand_mass1 = WORLD['ondemand']['current_edge_masses1'][tag]            
    #             ondemand_mass2 = WORLD['ondemand']['current_edge_masses2'][tag]


    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    # ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    
    drive_edges = list(GRAPHS['drive'].edges)
    walk_edges = list(GRAPHS['walk'].edges)
    ondemand_edges = list(GRAPHS['ondemand'].edges)
    transit_edges = list(GRAPHS['transit'].edges)

    transit_edges2 = [];
    for i,edge in enumerate(transit_edges):
        transit_edges2.append((edge[0],edge[1],));


    for k,edgex in enumerate(FGRAPH.edges):        
        edge = (edgex[0],edgex[1]);
        edge2 = (edgex[0],edgex[1],0);

        if edge2 in transit_edges:
            tag = 'lines'
            edges[tag].append(edge2)
            edge_colors[tag].append(colors[tag]+[1])
            edge_widths[tag].append(2); #SIZES['work_sizes'][node]*node_scale)

        if edge2 in drive_edges:
            tag = 'drive'            
            # drive_mass = WORLD['drive']['current_edge_masses'][edge]
            drive_mass = WORLD['drive']['base_edge_masses'][edge2]
            if drive_mass > 0: 
                edges[tag].append(edge2)
                edge_colors[tag].append(colors[tag]+[0.3*thresh(drive_mass,threshs[tag])])
                edge_widths[tag].append(maxwids[tag]*thresh(drive_mass,threshs[tag]))


        if edge2 in walk_edges:
            tag = 'walk'
            walk_mass = WORLD[tag]['current_edge_masses'][edge2]
            if walk_mass > 0: 
                edges[tag].append(edge2)
                edge_colors[tag].append(colors[tag]+[thresh(walk_mass,threshs[tag])])
                edge_widths[tag].append(maxwids[tag]*thresh(walk_mass,threshs[tag]))


        if edge in transit_edges2:
            tag = 'gtfs'            
            gtfs_mass = WORLD[tag]['current_edge_masses'][edge2]
            if gtfs_mass > 0: 
                edges[tag].append(edge2)
                edge_colors[tag].append(colors[tag]+[1])
                edge_widths[tag].append(maxwids[tag]*thresh(gtfs_mass,threshs[tag]))









        for gg,group in enumerate(ONDEMAND_EDGES):
            edge_lists4 = ONDEMAND_EDGES[group];
            for kk,edge_list in enumerate(edge_lists4):
                if edge in edge_list:
                    tag = 'ondemand'
                    edges[tag].append(edgex)
                    edge_colors[tag].append(colors['groups'][gg])
                    edge_widths[tag].append(4.)

                    tag = group;
                    edges[tag].append(edgex)
                    # edge_colors[tag].append(colors['groups'][gg])
                    edge_colors[tag].append(colors2['groups'][group])
                    edge_widths[tag].append(4.)


                    tag = group + '_delivery' + str(kk);
                    edges[tag].append(edgex)
                    # edge_colors[tag].append(colors['groups'][gg][::3]+[1.])
                    edge_colors[tag].append(colors2['groups'][group][::3]+[1.])
                    edge_widths[tag].append(10.)


        for _,trip in enumerate(ONDEMAND_TRIPS):
            edge_list = ONDEMAND_TRIPS[trip];
            # for kk,edge_list in enumerate(edge_lists4):
            if edge in edge_list:
                tag = trip
                edges[tag].append(edgex)
                edge_colors[tag].append(colors['ondemand_trips1'])
                edge_widths[tag].append(10.)

        for _,group in enumerate(ONDEMAND_RUNS):
            RUNS = ONDEMAND_RUNS[group]
            for _,runid in enumerate(RUNS):
                edge_list = RUNS[runid]
                # for kk,edge_list in enumerate(edge_lists4):
                if edge in edge_list:
                    tag = group + '_run' + str(runid)
                    edges[tag].append(edgex)
                    edge_colors[tag].append(colors2['groups'][group])
                    edge_widths[tag].append(10.)





        for mode in TRIP_EDGES:
            TRIPS = TRIP_EDGES[mode]
            for trip_tag in TRIPS:
                edge_list = TRIPS[trip_tag];
            # for kk,trip_tag in enumerate(TRIPS): #edge_list in enumerate(edge_lists4):
            # edge_list = TRIPS[trip_tag]
                # if mode == 'gtfs':
                #     print(trip_tag)
                if edge in edge_list:
                    edges[trip_tag].append(edgex)
                    edge_colors[trip_tag].append(colors2[mode+'_trips'])
                    edge_widths[trip_tag].append(10.)
        
    #     if walk_mass > 0:
    #         tag = 'walk'
    #         edges[tag].append(edge2)
    #         edge_colors[tag].append(colors['walk']+[1])
    #         edge_widths[tag].append(4); #SIZES['work_sizes'][node]*node_scale)


    #     if edge in transit_edges:

    #         if gtfs_mass > 0:
    #             tag = 'gtfs'
    #             edges[tag].append(edgex)
    #             edge_colors[tag].append(colors['gtfs']+[1])
    #             edge_widths[tag].append(maxwids['gtfs']*thresh(gtfs_mass,threshs['gtfs']))

    #         tag = 'lines'
    #         edges[tag].append(edgex)
    #         edge_colors[tag].append(colors[tag]+[1]);
    #         edge_widths[tag].append(maxwids[tag]);

    #     # if len(OTHER_EDGES)>0:
    #     #     for i,group in enumerate(OTHER_EDGES):
    #     #         GROUP = OTHER_EDGES[group]
    #     #         thresh_lims = GROUP['thresh']
    #     #         color = GROUP['color']
    #     #         maxwid = GROUP['maxwid']
    #     #         maxalpha = GROUP['maxalpha']
    #     #         edge2 = (edge[0],edge[1],0)
    #     #         if edge2 in GROUP['edges']:
    #     #             mmass = GROUP['edges'][edge2]    
    #     #             if mmass > 0.:
    #     #                 tag = 'gtfs'
    #     #                 # edges[tag].append(edgex)
    #     #                 # edge_colors[tag].append(colors['gtfs']+[1])
    #     #                 # edge_widths[tag].append(maxwids['gtfs']*thresh(gtfs_mass,threshs['gtfs']))                        
    #     #                 edge_color[-1] = color+[maxalpha*thresh(mmass,thresh_lims)]; #[chopoff(edge_mass,mx1,0,1)])
    #     #                 edge_width[-1] = maxwids['base']*thresh(mmass,thresh_lims);
                

    ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS 
    ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS 
    ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS 
    ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS ##### CONSTRUCTING LAYERS 

    # tags = ['line','source','target','drive','walk','ondemand','gtfs'];
    layers = {};
    for i,name in enumerate(tags):
        if len(nodes[name]) > 0  or len(edges[name]) > 0:
            layers[name] = {}
            layers[name]['name'] = name;
            layers[name]['bgcolor'] = [1,1,1,0];
            layers[name]['nodes'] = nodes[name].copy();
            layers[name]['edges'] = edges[name].copy();
            layers[name]['node_colors'] = node_colors[name].copy();
            layers[name]['edge_colors'] = edge_colors[name].copy();
            layers[name]['node_sizes'] = node_sizes[name].copy();
            layers[name]['edge_widths'] = edge_widths[name].copy();
            layers[name]['node_edge_colors'] = node_edge_colors[name].copy();
            layers[name] = {**basic_layer,**layers[name]}

            if name in drive_trip_tags: layers[name]['subfolder'] = 'drive_trips/'
            if name in walk_trip_tags: layers[name]['subfolder'] = 'walk_trips/'
            if name in gtfs_trip_tags: layers[name]['subfolder'] = 'gtfs_trips/'
            if name in ondemand_trip_tags: layers[name]['subfolder'] = 'ondemand_trips/'

    for i,name in enumerate(ondemand_trip_tags):
        if len(nodes[name]) > 0  or len(edges[name]) > 0:
            layers[name] = {}
            layers[name]['name'] = name;
            layers[name]['bgcolor'] = [1,1,1,0];
            layers[name]['nodes'] = nodes[name].copy();
            layers[name]['edges'] = edges[name].copy();
            layers[name]['node_colors'] = node_colors[name].copy();
            layers[name]['edge_colors'] = edge_colors[name].copy();
            layers[name]['node_sizes'] = node_sizes[name].copy();
            layers[name]['edge_widths'] = edge_widths[name].copy();
            layers[name]['node_edge_colors'] = node_edge_colors[name].copy();
            layers[name] = {**basic_layer,**layers[name]}
            layers[name]['subfolder'] = 'groups/'


    for i,name in enumerate(ondemand_run_tags):
        if len(nodes[name]) > 0  or len(edges[name]) > 0:
            layers[name] = {}
            layers[name]['name'] = name;
            layers[name]['bgcolor'] = [1,1,1,0];
            layers[name]['nodes'] = nodes[name].copy();
            layers[name]['edges'] = edges[name].copy();
            layers[name]['node_colors'] = node_colors[name].copy();
            layers[name]['edge_colors'] = edge_colors[name].copy();
            layers[name]['node_sizes'] = node_sizes[name].copy();
            layers[name]['edge_widths'] = edge_widths[name].copy();
            layers[name]['node_edge_colors'] = node_edge_colors[name].copy();
            layers[name] = {**basic_layer,**layers[name]}
            layers[name]['subfolder'] = 'runs/'




    tag = 'base'

    layers[tag] = {};
    layers[tag]['name'] = tag
    layers[tag]['graph'] = 'all';
    layers[tag]['bgcolor'] = list(np.array([0.7,0.7,0.8,0.9]));
    layers[tag]['nodes'] = list(FGRAPH.nodes)
    layers[tag]['edges'] = list(FGRAPH.edges)
    nn = len(list(FGRAPH.nodes))
    ne = len(list(FGRAPH.edges))
    layers[tag]['node_colors'] = np.outer(np.ones(ne),np.array([1,1,1,1]))
    layers[tag]['node_sizes'] = 0.5*np.ones(nn)
    layers[tag]['node_edge_colors'] = np.outer(np.ones(ne),np.array([1,1,1,1]))
    layers[tag]['edge_colors'] = np.outer(np.ones(ne),np.array([1,1,1,1]))
    layers[tag]['edge_widths'] = 1.*np.ones(nn)
    layers[tag] = {**basic_layer,**layers[tag]}

    tag = 'drive'
    layers[tag]['node_sizes'] = 0.;
    layers[tag]['node_colors'] = [0,0,0,0];
    layers[tag]['node_edge_colors'] = [0,0,0,0];


    end_time = time.time()
    print('time to create layers:',end_time-start_time)
    return layers


def plotLAYERS(GRAPHS,folder,layers):
    print('starting layers...')


    for i,layer in enumerate(layers):
        print('drawing layer: ',layer)
        LAYER = layers[layer]
        plotLayer(GRAPHS,LAYER,folder=folder);
    print('finished.')


def plotLayer(GRAPHS,data,folder=''):

    GRAPH = GRAPHS[data['graph']];

    if 'nodes' in data: nodes = data['nodes'];
    else: nodes = [];
    if 'edges' in data: edges = data['edges'];
    else: edges = []


    SUBGRAPH_NODES = GRAPH.subgraph(nodes);
    SUBGRAPH_EDGES = GRAPH.edge_subgraph(edges);
    SUBGRAPH = nx.compose(SUBGRAPH_NODES,SUBGRAPH_EDGES)

    nn = len(nodes);
    ne = len(edges);

    if 'node_colors' in data: node_colors = data['node_colors'];
    else: node_colors = np.outer(np.ones(nn),np.array([1,1,1,1]));
    if 'edge_colors' in data: edge_colors = data['edge_colors'];
    else: edge_colors = np.outer(np.ones(ne),np.array([1,1,1,1]));

    if 'node_sizes' in data: node_sizes = data['node_sizes'];
    else: node_sizes = 10.0*np.ones(nn);
    if 'edge_widths' in data: edge_widths = data['edge_widths'];
    else: edge_widths = 1.0*np.ones(ne);

    if 'node_edge_colors' in data: node_edge_colors = data['node_edge_colors'];
    else: node_edge_colors = np.outer(np.ones(nn),np.array([1,1,1,1]));

    if 'subfolder' in data: fileName = folder + data['subfolder'] + data['name']+'.png';
    else: fileName = folder +  data['name']+'.png';


    bgcolor = data['bgcolor']
    if 'bgcolor' in data: bgcolor = data['bgcolor'];
    else: bgcolor = [0.8,0.8,0.9,0.9]

    if 'lims' in data: lims = data['lims'];
    #else:


    lims = [-85.340921162, -85.246387138, 34.98229815, 35.06743515]


    NODE_COLORS = node_colors;
    NODE_SIZES = node_sizes;
    EDGE_COLORS = edge_colors;
    EDGE_WIDTHS = edge_widths;
    NODE_EDGECOLORS = node_edge_colors;

    

    NODE_COLORS = [];
    NODE_SIZES = [];
    EDGE_COLORS = [];
    EDGE_WIDTHS = [];
    NODE_EDGECOLORS = [];

    # # scale = 1;
    node_ind = 0;
    for k,node in enumerate(SUBGRAPH.nodes):
        NODE_COLORS.append([1,1,1,1]);
        NODE_SIZES.append(1);
        NODE_EDGECOLORS.append([1,1,1,1]);
        if node in nodes:
            NODE_COLORS[-1] = node_colors[node_ind];
            NODE_SIZES[-1] = node_sizes[node_ind];
            NODE_EDGECOLORS[-1] = node_edge_colors[node_ind];
            node_ind = node_ind + 1;


    #     # for j,lst in enumerate(nodes):
    #     #     if node in lst:
    #     #         NODE_COLORS[-1] = node_colors[j];
    #     #         NODE_SIZES[-1] = node_sizes[j];
    edge_ind = 0;
    for k,edge in enumerate(SUBGRAPH.edges):

        # print(edge)
        # print(edges)

        EDGE_COLORS.append([0,0,0,1]);
        EDGE_WIDTHS.append(0.2);
        if edge in edges:
            EDGE_COLORS[-1] = edge_colors[edge_ind]
            EDGE_WIDTHS[-1] = edge_widths[edge_ind]

        # if edge in edges:
    #     # tag = (edge[0],edge[1])
    #     # for j,lst in enumerate(edges):
    #     #     if tag in lst:
    

    # NODE_COLORS = [1,1,1,1]
    # NODE_SIZES = 10;
    # EDGE_COLORS = [1,1,1,1]
    # EDGE_WIDTHS = 2;
    if data['name'] == 'walk':
        print(NODE_SIZES[:10])

    if len(SUBGRAPH.nodes())>0 or len(SUBGRAPH.edges)>0:
        fig, ax = ox.plot_graph(SUBGRAPH,bgcolor=bgcolor,  
                                node_color=NODE_COLORS,
                                node_size = NODE_SIZES,
                                edge_color=EDGE_COLORS,
                                edge_linewidth=EDGE_WIDTHS,
                                node_edgecolor = NODE_EDGECOLORS,
                                figsize=(20,20),
                                show=False,); #file_format='svg')        
        ax.set_xlim([lims[0],lims[1]])
        ax.set_ylim([lims[2],lims[3]])
        plt.savefig(fileName,bbox_inches='tight',pad_inches = 0,transparent=False)
        plt.close()    

    else:
        fig,ax = plt.subplots(1,1)
        ax.set_xlim([lims[0],lims[1]])
        ax.set_ylim([lims[2],lims[3]])
        plt.savefig(fileName,bbox_inches='tight',pad_inches = 0,transparent=False)
        plt.close()    






def plotGRAPH(GRAPH,data):

    if 'node_lists' in data: node_lists = data['node_lists'];
    else: node_list = [];
    if 'edge_lists' in data: edge_lists = data['edge_lists'];
    else: edge_list = []

    nn = len(node_lists);
    ne = len(edge_lists);

    if 'node_colors' in data: node_colors = data['node_colors'];
    else: node_colors = np.outer(np.ones(nn),np.array([1,1,1,1]));
    if 'edge_colors' in data: edge_colors = data['edge_colors'];
    else: edge_colors = np.outer(np.ones(ne),np.array([1,1,1,1]));

    if 'node_sizes' in data: node_sizes = data['node_sizes'];
    else: node_sizes = 10.0*np.ones(nn);
    if 'edge_widths' in data: edge_widths = data['edge_widths'];
    else: edge_widths = 1.0*np.ones(ne);


    bgcolor = [0.8,0.8,0.8,1];
    NODE_COLORS = [];
    NODE_SIZES = [];

    # scale = 1;
    for k,node in enumerate(GRAPH.nodes):
        NODE_COLORS.append([1,1,1,1]);
        NODE_SIZES.append(0);
        for j,lst in enumerate(node_lists):
            if node in lst:
                NODE_COLORS[-1] = node_colors[j];
                NODE_SIZES[-1] = node_sizes[j];

    EDGE_COLORS = [];
    EDGE_WIDTHS = [];
    for k,edge in enumerate(GRAPH.edges):
        EDGE_COLORS.append([1,1,1,1]);
        EDGE_WIDTHS.append(2);
        tag = (edge[0],edge[1])
        for j,lst in enumerate(edge_lists):
            if tag in lst:
                EDGE_COLORS[-1] = edge_colors[j]
                EDGE_WIDTHS[-1] = edge_widths[j]




    fig, ax = ox.plot_graph(GRAPH,bgcolor=bgcolor,  
                            node_color=NODE_COLORS,
                            node_size = NODE_SIZES,
                            edge_color=EDGE_COLORS,
                            edge_linewidth=EDGE_WIDTHS,
                            figsize=(20,20),
                            show=False,); #file_format='svg')        











def generate_graph_presets(fileName,bgcolor,shows,WORLD,maxwids,mxpops,lims = [],other_edges = False):    

    include_graphs = {'ondemand':False,'drive':True,'transit':True,'walk':shows['walk']};
    colors = {'drive':[1,0,0],'walk':[1.,1.,0.],#[0.7,0.7,0.7],
              'lines':[0.,0.,0.],'transit':[1.,0.,1.],'gtfs': [1.,0.5,0.],   
              'ondemand':[0.,0.,1.],'direct':[0.6,0.,1.],'shuttle':[0.,0.,1.],          
              'source':[0.,0.,1.,0.5],'target':[1.,0.,0.,0.5], #[0.8,0.8,0.8],
              'shuttle_nodes':[1.,0.5,0.,0.5],
              'ondemand_indiv':[1.,0.,1.0],
              'ondemand1':[0.,0.,1.0],'ondemand2':[1.,0.,1.],
              'default_edge':[1,1,1,1]}
    #colors = {**colors, **colors0}

    sizes = {'source':100,'target':100,'shuttle':300,'direct':300,'gtfs':5}
    node_edgecolors = {'source':[0,0,0],'target':[0,0,0],'shuttle':'k','direct':'k','gtfs':'k'};

    # mxpop1 = 1.; #num_sources/10;
    threshs = {'drive': [0,0,mxpops['drive'],1],'walk': [0,0,mxpops['walk'],1],
               'transit': [0,0,mxpops['transit'],1],'gtfs': [0,0,mxpops['gtfs'],1],'ondemand': [0,0,mxpops['ondemand'],1]}

    params = {}
    #params['other'] = {}; 
    #params['other'] = {'nodes':other_nodes,'sizes':other_sizes,'color':other_color}
    params['shows'] = shows; params['colors'] = colors;
    params['sizes'] = sizes; params['node_edgecolors'] = node_edgecolors;
    params['maxwids'] = maxwids; params['threshs'] = threshs; params['mxpops'] = mxpops;
    params['filename'] = fileName; params['include_graphs'] = include_graphs
    # bgcolor = list(0.8*np.array([0.9,0.9,1])) + [0]
    params['lims'] = lims
    params['bgcolor'] = bgcolor;
    params['node_scale'] = 100.5;



    if other_edges == True: 
        params['other_edges'] = {'grp1':{'edges':WORLD['drive']['base_edge_masses'],
                                         'thresh': threshs['drive'],
                                         'maxwid':maxwids['drive'],
                                         'color':[0.9,0.1,0.1],
                                         'maxalpha':0.3}}; #colors['drive']}}
    return params







def plot_multimode(GRAPHS,NODES,DELIVERY,WORLD,params):


    start_time = time.time();
    shows = params['shows'];
    colors = params['colors']
    sizes = params['sizes']
    node_edgecolors = params['node_edgecolors']
    maxwids = params['maxwids'];
    threshs = params['threshs'];
    mxpops = params['mxpops'];
    node_scale = params['node_scale']
    fileName = params['filename'];
    include_graphs = params['include_graphs'];
    SIZES = params['SIZES']
    bgcolor = params['bgcolor']
    if 'other_edges' in params: OTHER_EDGES = params['other_edges']
    else: OTHER_EDGES = [];




    start_nodes = {'drive':[],'walk':[],'ondemand':[],'gtfs':[]}
    end_nodes = {'drive':[],'walk':[],'ondemand':[],'gtfs':[]}

    for m,mode in enumerate(start_nodes):
        for i,trip in enumerate(WORLD[mode]['active_trips']):
            start_nodes[mode].append(trip[0])
            end_nodes[mode].append(trip[1])


    DRIVE = WORLD['drive'];
    TRANSIT = WORLD['transit'];
    ONDEMAND = WORLD['ondemand'];
    WALK = WORLD['walk'];

    if 'other' in params:
        other_nodes = params['other']['nodes'];
        other_sizes = params['other']['sizes'];
        other_color = params['other']['color'];
    else:
        other_nodes = []
        other_sizes = [];
        other_color = [0,0,0,1];

    ONDEMANDS = DELIVERY['shuttle']
    shuttleNodes = []

    for i,ondemand in enumerate(ONDEMANDS):
        ONDEMAND = ONDEMANDS[ondemand]
    #     print(ONDEMAND['active_trips']);#['people'])
    #     print(ONDEMAND['nodes']['source'])
    #     print(ONDEMAND['nodes']['transit'])
    #     print(ONDEMAND['nodes']['transit2'])    
    #     print(len(ONDEMAND['sources']))
    #     print(len(ONDEMAND['targets']))    
        shuttleNodes.append(ONDEMAND['sources']+ONDEMAND['targets'])
    ONDEMANDS = DELIVERY['direct']
    directNodes = [];
    for i,ondemand in enumerate(ONDEMANDS):
        ONDEMAND = ONDEMANDS[ondemand]
        directNodes.append(ONDEMAND['sources']+ONDEMAND['targets'])
    

    
    #include_graphs = {'ondemand':False,'drive':False,'transit':False,'walk':True};
    plot_graphs = [];
    
    zz = WORLD['gtfs']['current_edge_masses'];
    keys = list(zz);
    gtfsEdgeList = [zz[key] for _,key in enumerate(keys)]
    
    for _,mode in enumerate(include_graphs):
        if include_graphs[mode]==True:
            plot_graphs.append(GRAPHS[mode])
    full_graph = nx.compose_all(plot_graphs);

    typ = 'shuttle'
    shuttle_sources = []
    shuttle_transits = []
    shuttle_transit2s = [];
    for i,deliv in enumerate(DELIVERY[typ]):
        shuttle_sources.append(DELIVERY[typ][deliv]['nodes']['source']);
        shuttle_transits.append(DELIVERY[typ][deliv]['nodes']['transit']);
        shuttle_transit2s.append(DELIVERY[typ][deliv]['nodes']['transit2']);
    
    typ = 'direct'
    direct_sources = [];
    direct_transits = [];
    direct_transit2s = [];
    for i,deliv in enumerate(DELIVERY[typ]):
        direct_sources.append(DELIVERY[typ][deliv]['nodes']['source']);
        direct_transits.append(DELIVERY[typ][deliv]['nodes']['transit']);
        direct_transit2s.append(DELIVERY[typ][deliv]['nodes']['transit2']);
    
    transit_nodes = list(GRAPHS['transit'].nodes)
    DNODES,DLOCS = listDeliveryNodes(GRAPHS,WORLD);
    
    SHUTTLES = DELIVERY['shuttle']
    DIRECTS = DELIVERY['direct']
        
    shuttleNodes = []
    shuttleEdges = []
    directNodes = [];
    directEdges = [];
        
    ONDEMANDS = SHUTTLES
    for i,ondemand in enumerate(ONDEMANDS):
        ONDEMAND = ONDEMANDS[ondemand]
        if len(ONDEMAND['active_trip_history'])>0:
            trips = ONDEMAND['active_trip_history'][-1]
            visited_nodes = nodesFromTrips(trips)
        else:
            visited_nodes = [];
    
        current_path = ONDEMAND['current_path']
        shuttleNodes.append(visited_nodes+current_path)
        visited_edges = edgesFromPath(current_path)
        shuttleEdges.append(visited_edges)
        
    ONDEMANDS = DIRECTS

    special_ind = sample(list(range(len(ONDEMANDS))),1)[0]
    specialEdges = [];
    for i,ondemand in enumerate(ONDEMANDS):
        ONDEMAND = ONDEMANDS[ondemand]
        if len(ONDEMAND['active_trip_history'])>0:
            trips = ONDEMAND['active_trip_history'][-1]
            visited_nodes = nodesFromTrips(trips)
        else: 
            visited_nodes = [];


        current_path = ONDEMAND['current_path']    
        directNodes.append(visited_nodes+current_path)        
        visited_edges = edgesFromPath(current_path)

        directEdges.append(visited_edges)
        if i==special_ind:
            specialEdges = visited_edges

    mode = 'ondemand' 
    indz = [1]; typ = 'shuttle'; #'direct'
    ondemand_path_edges = []
    for _,ind in enumerate(indz):
        deliv = list(DELIVERY[typ].keys())[ind]
        path = DELIVERY[typ][deliv]['current_path']
        
        for k,node in enumerate(path):
            if k<len(path)-1:
                node1 = node; node2 = path[k+1];
                ondemand_path_edges.append((node1,node2));
    # ondemand_path_edges = []
        

    node_color = [];
    NODE_COLOR = [];
    node_size = [];
    node_edgecolor = [];
    node_zorder = [];



    # PDF = DELIVERY['current_PDF'];
    # MDF = DELIVERY['current_MDF'];

    payload = DELIVERY['current_payload'];
    manifest = DELIVERY['current_manifest'];
    PDF = payloadDF(payload,GRAPHS,include_drive_nodes = True);
    MDF = manifestDF(manifest,PDF)
    ZZ = MDF[MDF['run_id']==1]
    zz = list(ZZ.sort_values(['scheduled_time'])['drive_node'])


    # print('asdfasdfafd')
    # asdfadfsda
    PLANS = routesFromManifest(MDF,GRAPHS)
    # PATHS = routeFromStops(zz,GRAPHS['drive'])
    edge_lists4 = [PLANS[plan]['edges'] for _,plan in enumerate(PLANS)]
    
    


# %load_ext autoreload
# %autoreload 2
# from multimodal_functions import * 
# #         node_color.append(cmap(Mlocs[ii]/maxMloc))
# show_nodes = {}
# show_nodes['pickup'] = list(PYY['pickup_drive_node'])
# show_nodes['dropoff'] = list(PYY['dropoff_drive_node'])

# # print(PATH['edges'])
# node_colors = [[0,0,1,1],[1,0,0,1]]
# cmap = plt.get_cmap('viridis') 
# temp = 8; edge_colors = []
# for i in range(temp):
#     edge_colors.append(list(cmap(i/temp)))
    
# node_lists = [list(PYY['pickup_drive_node']), list(PYY['dropoff_drive_node'])]
# edge_lists = [PLANS[plan]['edges'] for _,plan in enumerate(PLANS)]



# dat = {'node_lists':node_lists,'edge_lists':edge_lists,'node_colors':node_colors,'edge_colors':edge_colors,
#       'edge_widths':4*np.ones(len(edge_lists))}
# plotGRAPH(GRAPHS['drive'],dat)







    for k,node in enumerate(full_graph.nodes):

        node_color.append([1,1,1,1.]);
        node_size.append(2.)  
        node_edgecolor.append('w')
        node_zorder.append(1);


        if False:
            print('asdf')        
    
        ################################################
    #     elif node in shuttle_sources:
    #         node_color.append(colors['shuttle']);
    #         node_size.append(10)        
    #         node_edgecolor.append('k')   
    #         node_zorder.append(50); 
    #     elif node in shuttle_transit2s:
    #         node_color.append(colors['shuttle']);
    #         node_size.append(150)        
    #         node_edgecolor.append('k')   
    #         node_zorder.append(50);
        ################################################

        elif node == DELIVERY['depot_node']:
            if shows['ondemand']:
                node_color[-1] = [0,0,1];
                node_size[-1] = 1000; #ther_sizes[node]);
                node_edgecolor[-1] = 'k'
                node_zorder[-1] = 1000;



        elif node in other_nodes:
            node_color[-1] = other_color;
            node_size[-1] = other_sizes[node];
            node_edgecolor[-1] = 'k'  
            node_zorder[-1] = 1000;

    
        elif (node in shuttle_transits) & shows['shuttle']:
            node_color[-1] = colors['shuttle_nodes']
            node_size[-1] = sizes['shuttle']        
            node_edgecolor[-1] = 'k'
            node_zorder[-1] = 1000;
    
        elif (node in direct_sources) & shows['direct']:
            node_color[-1] = colors['direct'] 
            node_size[-1] = sizes['direct']
            node_edgecolor[-1] = node_edgecolors['direct']
            node_zorder[-1] = 1000;
    #     elif node in direct_transits:
    #         node_color.append(colors['direct']);
    #         node_size.append(150)        
    #         node_edgecolor.append('k')   
    #         node_zorder.append(50);
    #     elif node in direct_transit2s:
    #         node_color.append(colors['direct']);
    #         node_size.append(150)        
    #         node_edgecolor.append('k')   
    #         node_zorder.append(50);        
        ###############################################        
            
        elif (node in NODES['orig']) & shows['source']:
            node_color[-1] = colors['source']
            #node_size.append(sizes['source'])
            node_size[-1] = SIZES['home_sizes'][node]*node_scale
            node_edgecolor[-1] = node_edgecolors['source']#colors['source'])
            node_zorder[-1] = 100
        elif (node in NODES['dest']) & shows['target']:
            node_color[-1] = colors['target']
            #node_size.append(sizes['target'])
            node_size[-1] = SIZES['work_sizes'][node]*node_scale
            node_edgecolor[-1] = node_edgecolors['target']
            node_zorder[-1] = 100
            
        ################################################
        elif (node in transit_nodes) & shows['gtfs']:
            node_color[-1] = [0,0,0] #colors['gtfs']);
            node_size[-1] = sizes['gtfs']
            node_edgecolor[-1] = node_edgecolors['gtfs']
            node_zorder[-1] = 10


        for m,mode in enumerate(start_nodes):
            if shows[mode]:
                if node in start_nodes[mode]:
                    node_color[-1] = colors[mode];
                    node_size[-1] = 100; #sizes[mode]
                    node_edgecolor[-1] = 'k';#node_edgecolors[mode]
                    node_zorder[-1] = 10
        for m,mode in enumerate(end_nodes):
            if shows[mode]:
                if node in end_nodes[mode]:
                    node_color[-1] = colors[mode];
                    node_size[-1] = 100; #sizes[mode]
                    node_edgecolor[-1] ='k';# node_edgecolors[mode]
                    node_zorder[-1] = 10


            
    
    edge_color = [];
    edge_width = [];
    drive_edges = list(GRAPHS['drive'].edges())
    ondemand_edges = list(GRAPHS['ondemand'].edges())
    walk_edges = list(GRAPHS['walk'].edges())
    transit_edges = list(GRAPHS['transit'].edges())



    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    
    
    fade_drive = False;
    
    for k,edgex in enumerate(full_graph.edges):
        
        # if edge_found == False: 
        # edge_found = False;
        
        edge = (edgex[0],edgex[1]);
        edge_color.append([1,1,1,1.]);
        edge_width.append(2.)  


        #### COLORING OTHER EDGES
        if shows['base']:
            if len(OTHER_EDGES)>0:
                for i,group in enumerate(OTHER_EDGES):
                    GROUP = OTHER_EDGES[group]
                    thresh_lims = GROUP['thresh']
                    color = GROUP['color']
                    maxwid = GROUP['maxwid']
                    maxalpha = GROUP['maxalpha']
                    edge2 = (edge[0],edge[1],0)
                    if edge2 in GROUP['edges']:
                        mmass = GROUP['edges'][edge2]    
                        if mmass > 0.:
                            edge_color[-1] = color+[maxalpha*thresh(mmass,thresh_lims)]; #[chopoff(edge_mass,mx1,0,1)])
                            edge_width[-1] = maxwids['base']*thresh(mmass,thresh_lims);
                

        #### COLORING PRIMARY EDGES 
        if False:
            print('blah');
            
        elif edge in drive_edges:


            if fade_drive:
                edge_color[-1] = [1,1,1,0.5];
                edge_width[-1] = 1.
                    
            if shows['drive'] or shows['ondemand'] or shows['ondemand_indiv'] or shows['shuttle'] or shows['direct']:
                tag = (edge[0],edge[1],0)
                edge_mass = WORLD['drive']['current_edge_masses'][tag]
                ondemand_mass = WORLD['ondemand']['current_edge_masses'][tag]
                ondemand_mass1 = WORLD['ondemand']['current_edge_masses1'][tag]            
                ondemand_mass2 = WORLD['ondemand']['current_edge_masses2'][tag]
                            
                if False: #edge in ondemand_path_edges:
                    edge_color[-1] = colors['ondemand_indiv']+[1]; #[chopoff(edge_mass,mx1,0,1)])
                    edge_width[-1] = maxwids['ondemand_indiv']; #maxwids['ondemand']*thresh(edge_mass,threshs['ondemand']));  
                elif (edge_mass > 0) & shows['drive']:
                    edge_color[-1] = colors['drive']+[1]; #[thresh(edge_mass,threshs['drive'])]); #[chopoff(edge_mass,mx1,0,1)])
                    edge_width[-1] = maxwids['drive']*thresh(edge_mass,threshs['drive']);
    
                    #maxwids['drive']*chopoff(edge_mass,mx1,0.5,1))            
                elif False: #(ondemand_mass > 0) & shows['ondemand']:
                    #print('found ondemand edge...')
                    edge_color[-1] = colors['ondemand']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
                    edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);
                elif False: #(ondemand_mass2 > 0) & shows['ondemand']:
                    #print('found ondemand edge...')
                    edge_color[-1] = colors['ondemand2']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
                    edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                                
                elif False: #(ondemand_mass1 > 0) & shows['ondemand']:
                    #print('found ondemand edge...')
                    edge_color[-1] = colors['ondemand1']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
                    edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
    #             elif shows['ondemand']: 

                if shows['shuttle']:
                    for _,edges in enumerate(shuttleEdges):
                        if edge in edges:
                            edge_color[-1] = colors['shuttle']+[params['set_alphas']['shuttle']]
                            #[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
                            edge_width[-1] = params['set_wids']['shuttle']; #maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
    
                if shows['direct']:
                    for _,edges in enumerate(directEdges):
                        if edge in edges:
                            edge_color[-1] = colors['direct']+[params['set_alphas']['direct']]
                            #[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
                            edge_width[-1] = params['set_wids']['shuttle']; #maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
                    if edge in specialEdges:
                        edge_color[-1] = colors['direct']+[1.];#thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
                        edge_width[-1] = 10; #maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                

                        
    #             for _,edges in enumerate(directEdges):
    #                 tag2 = (tag[0],tag[1])
    #                 if tag2 in edges:
    #                     edge_color[-1] = colors['ondemand']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
    #                     edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);
    

            if shows['ondemand']:
                for _,edge_list in enumerate(edge_lists4):
                    if edge in edge_list:
                        edge_color[-1] = [0,0,1,0.5];
                        edge_width[-1] = 6.
    
                
    #     elif edge in ondemand_edges:
    #         if shows['ondemand']:
    #             edge_found = True;
    #             tag = (edge[0],edge[1],0)
    #             edge_mass = WORLD['ondemand']['current_edge_masses'][tag]
    #             if edge_mass > 0:
    #                 #print('found ondemand edge...')
    #                 edge_color.append(colors['ondemand']+[thresh(ondemand_mass,threshs['ondemand'])]); #[chopoff(edge_mass,mx1,0,1)])
    #                 edge_width.append(maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']));
    #             else:
    #                 edge_color.append([1,1,1]); 
    #                 edge_width.append(4)
    #         else:
    #             edge_color.append([1,1,1]); 
    #             edge_width.append(1)
        
        elif edge in transit_edges:
            if shows['transit'] or shows['gtfs'] or shows['lines']:
                edge_found = True;
                tag = (edge[0],edge[1],0)
                edge_mass = WORLD['transit']['current_edge_masses'][tag]
                #try:
                edge_mass2 = WORLD['gtfs']['current_edge_masses'][tag]   
                
    #             if edge_mass2 > 0:
    #                 print(edge_mass2)
                #except:
    #                 tag1 = (edge[0],edge[1],1)
    #                 edge_mass2 = WORLD['gtfs']['current_edge_masses'][tag1]        
    
                if (edge_mass2 > 0) & shows['gtfs']:
                    edge_color[-1] = colors['gtfs']+[1];#thresh(edge_mass2,threshs['gtfs'])]);
                    edge_width[-1] = maxwids['gtfs']*thresh(edge_mass2,threshs['gtfs']);
                elif (edge_mass > 0) & shows['transit']:           
                    edge_color[-1] = colors['transit']+[thresh(edge_mass,threshs['transit'])];
                    edge_width[-1] = maxwids['transit']*thresh(edge_mass,threshs['transit']);
                elif shows['lines']: 
                    edge_color[-1] = colors['lines']+[1]; #colors['transit']);
                    edge_width[-1] = maxwids['lines'];
                    
            else: 
                edge_color[-1] = [1,1,1,1]; #colors['transit']);
                edge_width[-1] = 1
    
                
        elif edge in walk_edges:        
            
            if shows['walk']:
                # tag = (edge[0],edge[1],0)
                # edge_mass = WORLD['walk']['current_edge_masses'][tag]
                # if edge_mass > 0:
                #     #edge_color.append(winter_cmap(chopoff(edge_mass,mx1,0,1)))
                #     edge_color[-1] = colors['walk'] + [1]; #[thresh(edge_mass,threshs['walk'])]); #[chopoff(edge_mass,mwalk,0,1)])
                #     edge_width[-1] = 10.;#maxwids['walk']*thresh(edge_mass,threshs['walk']); #chopoff(edge_mass,mwalk,0.5,1))
                #     #edge_width.append(4); 
                # else: 
                #     edge_color[-1] = [1,1,1,1]; 
                #     edge_width[-1] = 0.5

                extra_ints = [0,1,2];
                for _,kk in enumerate(extra_ints): 
                    tag = (edge[0],edge[1],kk)
                    if tag in WORLD['walk']['current_edge_masses']:
                        edge_mass = WORLD['walk']['current_edge_masses'][tag]
                        if edge_mass > 0:
                            #edge_color.append(winter_cmap(chopoff(edge_mass,mx1,0,1)))
                            edge_color[-1] = colors['walk'] + [1]; #[thresh(edge_mass,threshs['walk'])]); #[chopoff(edge_mass,mwalk,0,1)])
                            edge_width[-1] = 4. #maxwids['walk']*thresh(edge_mass,threshs['walk']); #chopoff(edge_mass,mwalk,0.5,1))
                            #edge_width.append(4); 
                        else: 
                            edge_color[-1] = [1,1,1,1]; 
                            edge_width[-1] = 0.5

            else: 
                edge_color[-1] = [1,1,1,1]; 
                edge_width[-1] = 0.5
                
        # else:
        #     edge_color[-1] = [1,1,1,1.]
        #     edge_width[-1] = 2.
            

        
        
    # edge_masses = {'drive':[]};
    # zmodes = ['drive','transit','walk'];
    # edge_masses = {}
    # for _,mode in enumerate(zmodes)
    #     drive_trip_edges = []; NETWORK = WORLD['drive'];
    #     for e,edge in enumerate(NETWORK['current_edge_masses']):  
    #     path = DRIVE['trips'][trip]['path'][-1]
    #     for j,node in enumerate(path):
    #         if j < len(path)-1:
    #             edge = (path[j],path[j+1],0);
    #         drive_trip_edges.append(edge)
    # graph = GRAPHS['drive']
    
    print('starting plotting...')
    fig, ax = ox.plot_graph(full_graph,bgcolor=bgcolor,  
                            node_color=node_color,node_edgecolor=node_edgecolor,
                            node_size=node_size,node_zorder=1,#node_zorder,
                            edge_color=edge_color,
                            edge_linewidth=edge_width,figsize=(30,30),
                            show=False,); #file_format='svg')




    x = np.linspace(0, 10, 1000)
    
    if shows['shuttle']:     
        params = {'linewidth':2,'color': colors['shuttle'],'alpha':0.05}
        # params = {'edge_color':[0,0,0]}
        for _,nodes in enumerate(shuttleNodes):
            shape = locsFromNodes(nodes,GRAPHS['ondemand'])
        #     try:
            if len(shape)>2: plotCvxHull(ax,shape,params)    
        #     except: continue
    if False: #shows['direct']: ### TOO UGLY
        params = {'linewidth':2,'color':colors['direct'],'alpha':0.05}       
        for _,nodes in enumerate(directNodes):
            shape = locsFromNodes(nodes,GRAPHS['ondemand'])
            try:
                if len(shape)>2: plotCvxHull(ax,shape,params)
            except: continue
            
    
    
    # if shows['direct']:
    #     for _,deliv in enumerate(DLOCS):
    #         if True: #deliv[:9] == 'delivery2':
    #             locs = DLOCS[deliv]
    #             shape = np.array(locs);
    #             params['color'] = colors['direct']
    #             # params['color'] = colors['shuttle']

    #             if len(shape)>2:
    #                 plotCvxHull(ax,shape,params)
    #     # fig, ax = plt.subplots()
    
    
    fontsize=24;
    if True:
        lw1 = 8; lw2 = 6; markersize = 32;    
        if shows['source']:
            ax.plot([0],[0], color=colors['source'],marker='o',
                    markersize=20,markeredgecolor='k',linewidth=0, label='origins')
        if shows['target']:
            ax.plot([0],[0], color=colors['target'],marker='o',
                    markersize=20,markeredgecolor='k',linewidth=0, label='dests')
        if shows['direct']:
            ax.plot([0],[0], color=colors['direct'],marker='o',
                    markersize=20,markeredgecolor='k',linewidth=0, label='direct centers')
            ax.plot([0,1],[0,1], color=colors['direct'],linestyle='-',
                    markersize=20,markeredgecolor='k',linewidth=lw1, label='direct regions')
        if shows['shuttle']:
            ax.plot([0],[0], color=colors['shuttle_nodes'],marker='o',markersize=20,markeredgecolor='k',
                    linewidth=0, label='shuttle stops')
            ax.plot([0,1],[0,1], color=colors['shuttle'],linestyle='-',
                    markersize=20,markeredgecolor='k',linewidth=lw1, label='shuttle regions')
        if shows['lines']:
            ax.plot([0,1],[0,1], color=colors['lines'],linestyle='-',
                    markersize=20,markeredgecolor='k',linewidth=lw1, label='transit lines')
    
        lw1 = 10; lw2 = 6;
        if shows['drive']:
            ax.plot([0],[0], color=colors['drive'], linewidth=lw1, label='drive traffic')
        if shows['walk']:
            ax.plot([0],[0], color=colors['walk'], linewidth=lw2, label='walk traffic')
        if shows['ondemand']:
            ax.plot([0],[0], color=colors['ondemand'],linewidth=lw1, label='ondemand routes')
        if shows['gtfs']:
            ax.plot([0],[0], color=colors['gtfs'],linewidth=lw1, label='transit traffic')
            # ax.plot([0],[0], color=[0,0,0],linewidth=lw2, label='transit lines')
        # leg = ax.legend();
        # plt.show()
        if shows['legend']:
            ax.legend(fancybox=True, fontsize=fontsize,
                      framealpha=0.6, shadow=True,
                      borderpad=1,loc = 'upper left')
        end_time = time.time()
        print('time to draw graph: ',end_time-start_time)
    
    if False:
        lw1 = 0; lw2 = 6; markersize = 32;
        ax.plot([0],[0], color=[1,1,1],marker='o', markersize=markersize,
                markeredgecolor='w',linewidth=0, label='Full Transit Graph')    
        ax.legend(fancybox=True, fontsize=24, framealpha=0.6, shadow=True, borderpad=1,loc = 'upper left')


    if len(params['lims'])>0:
        lims = params['lims'];
        ax.set_xlim([lims[0],lims[1]])
        ax.set_ylim([lims[2],lims[3]])
    
    # fileName = 'current.pdf'
    printFigs = False;
    # plt.axis('off')
    plt.savefig(fileName,bbox_inches='tight',pad_inches = 0,transparent=False)

    return {'fig':fig,'ax':ax}
    







# def plotMultiMode(GRAPHS,data):

#     if 'node_lists' in data: node_lists = data['node_lists'];
#     else: node_list = [];
#     if 'edge_lists' in data: edge_lists = data['edge_lists'];
#     else: edge_list = []

#     nn = len(node_lists);
#     ne = len(edge_lists);

#     if 'node_colors' in data: node_colors = data['node_colors'];
#     else: node_colors = np.outer(np.ones(nn),np.array([1,1,1,1]));
#     if 'edge_colors' in data: edge_colors = data['edge_colors'];
#     else: edge_colors = np.outer(np.ones(ne),np.array([1,1,1,1]));

#     if 'node_sizes' in data: node_sizes = data['node_sizes'];
#     else: node_sizes = 10.0*np.ones(nn);
#     if 'edge_widths' in data: edge_widths = data['edge_widths'];
#     else: edge_widths = 1.0*np.ones(ne);

#     bgcolor = [0.8,0.8,0.8,1];
#     NODE_COLORS = [];
#     NODE_SIZES = [];

    
#     # scale = 1;
#     for k,node in enumerate(GRAPH.nodes):
#         NODE_COLORS.append([1,1,1,1]);
#         NODE_SIZES.append(0);
#         for j,lst in enumerate(node_lists):
#             if node in lst:
#                 NODE_COLORS[-1] = node_colors[j];
#                 NODE_SIZES[-1] = node_sizes[j];

#     EDGE_COLORS = [];
#     EDGE_WIDTHS = [];
#     for k,edge in enumerate(GRAPH.edges):
#         EDGE_COLORS.append([1,1,1,1]);
#         EDGE_WIDTHS.append(2);
#         tag = (edge[0],edge[1])
#         for j,lst in enumerate(edge_lists):
#             if tag in lst:
#                 EDGE_COLORS[-1] = edge_colors[j]
#                 EDGE_WIDTHS[-1] = edge_widths[j]

#     fig, ax = ox.plot_graph(GRAPH,bgcolor=bgcolor,  
#                             node_color=NODE_COLORS,
#                             node_size = NODE_SIZES,
#                             edge_color=EDGE_COLORS,
#                             edge_linewidth=EDGE_WIDTHS,
#                             figsize=(20,20),
#                             show=False,); #file_format='svg')        






# def generate_graph_presets(fileName,shows,WORLD,maxwids,mxpops,other_edges = False):    

#     include_graphs = {'ondemand':False,'drive':True,'transit':True,'walk':shows['walk']};
#     colors = {'drive':[1,0,0],'walk':[1.,1.,0.],#[0.7,0.7,0.7],
#               'lines':[0.,0.,0.],'transit':[1.,0.,1.],'gtfs': [1.,0.5,0.],   
#               'ondemand':[0.,0.,1.],'direct':[0.6,0.,1.],'shuttle':[0.,0.,1.],          
#               'source':[0.,0.,1.,0.5],'target':[1.,0.,0.,0.5], #[0.8,0.8,0.8],
#               'shuttle_nodes':[1.,0.5,0.,0.5],
#               'ondemand_indiv':[1.,0.,1.0],
#               'ondemand1':[0.,0.,1.0],'ondemand2':[1.,0.,1.],
#               'default_edge':[1,1,1,1]}
#     #colors = {**colors, **colors0}

#     sizes = {'source':100,'target':100,'shuttle':300,'direct':300,'gtfs':5}
#     node_edgecolors = {'source':[0,0,0],'target':[0,0,0],'shuttle':'k','direct':'k','gtfs':'k'};

#     # mxpop1 = 1.; #num_sources/10;
#     threshs = {'drive': [0,0,mxpops['drive'],1],'walk': [0,0,mxpops['walk'],1],
#                'transit': [0,0,mxpops['transit'],1],'gtfs': [0,0,mxpops['gtfs'],1],'ondemand': [0,0,mxpops['ondemand'],1]}

#     params = {}
#     #params['other'] = {}; 
#     #params['other'] = {'nodes':other_nodes,'sizes':other_sizes,'color':other_color}
#     params['shows'] = shows; params['colors'] = colors;
#     params['sizes'] = sizes; params['node_edgecolors'] = node_edgecolors;
#     params['maxwids'] = maxwids; params['threshs'] = threshs; params['mxpops'] = mxpops;
#     params['filename'] = fileName; params['include_graphs'] = include_graphs
#     bgcolor = list(0.8*np.array([0.9,0.9,1])) + [1]
#     params['bgcolor'] = bgcolor;
#     params['node_scale'] = 10.5;

#     if other_edges == True: 
#         params['other_edges'] = {'grp1':{'edges':WORLD['drive']['base_edge_masses'],
#                                          'thresh': threshs['drive'],
#                                          'maxwid':maxwids['drive'],
#                                          'color':[0.9,0.1,0.1],
#                                          'maxalpha':0.3}}; #colors['drive']}}
#     return params



# def plot_multimode(GRAPHS,NODES,DELIVERY,WORLD,params):

#     start_time = time.time();

#     shows = params['shows'];
#     colors = params['colors']
#     sizes = params['sizes']
#     node_edgecolors = params['node_edgecolors']
#     maxwids = params['maxwids'];
#     threshs = params['threshs'];
#     mxpops = params['mxpops'];
#     node_scale = params['node_scale']
#     fileName = params['filename'];
#     include_graphs = params['include_graphs'];
#     SIZES = params['SIZES']
#     bgcolor = params['bgcolor']
#     if 'other_edges' in params: OTHER_EDGES = params['other_edges']
#     else: OTHER_EDGES = [];


#     DRIVE = WORLD['drive'];
#     TRANSIT = WORLD['transit'];
#     ONDEMAND = WORLD['ondemand'];
#     WALK = WORLD['walk'];

#     if 'other' in params:
#         other_nodes = params['other']['nodes'];
#         other_sizes = params['other']['sizes'];
#         other_color = params['other']['color'];
#     else:
#         other_nodes = []
#         other_sizes = [];
#         other_color = [0,0,0,1];

#     ONDEMANDS = DELIVERY['shuttle']
#     shuttleNodes = []

#     for i,ondemand in enumerate(ONDEMANDS):
#         ONDEMAND = ONDEMANDS[ondemand]
#     #     print(ONDEMAND['active_trips']);#['people'])
#     #     print(ONDEMAND['nodes']['source'])
#     #     print(ONDEMAND['nodes']['transit'])
#     #     print(ONDEMAND['nodes']['transit2'])    
#     #     print(len(ONDEMAND['sources']))
#     #     print(len(ONDEMAND['targets']))    
#         shuttleNodes.append(ONDEMAND['sources']+ONDEMAND['targets'])
#     ONDEMANDS = DELIVERY['direct']
#     directNodes = [];
#     for i,ondemand in enumerate(ONDEMANDS):
#         ONDEMAND = ONDEMANDS[ondemand]
#         directNodes.append(ONDEMAND['sources']+ONDEMAND['targets'])
    

    
#     #include_graphs = {'ondemand':False,'drive':False,'transit':False,'walk':True};
#     plot_graphs = [];
    
#     zz = WORLD['gtfs']['current_edge_masses'];
#     keys = list(zz);
#     gtfsEdgeList = [zz[key] for _,key in enumerate(keys)]
    
#     for _,mode in enumerate(include_graphs):
#         if include_graphs[mode]==True:
#             plot_graphs.append(GRAPHS[mode])
#     full_graph = nx.compose_all(plot_graphs);

#     typ = 'shuttle'
#     shuttle_sources = []
#     shuttle_transits = []
#     shuttle_transit2s = [];
#     for i,deliv in enumerate(DELIVERY[typ]):
#         shuttle_sources.append(DELIVERY[typ][deliv]['nodes']['source']);
#         shuttle_transits.append(DELIVERY[typ][deliv]['nodes']['transit']);
#         shuttle_transit2s.append(DELIVERY[typ][deliv]['nodes']['transit2']);
    
#     typ = 'direct'
#     direct_sources = [];
#     direct_transits = [];
#     direct_transit2s = [];
#     for i,deliv in enumerate(DELIVERY[typ]):
#         direct_sources.append(DELIVERY[typ][deliv]['nodes']['source']);
#         direct_transits.append(DELIVERY[typ][deliv]['nodes']['transit']);
#         direct_transit2s.append(DELIVERY[typ][deliv]['nodes']['transit2']);
    
#     transit_nodes = list(GRAPHS['transit'].nodes)
#     DNODES,DLOCS = listDeliveryNodes(GRAPHS,WORLD);
    
#     SHUTTLES = DELIVERY['shuttle']
#     DIRECTS = DELIVERY['direct']
        
#     shuttleNodes = []
#     shuttleEdges = []
#     directNodes = [];
#     directEdges = [];
        
#     ONDEMANDS = SHUTTLES
#     for i,ondemand in enumerate(ONDEMANDS):
#         ONDEMAND = ONDEMANDS[ondemand]
#         if len(ONDEMAND['active_trip_history'])>0:
#             trips = ONDEMAND['active_trip_history'][-1]
#             visited_nodes = nodesFromTrips(trips)
#         else:
#             visited_nodes = [];
    
#         current_path = ONDEMAND['current_path']
#         shuttleNodes.append(visited_nodes+current_path)
#         visited_edges = edgesFromPath(current_path)
#         shuttleEdges.append(visited_edges)
        
#     ONDEMANDS = DIRECTS

#     special_ind = sample(list(range(len(ONDEMANDS))),1)[0]
#     specialEdges = [];
#     for i,ondemand in enumerate(ONDEMANDS):
#         ONDEMAND = ONDEMANDS[ondemand]
#         if len(ONDEMAND['active_trip_history'])>0:
#             trips = ONDEMAND['active_trip_history'][-1]
#             visited_nodes = nodesFromTrips(trips)
#         else: 
#             visited_nodes = [];


#         current_path = ONDEMAND['current_path']    
#         directNodes.append(visited_nodes+current_path)        
#         visited_edges = edgesFromPath(current_path)

#         directEdges.append(visited_edges)
#         if i==special_ind:
#             specialEdges = visited_edges

#     mode = 'ondemand' 
#     indz = [1]; typ = 'shuttle'; #'direct'
#     ondemand_path_edges = []
#     for _,ind in enumerate(indz):
#         deliv = list(DELIVERY[typ].keys())[ind]
#         path = DELIVERY[typ][deliv]['current_path']
        
#         for k,node in enumerate(path):
#             if k<len(path)-1:
#                 node1 = node; node2 = path[k+1];
#                 ondemand_path_edges.append((node1,node2));
#     # ondemand_path_edges = []
        

#     node_color = [];
#     NODE_COLOR = [];
#     node_size = [];
#     node_edgecolor = [];
#     node_zorder = [];



#     # PDF = DELIVERY['current_PDF'];
#     # MDF = DELIVERY['current_MDF'];

#     payload = DELIVERY['current_payload'];
#     manifest = DELIVERY['current_manifest'];
#     PDF = payloadDF(payload,GRAPHS,include_drive_nodes = True);
#     MDF = manifestDF(manifest,PDF)
#     ZZ = MDF[MDF['run_id']==1]
#     zz = list(ZZ.sort_values(['scheduled_time'])['drive_node'])


#     # print('asdfasdfafd')
#     # asdfadfsda
#     PLANS = routesFromManifest(MDF,GRAPHS)
#     # PATHS = routeFromStops(zz,GRAPHS['drive'])
#     edge_lists4 = [PLANS[plan]['edges'] for _,plan in enumerate(PLANS)]
    

#     for k,node in enumerate(full_graph.nodes):
#         if False:
#             print('asdf')

#         elif node == DELIVERY['depot_node']:
#             node_color.append([0,0,1]);
#             node_size.append(1000); #ther_sizes[node]);
#             node_edgecolor.append('k')   
#             node_zorder.append(1000);



#         elif node in other_nodes:
#             node_color.append(other_color);
#             node_size.append(other_sizes[node]);
#             node_edgecolor.append('k')   
#             node_zorder.append(1000);

    
#         elif (node in shuttle_transits) & shows['shuttle']:
#             node_color.append(colors['shuttle_nodes']);
#             node_size.append(sizes['shuttle'])        
#             node_edgecolor.append('k')   
#             node_zorder.append(1000);
    
#         elif (node in direct_sources) & shows['direct']:
#             node_color.append(colors['direct']);
#             node_size.append(sizes['direct'])        
#             node_edgecolor.append(node_edgecolors['direct'])
#             node_zorder.append(1000);        
            
#         elif (node in NODES['orig']) & shows['source']:
#             node_color.append(colors['source'])
#             #node_size.append(sizes['source'])
#             node_size.append(SIZES['home_sizes'][node]*node_scale)        
#             node_edgecolor.append(node_edgecolors['source']);#colors['source'])
#             node_zorder.append(100);
#         elif (node in NODES['dest']) & shows['target']:
#             node_color.append(colors['target'])
#             #node_size.append(sizes['target'])
#             node_size.append(SIZES['work_sizes'][node]*node_scale);
#             node_edgecolor.append(node_edgecolors['target'])
#             node_zorder.append(100);
            
#         ################################################
#         elif (node in transit_nodes) & shows['gtfs']:
#             node_color.append([0,0,0]); #colors['gtfs']);
#             node_size.append(sizes['gtfs'])        
#             node_edgecolor.append(node_edgecolors['gtfs'])   
#             node_zorder.append(10);        
        
#         else: 
#             node_color.append([1,1,1]); #[0,0,1])
#             node_size.append(0)
#             node_edgecolor.append('w')
#             node_zorder.append(1);        
    
    
    
#     edge_color = [];
#     edge_width = [];
#     drive_edges = list(GRAPHS['drive'].edges())
#     ondemand_edges = list(GRAPHS['ondemand'].edges())
#     walk_edges = list(GRAPHS['walk'].edges())
#     transit_edges = list(GRAPHS['transit'].edges())



#     ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
#     ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
#     ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
#     ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE ###### EDGE 
    
    
#     fade_drive = False;
    
#     for k,edgex in enumerate(full_graph.edges):
        
#         # if edge_found == False: 
#         # edge_found = False;
        
#         edge = (edgex[0],edgex[1]);
#         edge_color.append([1,1,1,1.]);
#         edge_width.append(2.)  


#         #### COLORING OTHER EDGES
#         if shows['base']:
#             if len(OTHER_EDGES)>0:
#                 for i,group in enumerate(OTHER_EDGES):
#                     GROUP = OTHER_EDGES[group]
#                     thresh_lims = GROUP['thresh']
#                     color = GROUP['color']
#                     maxwid = GROUP['maxwid']
#                     maxalpha = GROUP['maxalpha']
#                     edge2 = (edge[0],edge[1],0)
#                     if edge2 in GROUP['edges']:
#                         mmass = GROUP['edges'][edge2]    
#                         if mmass > 0.:
#                             edge_color[-1] = color+[maxalpha*thresh(mmass,thresh_lims)]; #[chopoff(edge_mass,mx1,0,1)])
#                             edge_width[-1] = maxwids['base']*thresh(mmass,thresh_lims);
                

#         #### COLORING PRIMARY EDGES 
#         if False:
#             print('blah');
            
#         elif edge in drive_edges:

#             for _,edge_list in enumerate(edge_lists4):
#                 if edge in edge_list:
#                     edge_color[-1] = [0,0,1,0.5];
#                     edge_width[-1] = 5.

#             if fade_drive:
#                 edge_color[-1] = [1,1,1,0.5];
#                 edge_width[-1] = 1.
                    
#             if shows['drive'] or shows['ondemand'] or shows['ondemand_indiv'] or shows['shuttle'] or shows['direct']:
#                 tag = (edge[0],edge[1],0)
#                 edge_mass = WORLD['drive']['current_edge_masses'][tag]
#                 ondemand_mass = WORLD['ondemand']['current_edge_masses'][tag]
#                 ondemand_mass1 = WORLD['ondemand']['current_edge_masses1'][tag]            
#                 ondemand_mass2 = WORLD['ondemand']['current_edge_masses2'][tag]
                            
#                 if False: #edge in ondemand_path_edges:
#                     edge_color[-1] = colors['ondemand_indiv']+[1]; #[chopoff(edge_mass,mx1,0,1)])
#                     edge_width[-1] = maxwids['ondemand_indiv']; #maxwids['ondemand']*thresh(edge_mass,threshs['ondemand']));  
#                 elif (edge_mass > 0) & shows['drive']:
#                     edge_color[-1] = colors['drive']+[1]; #[thresh(edge_mass,threshs['drive'])]); #[chopoff(edge_mass,mx1,0,1)])
#                     edge_width[-1] = maxwids['drive']*thresh(edge_mass,threshs['drive']);
    
#                     #maxwids['drive']*chopoff(edge_mass,mx1,0.5,1))            
#                 elif False: #(ondemand_mass > 0) & shows['ondemand']:
#                     #print('found ondemand edge...')
#                     edge_color[-1] = colors['ondemand']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
#                     edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);
#                 elif False: #(ondemand_mass2 > 0) & shows['ondemand']:
#                     #print('found ondemand edge...')
#                     edge_color[-1] = colors['ondemand2']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
#                     edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                                
#                 elif False: #(ondemand_mass1 > 0) & shows['ondemand']:
#                     #print('found ondemand edge...')
#                     edge_color[-1] = colors['ondemand1']+[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
#                     edge_width[-1] = maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
#     #             elif shows['ondemand']: 

#                 if shows['shuttle']:
#                     for _,edges in enumerate(shuttleEdges):
#                         if edge in edges:
#                             edge_color[-1] = colors['shuttle']+[params['set_alphas']['shuttle']]
#                             #[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
#                             edge_width[-1] = params['set_wids']['shuttle']; #maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
    
#                 if shows['direct']:
#                     for _,edges in enumerate(directEdges):
#                         if edge in edges:
#                             edge_color[-1] = colors['direct']+[params['set_alphas']['direct']]
#                             #[thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
#                             edge_width[-1] = params['set_wids']['shuttle']; #maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
#                     if edge in specialEdges:
#                         edge_color[-1] = colors['direct']+[1.];#thresh(ondemand_mass,threshs['ondemand'])]; #[chopoff(edge_mass,mx1,0,1)])
#                         edge_width[-1] = 10; #maxwids['ondemand']*thresh(ondemand_mass,threshs['ondemand']);                
        
#         elif edge in transit_edges:
#             if shows['transit'] or shows['gtfs'] or shows['lines']:
#                 edge_found = True;
#                 tag = (edge[0],edge[1],0)
#                 edge_mass = WORLD['transit']['current_edge_masses'][tag]
#                 #try:
#                 edge_mass2 = WORLD['gtfs']['current_edge_masses'][tag]   
                
#     #             if edge_mass2 > 0:
#     #                 print(edge_mass2)
#                 #except:
#     #                 tag1 = (edge[0],edge[1],1)
#     #                 edge_mass2 = WORLD['gtfs']['current_edge_masses'][tag1]        
    
#                 if (edge_mass2 > 0) & shows['gtfs']:
#                     edge_color[-1] = colors['gtfs']+[1];#thresh(edge_mass2,threshs['gtfs'])]);
#                     edge_width[-1] = maxwids['gtfs']*thresh(edge_mass2,threshs['gtfs']);
#                 elif (edge_mass > 0) & shows['transit']:           
#                     edge_color[-1] = colors['transit']+[thresh(edge_mass,threshs['transit'])];
#                     edge_width[-1] = maxwids['transit']*thresh(edge_mass,threshs['transit']);
#                 elif shows['lines']: 
#                     edge_color[-1] = colors['lines']+[1]; #colors['transit']);
#                     edge_width[-1] = maxwids['lines'];
                    
#             else: 
#                 edge_color[-1] = [1,1,1,1]; #colors['transit']);
#                 edge_width[-1] = 1
    
                
#         elif edge in walk_edges:        
            
#             if shows['walk']:
#                 tag = (edge[0],edge[1],0)
#                 edge_mass = WORLD['walk']['current_edge_masses'][tag]
#                 if edge_mass > 0:
#                     #edge_color.append(winter_cmap(chopoff(edge_mass,mx1,0,1)))
#                     edge_color[-1] = colors['walk'] + [1]; #[thresh(edge_mass,threshs['walk'])]); #[chopoff(edge_mass,mwalk,0,1)])
#                     edge_width[-1] = maxwids['walk']*thresh(edge_mass,threshs['walk']); #chopoff(edge_mass,mwalk,0.5,1))
#                     #edge_width.append(4); 
#                 else: 
#                     edge_color[-1] = [1,1,1,1]; 
#                     edge_width[-1] = 0.5

#                 tag = (edge[0],edge[1],1)
#                 if tag in WORLD['walk']['current_edge_masses']:
#                     edge_mass = WORLD['walk']['current_edge_masses'][tag]
#                     if edge_mass > 0:
#                         #edge_color.append(winter_cmap(chopoff(edge_mass,mx1,0,1)))
#                         edge_color[-1] = colors['walk'] + [1]; #[thresh(edge_mass,threshs['walk'])]); #[chopoff(edge_mass,mwalk,0,1)])
#                         edge_width[-1] = maxwids['walk']*thresh(edge_mass,threshs['walk']); #chopoff(edge_mass,mwalk,0.5,1))
#                         #edge_width.append(4); 
#                     else: 
#                         edge_color[-1] = [1,1,1,1]; 
#                         edge_width[-1] = 0.5

#             else: 
#                 edge_color[-1] = [1,1,1,1]; 
#                 edge_width[-1] = 0.5
                
#     print('starting plotting...')
#     fig, ax = ox.plot_graph(full_graph,bgcolor=bgcolor,  
#                             node_color=node_color,node_edgecolor=node_edgecolor,
#                             node_size=node_size,node_zorder=1,#node_zorder,
#                             edge_color=edge_color,
#                             edge_linewidth=edge_width,figsize=(30,30),
#                             show=False,); #file_format='svg')




#     x = np.linspace(0, 10, 1000)
    
#     if shows['shuttle']:     
#         params = {'linewidth':2,'color': colors['shuttle'],'alpha':0.05}
#         # params = {'edge_color':[0,0,0]}
#         for _,nodes in enumerate(shuttleNodes):
#             shape = locsFromNodes(nodes,GRAPHS['ondemand'])
#         #     try:
#             if len(shape)>2: plotCvxHull(ax,shape,params)    
#         #     except: continue
#     if False: #shows['direct']: ### TOO UGLY
#         params = {'linewidth':2,'color':colors['direct'],'alpha':0.05}       
#         for _,nodes in enumerate(directNodes):
#             shape = locsFromNodes(nodes,GRAPHS['ondemand'])
#             try:
#                 if len(shape)>2: plotCvxHull(ax,shape,params)
#             except: continue
    
#     fontsize=24;
#     if True:
#         lw1 = 8; lw2 = 6; markersize = 32;    
#         if shows['source']:
#             ax.plot([0],[0], color=colors['source'],marker='o',
#                     markersize=20,markeredgecolor='k',linewidth=0, label='origins')
#         if shows['target']:
#             ax.plot([0],[0], color=colors['target'],marker='o',
#                     markersize=20,markeredgecolor='k',linewidth=0, label='dests')
#         if shows['direct']:
#             ax.plot([0],[0], color=colors['direct'],marker='o',
#                     markersize=20,markeredgecolor='k',linewidth=0, label='direct centers')
#             ax.plot([0,1],[0,1], color=colors['direct'],linestyle='-',
#                     markersize=20,markeredgecolor='k',linewidth=lw1, label='direct regions')
#         if shows['shuttle']:
#             ax.plot([0],[0], color=colors['shuttle_nodes'],marker='o',markersize=20,markeredgecolor='k',
#                     linewidth=0, label='shuttle stops')
#             ax.plot([0,1],[0,1], color=colors['shuttle'],linestyle='-',
#                     markersize=20,markeredgecolor='k',linewidth=lw1, label='shuttle regions')
#         if shows['lines']:
#             ax.plot([0,1],[0,1], color=colors['lines'],linestyle='-',
#                     markersize=20,markeredgecolor='k',linewidth=lw1, label='transit lines')
    
#         lw1 = 10; lw2 = 6;
#         if shows['drive']:
#             ax.plot([0],[0], color=colors['drive'], linewidth=lw1, label='drive traffic')
#         if shows['walk']:
#             ax.plot([0],[0], color=colors['walk'], linewidth=lw2, label='walk traffic')
#         if shows['ondemand']:
#             ax.plot([0],[0], color=colors['ondemand'],linewidth=lw1, label='ondemand routes')
#         if shows['gtfs']:
#             ax.plot([0],[0], color=colors['gtfs'],linewidth=lw1, label='transit traffic')
#             ax.plot([0],[0], color=[0,0,0],linewidth=lw2, label='transit lines')
#         # leg = ax.legend();
#         # plt.show()
#         if shows['legend']:
#             ax.legend(fancybox=True, fontsize=fontsize,
#                       framealpha=0.6, shadow=True,
#                       borderpad=1,loc = 'upper left')
#         end_time = time.time()
#         print('time to draw graph: ',end_time-start_time)
    
#     if False:
#         lw1 = 0; lw2 = 6; markersize = 32;
#         ax.plot([0],[0], color=[1,1,1],marker='o', markersize=markersize,
#                 markeredgecolor='w',linewidth=0, label='Full Transit Graph')    
#         ax.legend(fancybox=True, fontsize=24, framealpha=0.6, shadow=True, borderpad=1,loc = 'upper left')
    
#     # fileName = 'current.pdf'
#     printFigs = False;
#     # plt.axis('off')
#     plt.savefig(fileName,bbox_inches='tight',pad_inches = 0,transparent=False)

#     return {'fig':fig,'ax':ax}
    
