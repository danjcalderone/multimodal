import numpy as np
import numpy.linalg as mat
import scipy as sp
import scipy.linalg as smat
import cvxpy as cp

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import peartree as pt #turns GTFS feed into a graph
import folium

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

from matplotlib.patches import FancyArrow
from itertools import product 
from random import sample
from shapely.geometry import Polygon, Point

import time
import warnings
warnings.filterwarnings('ignore')

import random
import sys
# sys.path.append('/Users/dan/Documents/code');
# from drawing import * 
# sys.path.remove('/Users/dan/Documents/code');



##### NODE CONVERSION ##### NODE CONVERSION ##### NODE CONVERSION 
##### NODE CONVERSION ##### NODE CONVERSION ##### NODE CONVERSION 
##### NODE CONVERSION ##### NODE CONVERSION ##### NODE CONVERSION 


def convertNode(node,from_type,to_type,NODES):
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
    if from_type == 'drive':
        out = NODES['drive'][to_type][node]
    if from_type == 'transit':
        out = NODES['transit'][to_type][node]
    if from_type == 'walk':
        out = NODES['walk'][to_type][node]
    if from_type == 'ondemand':
        out = NODES['ondemand'][to_type][node]        
    return out



def findNode(node,from_type,to_type,NODES):
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

# def find_close_node(node,graph,find_in_graph):
#     lon = graph.nodes[node]['x'];
#     lat = graph.nodes[node]['y'];
#     found_node = ox.distance.nearest_nodes(find_in_graph, lon,lat); #ORIG_LOC[i][0], ORIG_LOC[i][1]);
#     xx = find_in_graph.nodes[found_node]['x'];
#     yy = find_in_graph.nodes[found_node]['y'];
#     if not(np.abs(xx-lon) + np.abs(yy-lat) <= 0.1):
#         found_node = None;
#     return found_node







def addNodeToDF(node,mode,GRAPHS,NODES):
    if not(node in NODES['all'][mode]):
        node_tags = {'drive':[find_close_node(node,GRAPHS[mode],GRAPHS['drive'])],
                     'walk':[find_close_node(node,GRAPHS[mode],GRAPHS['walk'])],
                     'transit':[find_close_node(node,GRAPHS[mode],GRAPHS['transit'])],
                     'ondemand':[find_close_node(node,GRAPHS[mode],GRAPHS['ondemand'])]}    
        node_index = 'node'+str(len(NODES['all'].index));
        new_nodes = pd.DataFrame(node_tags,index=[node_index])
        NODES['all'] = NODES['all'].append(new_nodes);


# def addNodeToDF(node,mode,GRAPHS,NODES):
#     if not(node in NODES['all'][mode]):
#         node_tags = {'drive':[find_close_node(node,GRAPHS[mode],GRAPHS['drive'])],
#                      'walk':[find_close_node(node,GRAPHS[mode],GRAPHS['walk'])],
#                      'transit':[find_close_node(node,GRAPHS[mode],GRAPHS['transit'])],
#                      'ondemand':[find_close_node(node,GRAPHS[mode],GRAPHS['ondemand'])]}    
#         node_index = 'node'+str(len(NODES['all'].index));
#         new_nodes = pd.DataFrame(node_tags,index=[node_index])
#         NODES['all'] = NODES['all'].append(new_nodes);

    
def updateNodesDF(NODES):
    NODES['transit'] = NODES['all'].copy();
    NODES['drive'] = NODES['all'].copy();
    NODES['walk'] = NODES['all'].copy();
    NODES['ondemand'] = NODES['all'].copy();

    NODES['transit']['transit2'] = NODES['transit']['transit'].copy(); 
    NODES['drive']['drive2'] = NODES['drive']['drive'].copy();
    NODES['walk']['walk2'] = NODES['walk']['walk'].copy();
    NODES['ondemand']['ondemand2'] = NODES['ondemand']['ondemand'].copy(); 

    NODES['transit'] = NODES['transit'].set_index('transit2')
    NODES['drive'] = NODES['drive'].set_index('drive2')
    NODES['walk'] = NODES['walk'].set_index('walk2')
    NODES['ondemand'] = NODES['ondemand'].set_index('ondemand2')
    
    NODES['transit'] = NODES['transit'][~NODES['transit'].index.duplicated(keep='first')];
    NODES['drive'] = NODES['drive'][~NODES['drive'].index.duplicated(keep='first')];
    NODES['walk'] = NODES['walk'][~NODES['walk'].index.duplicated(keep='first')];
    NODES['ondemand'] = NODES['ondemand'][~NODES['ondemand'].index.duplicated(keep='first')];

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






#### BUS STOP NODES #### BUS STOP NODES #### BUS STOP NODES #### BUS STOP NODES #### 
#### BUS STOP NODES #### BUS STOP NODES #### BUS STOP NODES #### BUS STOP NODES #### 
#### BUS STOP NODES #### BUS STOP NODES #### BUS STOP NODES #### BUS STOP NODES #### 




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

  
  
def makeTrip(modes,nodes,node_types,NODES,deliveries = []):
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

def makeTrip2(modes,nodes,NODES,deliveries=[]): #,node_types,NODES,deliveries = []):
    trip = {};
    segs = [];
    deliv_counter = 0;
    for i,mode in enumerate(modes):
        segs.append({});
        segs[-1]['mode'] = mode;
        segs[-1]['start_nodes'] = []; #nodes[i]
        segs[-1]['end_nodes'] = []; #nodes[i+1]
        segs[-1]['path'] = [];

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


# def makeTrip2(modes,nodes,NODES,deliveries=[]): #,node_types,NODES,deliveries = []):
#     trip = {};
#     segs = [];
#     deliv_counter = 0;
#     for i,mode in enumerate(modes):
#         segs.append({});
#         segs[-1]['mode'] = mode;
#         segs[-1]['start_nodes'] = []; #nodes[i]
#         segs[-1]['end_nodes'] = []; #nodes[i+1]
#         segs[-1]['path'] = [];

#         for j,node in enumerate(nodes[i]['nodes']):
#             node2 = convertNode(node,nodes[i]['type'],mode,NODES)
#             segs[-1]['start_nodes'].append(node2)
#         for j,node in enumerate(nodes[i+1]['nodes']):
#             node2 = convertNode(node,nodes[i+1]['type'],mode,NODES)
#             segs[-1]['end_nodes'].append(node2);
                
# #         segs[-1]['start_types'] = node_types[i]
# #         segs[-1]['end_types'] = node_types[i+1]

#         if mode == 'ondemand':
#             segs[-1]['delivery'] = deliveries[deliv_counter]
#             deliv_counter = deliv_counter + 1;
            
#     trip = {}
#     trip['current'] = {};
#     trip['structure'] = segs; 
#     trip['current']['cost'] = 100000000;
#     trip['current']['traj'] = []; 
#     return trip



def querySeg(start,end,mode,PERSON,NODES,GRAPHS,WORLD):
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
    cost = 0;
    if not((start,end) in WORLD[mode]['trips'].keys()):  
        planSeg(start,end,mode,GRAPHS,WORLD,mass=0);
    # print(PERSON['prefs'].keys())
    for l,factor in enumerate(PERSON['prefs'][mode]):
        cost = WORLD[mode]['trips'][(start,end)]['costs']['current_'+factor] # possibly change for delivery
        diff = cost-PERSON['prefs'][mode][factor]
        cost = cost + PERSON['weights'][mode][factor]*diff;
    path = WORLD[mode]['trips'][(start,end)]['current_path']
    return cost,path

def planSeg(source,target,mode,GRAPHS,WORLD,mass=1,track=False):
    trip = (source,target);
    GRAPH = GRAPHS[mode];
    try:
        temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c');
        distance = temp[0];
        path = temp[1]; 
    except: 
        #print('no path found for bus trip ',trip,'...')
        distance = 1000000;
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
                edge_cost = 1.*edge_mass + 1.;
                WORLD[mode]['edge_masses'][edge][-1] = edge_mass;
                WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                WORLD[mode]['current_edge_costs'][edge] = edge_cost;    
                WORLD[mode]['current_edge_masses'][edge] = edge_mass;



def queryTrip(TRIP,PERSON,NODES,GRAPHS,WORLD):
    trip_cost = 0;
    costs_to_go = [];
    next_inds = [];
    next_nodes = [];
    for k,SEG in enumerate(TRIP['structure'][::-1]):
        end_nodes = SEG['end_nodes'];
        start_nodes = SEG['start_nodes'];
#         print(SEG['end_types'])
#         end_types = SEG['end_types'];
#         start_types = SEG['start_types'];
        mode = SEG['mode'];
        NETWORK = WORLD[mode];
        costs_to_go.append([]);
        next_inds.append([]);
        next_nodes.append([]);
#         end_type = end_types[k]
#         start_type = start_types[k];
        for j,end in enumerate(end_nodes):
            possible_leg_costs = np.zeros(len(start_nodes));
            for i,start in enumerate(start_nodes):                
                end_node = end;#int(NODES[end_type][mode][end]);
                start_node = start;#int(NODES[start_type][mode][start]);
                cost,_ = querySeg(start_node,end_node,mode,PERSON,NODES,GRAPHS,WORLD)
                possible_leg_costs[i] = cost;
            ind = np.argmax(possible_leg_costs)
            next_inds[-1].append(ind)
            next_nodes[-1].append(end_nodes[ind])
            costs_to_go[-1].append(possible_leg_costs[ind]);

    init_ind = np.argmin(costs_to_go[-1]);
    init_cost = costs_to_go[-1][init_ind]
    init_node = TRIP['structure'][0]['start_nodes'][init_ind];

    costs_to_go = costs_to_go[::-1];
    next_inds = next_inds[::-1]
    next_nodes = next_nodes[::-1]
    

    inds = [init_ind];
    nodes = [init_node];
#     print(inds[-1])
#     print(next_nodes)
#     print(next_inds[inds[-1]])

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

#     print(costs_to_go)
#     print(next_inds)
#     print(next_nodes)
#     print(inds)
#     print(nodes)    
#     asdf
    
    trip_cost = costs_to_go[init_ind][0];
    TRIP['current']['cost'] = trip_cost
    TRIP['current']['traj'] = nodes;

    




##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES 
##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES 
##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES 
##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES 
##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES 
##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES ##### UPDATE CHOICES 

def update_choices(PEOPLE, DELIVERY, WORLD, version=1):
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
    

def update_choices2(PEOPLE, DELIVERY, NODES, GRAPHS, WORLD, version=1,verbose=False,takeall=False):
    if verbose: print('updating choices')
    ## clear options
#     for o,opt in enumerate(WORLD):
#         WORLD[opt]['people'] = [];
        
    people_chose = {};
    for i,person in enumerate(PEOPLE):
        PERSON = PEOPLE[person];
        
        delivery_grp = PERSON['delivery_grp'];        
        delivery_grp_inital = PERSON['delivery_grp_initial'];
        delivery_grp_final = PERSON['delivery_grp_final'];        
        COMPARISON = [];

        for k,TRIP in enumerate(PERSON['trips']):
            queryTrip(TRIP,PERSON,NODES,GRAPHS,WORLD)
            COMPARISON.append(TRIP['current']['cost']);
                            

        ind = np.argmin(COMPARISON);

        current_choice = ind;
        current_cost = COMPARISON[ind]

        PERSON['current_choice'] = current_choice;
        PERSON['current_cost'] = current_cost;

        PERSON['choice_traj'].append(current_choice);
        PERSON['cost_traj'].append(current_cost);
        # updating world choice...


        if takeall==False:
            tripstotake = [PERSON['trips'][ind]];
        else:
            tripstotake = PERSON['trips'];

        for _,CHOSEN_TRIP in enumerate(tripstotake):
            for k,SEG in enumerate(CHOSEN_TRIP['structure']):
                mode = SEG['mode'];
        #             start = SEG['start_nodes'][SEG['opt_start']]
        #             end = SEG['end_nodes'][SEG['opt_end']]
                start = SEG['opt_start']
                end = SEG['opt_end']
                #print(WORLD[mode]['trips'][(start,end)])
                #if not(mode in ['transit']):
                WORLD[mode]['trips'][(start,end)]['mass'] = WORLD[mode]['trips'][(start,end)]['mass'] + PERSON['mass'];
                WORLD[mode]['trips'][(start,end)]['active'] = True; 
                WORLD[mode]['active_trips'].append((start,end));

    

#         WORLD[current_choice]['people'].append(person);
        # WORLD[current_choice]['sources'].append(PERSON['sources'][current_choice])
        # WORLD[current_choice]['targets'].append(PERSON['targets'][current_choice])        
            
#         WORLD[current_choice]['sources'][''.append(node);
#         WORLD[current_choice]['targets'].append(PERSON['target']);    
            
########### WORLD[current_choice]['sources'][''.append(node); ####################
########### WORLD[current_choice]['targets'].append(PERSON['target']); ###########

########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########
########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ########### WORLD ###########


####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 
####### world of transit ####### world of transit ####### world of transit ####### world of transit ####### world of transit 

def world_of_transit_graph(WORLD,PEOPLE,GRAPHS,verbose=False):
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
    removeMassFromEdges('walk',WORLD,GRAPHS) 

    segs = WORLD['walk']['active_trips']
    mode = 'transit'
    for i,seg in enumerate(segs):
        source = seg[0];
        target = seg[1];
        trip = (source,target)
        planSeg(source,target,mode,GRAPHS,WORLD,mass=1,track=True);
        WORLD[mode]['trips'][trip]['active'] = False;    
    WORLD[mode]['active_trips']  = [];

    # source_tags = ['transit_walk1',
    #                'transit_drive1',
    #                'transit_ondemand1',
    #                'transit_walk1',
    #                'transit_drive1',
    #                'transit_ondemand1',
    #                'transit_walk1',
    #                'transit_drive1',
    #                'transit_ondemand1']
    
    # target_tags = ['transit_walk2',
    #                'transit_walk2',
    #                'transit_walk2',
    #                'transit_drive2',
    #                'transit_drive2',
    #                'transit_drive2',
    #                'transit_ondemand2',
    #                'transit_ondemand2',
    #                'transit_ondemand2']
    

    # removeMassFromEdges('transit',WORLD,GRAPHS)
    
    # for i,person in enumerate(WORLD['transit']['people']):
    #     for j,source_tag in enumerate(source_tags):
    #         target_tag = target_tags[j];
    #         PERSON = PEOPLE[person];
    #         source = PERSON['nodes'][source_tag]['transit'];
    #         target = PERSON['nodes'][target_tag]['transit'];
    #         #temp = nx.multi_source_dijkstra(GRAPH, sources, target=target, weight='c'); #
    #         #temp = nx.single_target_shortest_path(GRAPH,people_nodes[0])
    #         trip = (source,target);
    #         mass = 1;        
    #         try:
    #             temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c');
    #             distance = temp[0];
    #             path = temp[1]; 
    #         except: 
    #             #print('no path found for bus trip ',trip,'...')
    #             distance = 1000000;
    #             path = [];

    #         if not(trip in WORLD['transit']['trips'].keys()):
    #             TRANSIT['trips'][trip] = {};
    #             TRANSIT['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
    #             TRANSIT['trips'][trip]['path'] = [];
    #             TRANSIT['trips'][trip]['mass'] = 0;

    #         TRANSIT['trips'][trip]['costs']['time'].append(distance)
    #         TRANSIT['trips'][trip]['costs']['money'].append(1)
    #         TRANSIT['trips'][trip]['costs']['conven'].append(1)
    #         TRANSIT['trips'][trip]['costs']['switches'].append(1)
    #         TRANSIT['trips'][trip]['path'].append(path);

    #         # if True: #PERSON['current_choice'] in seg_types[tt]:
    #         #     for j,node in enumerate(path):
    #         #         if j < len(path)-1:
    #         #             edge = (path[j],path[j+1],0)
    #         #             edge_mass = TRANSIT['edge_masses'][edge][-1] + mass;
    #         #             edge_cost = 1.*edge_mass + 1.;
    #         #             TRANSIT['edge_masses'][edge][-1] = edge_mass;
    #         #             TRANSIT['edge_costs'][edge][-1] = edge_cost;
    #         #             TRANSIT['current_edge_costs'][edge] = edge_cost;    
    #         #             TRANSIT['current_edge_masses'][edge] = edge_mass;    
    #         TRANSIT['trips'][trip]['active'] = False;




def createBasicNetwork():
    print('asdf')

def removeMassFromEdges(mode,WORLD,GRAPHS):
    NETWORK = WORLD[mode]
    GRAPH = GRAPHS[mode];
    NETWORK['current_edge_costs'] = {};
    NETWORK['current_edge_masses'] = {};    
    for e,edge in enumerate(GRAPH.edges):
        NETWORK['current_edge_costs'][edge] = 0;
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




###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand 
###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand ###### world of ondemand 




def world_of_ondemand(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=False):    
    if verbose: print('starting on-demand computations...')    
    #### PREV: tsp_wgrps
    #lam = grads['lam']
    #pickups = current_pickups(lam,all_pickup_nodes) ####
    #nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
    GRAPH = GRAPHS['ondemand'];
    ONDEMAND = WORLD['ondemand'];

    trips_to_plan = WORLD['ondemand']['active_trips']
    divideDelivSegs(trips_to_plan,DELIVERY,GRAPHS,WORLD);

    for i,delivery in enumerate(DELIVERY):
        DELIV = DELIVERY[delivery];

        active_trips = DELIV['active_trips'];

        start = DELIV['nodes']['source'];
        sources = []; targets = [];
        [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
        [targets.append(trip[1]) for _,trip in enumerate(active_trips)]

        planDelivSegs(sources,targets,start,DELIVERY,GRAPHS,WORLD,track=False);

    for i,delivery in enumerate(DELIV):
        clients = [];
        sources = [];
        targets = [];
        # for j,client in enumerate(DELIV[delivery]['people']):
        #     PERSON = PEOPLE[client]
        #     if PERSON['choice'] == 'ondemand':
        #         clients.append(client);
        #         sources.append(DELIV[delivery]['sources'][j]);
        #         targets.append(DELIV[delivery]['targets'][j]);
        
        pickups = sources+targets;
        sink = DELIV[delivery]['nodes']['source'];
        ordered_pickups = order_pickups(GRAPH,pickups,sink);
        ordered_pickups = ordered_pickups[::-1] 
        tsp = traveling_salesman(GRAPH,ordered_pickups,sink) ####### 
        DELIV[delivery]['tsp'] = tsp.copy();
        DELIV[delivery]['costs'] = {'time':[tsp['cost']],'money':[1],'conven':[1],'switches':[1]}
    
    GRAPH = GRAPHS['ondemand']
    ### COMPUTE
    pickups= order_pickups2(GRAPH,sources,targets,start);
    plan = traveling_salesman(GRAPH,pickups[:-1],pickups[-1]);

    ### DOCUMENT TRIPS
    mode = 'ondemand';
    for i,source in enumerate(sources):
        target = targets[i];
        trip = (source,trip);
        mass = 1;
        distance = plan['cost'];
        path = plan['route'];
        if not(trip in WORLD[mode]['trips'].keys()):
            WORLD[mode]['trips'][trip] = {};
            WORLD[mode]['trips'][trip]['costs'] = {'time':[],'money':[],'conven':[],'switches':[]}
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


        for j,source in enumerate(sources):
            target = targets[j];
            trip = (source,target);
            ONDEMAND['trips'][trip] = {};
            ONDEMAND['trips'][trip]['costs'] = {'time':[tsp['cost']],'money':[1],'conven':[1],'switches':[1]};
            ONDEMAND['trips'][trip]['tsp'] = tsp.copy();
            ONDEMAND['trips'][trip]['delivery'] = delivery;
            ONDEMAND['trips'][trip]['path'] = tsp.copy(); # NEW
            ONDEMAND['trips'][trip]['mass'] = 0.; 

    DELIV['active_trips']  = [];
            
    DELIV = DELIVERY['shuttle'];

    for i,delivery in enumerate(DELIV):
        clients = [];
        sources = [];
        targets = [];
        # for j,client in enumerate(DELIV[delivery]['people']):
        #     PERSON = PEOPLE[client]
        #     if PERSON['choice'] == 'ondemand':
        #         clients.append(client);
        #         sources.append(DELIV[delivery]['sources'][j]);
        #         targets.append(DELIV[delivery]['targets'][j]);
        
        pickups = sources+targets;
        sink = DELIV[delivery]['nodes']['source'];
        ordered_pickups = order_pickups(GRAPH,pickups,sink);
        ordered_pickups = ordered_pickups[::-1]        
        tsp = traveling_salesman(GRAPH,ordered_pickups,sink) #######        
        DELIV[delivery]['tsp'] = tsp.copy();
        DELIV[delivery]['costs'] = {'time':[tsp['cost']],'money':[1],'conven':[1],'switches':[1]}
        
        for j,source in enumerate(sources):
            target = targets[j];
            trip = (source,target);
            ONDEMAND['trips'][trip] = {};
            ONDEMAND['trips'][trip]['costs'] = {'time':[tsp['cost']],'money':[1],'conven':[1],'switches':[1]};
            ONDEMAND['trips'][trip]['tsp'] = tsp.copy();
            ONDEMAND['trips'][trip]['delivery'] = delivery;
            ONDEMAND['trips'][trip]['path'] = tsp.copy(); # NEW             
            ONDEMAND['trips'][trip]['mass'] = 1.;

    DELIV['active_trips']  = [];      


def divideDelivSegs(trips,DELIVERY,GRAPHS,WORLD,maxTrips = 1000):
    GRAPH = GRAPHS['ondemand'];
    for i,trip in enumerate(trips):
        start = trip[0];
        xloc = GRAPH.nodes[start]['x']
        yloc = GRAPH.nodes[start]['y']
        end = trip[1];
        #max_dist = 100000000000;
        dists = [100000000000000];
        delivs = [None];
        current_deliv = None;
        for j,deliv in enumerate(DELIVERY):
            DELIV = DELIVERY[deliv];
            loc0 = DELIV['loc'];
            dist = np.sqrt((xloc-loc0[0])*(xloc-loc0[0])+(yloc-loc0[1])*(yloc-loc0[1]));

            ind = 0; spot_found = False;
            for ll in range(len(dists)):
                if dist < dists[ll]:
                    dists.insert(ll,dist);
                    delivs.insert(ll,deliv);
            # if dist < max_dist:
            #     current_deliv = deliv;
            #     max_dist = dist;
        dists = dists[:-1]
        delivs = delivs[:-1]
        for j,deliv in enumerate(delivs):
            DELIV = DELIVERY[deliv];
            if len(DELIV['active_trips'])<maxTrips:
                DELIV['active_trips'].append(trip)

def planDelivSegs(sources,targets,start,DELIV,GRAPHS,WORLD,maxSegs=100,track=False):
    GRAPH = GRAPHS['ondemand']
    maxind = np.min([len(sources),len(targets),maxSegs]);


    pickups= order_pickups2(GRAPH,sources[:maxind],targets[:maxind],start);
    plan = traveling_salesman(GRAPH,pickups[:-1],pickups[-1]);
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



def world_of_ondemand2(WORLD,PEOPLE,DELIVERY,GRAPHS,verbose=False):    
    if verbose: print('starting on-demand computations...')    
    #### PREV: tsp_wgrps
    #lam = grads['lam']
    #pickups = current_pickups(lam,all_pickup_nodes) ####
    #nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
    GRAPH = GRAPHS['ondemand'];
    ONDEMAND = WORLD['ondemand'];

    trips_to_plan = WORLD['ondemand']['active_trips']
    DELIVERY0 = DELIVERY['direct'];
    maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));
    divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD,maxTrips);
    for i,delivery in enumerate(DELIVERY0):
        DELIV = DELIVERY0[delivery];
        active_trips = DELIV['active_trips'];
        sources = []; targets = [];
        start = DELIV['nodes']['source'];        
        [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
        [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
        planDelivSegs(sources,targets,start,DELIV,GRAPHS,WORLD,track=False);
        path = DELIV['current_path'];          
        for j,node in enumerate(path):
            if j < len(path)-1:
                edge = (path[j],path[j+1],0)
                edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
                ONDEMAND['current_edge_masses'][edge] = edge_mass;
            


        # DELIV['active_trips']  = [];

    DELIVERY0 = DELIVERY['shuttle'];
    maxTrips = len(trips_to_plan)/len(list(DELIVERY0.keys()));    
    divideDelivSegs(trips_to_plan,DELIVERY0,GRAPHS,WORLD);
    for i,delivery in enumerate(DELIVERY0):
        DELIV = DELIVERY0[delivery];
        active_trips = DELIV['active_trips'];
        sources = []; targets = [];
        start = DELIV['nodes']['source'];        
        [sources.append(trip[0]) for _,trip in enumerate(active_trips)]
        [targets.append(trip[1]) for _,trip in enumerate(active_trips)]
        planDelivSegs(sources,targets,start,DELIV,GRAPHS,WORLD,track=False);
        DELIV['active_trips']  = [];
        path = DELIV['current_path'];          
        for j,node in enumerate(path):
            if j < len(path)-1:
                edge = (path[j],path[j+1],0)
                edge_mass = 1; #ONDEMAND['edge_masses'][edge][-1] + mass;
                ONDEMAND['current_edge_masses'][edge] = edge_mass;        





            

####### world of drive ####### world of drive ####### world of drive ####### world of drive ####### world of drive 
####### world of drive ####### world of drive ####### world of drive ####### world of drive ####### world of drive             

            
def world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=False): #graph,costs,sources, targets):
    if verbose: print('starting driving computations...')
    DRIVE = WORLD['drive'];
    GRAPH = GRAPHS[DRIVE['graph']];
    if not('edge_masses' in DRIVE.keys()):
        DRIVE['edge_masses'] = {};
        DRIVE['edge_costs'] = {};
        DRIVE['current_edge_costs'] = {};
        DRIVE['edge_a0'] = {};
        DRIVE['edge_a1'] = {};               
        for e,edge in enumerate(GRAPH.edges):
            DRIVE['edge_masses'][edge] = [0]
            DRIVE['edge_costs'][edge] = [1]
            DRIVE['current_edge_costs'][edge] = 1;
            DRIVE['edge_a0'][edge] = 1;
            DRIVE['edge_a1'][edge] = 1;               
    else: 
        for e,edge in enumerate(GRAPH.edges):
            DRIVE['edge_masses'][edge].append(0)
            DRIVE['edge_costs'][edge].append(0)
            DRIVE['current_edge_costs'][edge] = 1;            
        
    if 'current_edge_costs' in DRIVE.keys():
        current_costs = DRIVE['current_edge_costs'];
    else: 
        current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}
        
    nx.set_edge_attributes(GRAPH,current_costs,'c');
    removeMassFromEdges('drive',WORLD,GRAPHS)

    segs = WORLD['drive']['active_trips']

    mode = 'drive'
    for i,seg in enumerate(segs):
        source = seg[0];
        target = seg[1];
        trip = (source,target)
        planSeg(source,target,mode,GRAPHS,WORLD,mass=1,track=True);
        WORLD[mode]['trips'][trip]['active'] = False;    
    WORLD[mode]['active_trips']  = [];


####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
####### world of walk ####### world of walk ####### world of walk ####### world of walk ####### world of walk 
                        
def world_of_walk(WORLD,PEOPLE,GRAPHS,verbose=False): #graph,costs,sources, targets):
    if verbose: print('starting walking computations...')
    mode = 'walk'
    WALK = WORLD[mode];
    GRAPH = GRAPHS[WALK['graph']];    
    if not('edge_masses' in WALK.keys()):
        WALK['edge_masses'] = {};
        WALK['edge_costs'] = {};
        WALK['current_edge_costs'] = {};
        WALK['edge_a0'] = {};
        WALK['edge_a1'] = {};
        for e,edge in enumerate(GRAPH.edges):
            WALK['edge_masses'][edge] = [0]
            WALK['edge_costs'][edge] = [1]
            WALK['current_edge_costs'][edge] = 1;
            WALK['edge_a0'][edge] = 1;
            WALK['edge_a1'][edge] = 1;               

    else: 
        for e,edge in enumerate(GRAPH.edges):
            WALK['edge_masses'][edge].append(0)
            WALK['edge_costs'][edge].append(0)
            WALK['current_edge_costs'][edge] = 1;

        
    if 'current_edge_costs' in WALK.keys():
        current_costs = WALK['current_edge_costs'];
    else: 
        current_costs = {k:v for k,v in zip(GRAPH.edges,np.ones(len(GRAPH.edges)))}
    nx.set_edge_attributes(GRAPH,current_costs,'c');     
    removeMassFromEdges('walk',WORLD,GRAPHS)  

    segs = WORLD['walk']['active_trips']

    mode = 'walk'
    for i,seg in enumerate(segs):
        source = seg[0];
        target = seg[1];
        trip = (source,target)
        planSeg(source,target,mode,GRAPHS,WORLD,mass=1,track=True);
        WORLD[mode]['trips'][trip]['active'] = False;    
    WORLD[mode]['active_trips']  = [];


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



def compute_masses(cost):
    masses = {};
    for o,opt in enumerate(costs):
        cost = costs[opt];
        masses[opt] = 0;
        for i,person in enumerate(Mlocs):
            if Mlocs[person][opt] >= cost:
                masses[opt] = masses[opt] + 1;
    return masses    
    
def compute_masses(cost):
    masses = {};
    for o,opt in enumerate(costs):
        cost = costs[opt];
        masses[opt] = 0;
        for i,person in enumerate(Mlocs):
            if Mlocs[person][opt] >= cost:
                masses[opt] = masses[opt] + 1;
    return masses
            
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


##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 
##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND ##### ONDEMAND 





# def order_pickups2(graph,sources,targets,start,typ='straight'):
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
    

def order_pickups2(graph,sources,targets,start,typ='straight'):
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
def traveling_salesman(graph,pickups,sink):
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




##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND 
##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND 
##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND 
##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND ##### OLD ONDEMAND 



# def next_node(graph,nodes,current,sink,typ = 'straight'):
#     remaining_nodes = nodes;
#     sink_loc = np.array([graph.nodes[sink]['x'],graph.nodes[sink]['y']])
#     current_loc = np.array([graph.nodes[current]['x'],graph.nodes[current]['y']])
#     dists = [];
    
#     for i,node in enumerate(remaining_nodes):
#         if typ=='straight':        
#             x = graph.nodes[node]['x'];
#             y = graph.nodes[node]['y'];
#             loc = np.array([x,y]);
#             dist_to_sink = mat.norm(loc-sink_loc);
#             dist_to_current = mat.norm(loc-current_loc);
#             if i == 0:
#                 dist_total = dist_to_sink; # + dist_to_current;
#             else:
#                 dist_total = dist_to_current
#             dists.append(dist_total)
                    
#     dists = np.array(dists);
#     ind = np.where(dists==np.min(dists))[0][0]
#     return {'ind':ind,'node':nodes[ind]}
    

# # DONE 
# def order_pickups(graph,nodes,sink,typ='straight'):
#     nodes_left = nodes.copy();
#     path = [];
#     current = sink;
#     for i in range(len(nodes)):
#         next_stop = next_node(graph,nodes_left,current,sink,typ=typ);
#         path.append(next_stop['node']);
#         nodes_left.pop(next_stop['ind'])
#         current = path[-1]
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



# def traveling_salesman_groups(graph,pickups,dropoffs):
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
#         path = nx.shortest_path(graph, source=start_node, target=end_node,weight = 'c'); #, weight=None);
#         cost = cost + nx.shortest_path_length(graph, source=start_node, target=end_node, weight='c')
#         costs.append(cost)
#         route = route + path
#     current_len.append(len(route))
#     return {'route':route,'cost':cost,'current_len':current_len}


# def tsp_dual(grads):  
#     lam = grads['lam']
#     pickups = current_pickups(lam,all_pickup_nodes)
#     nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,grads['c'])},'c')
#     #tsp = nx.approximation.traveling_salesman_problem(graph, nodes=curr,cycle=True, method=None)
#     ordered_pickups = order_pickups(graph,pickups,sink);
#     ordered_pickups = ordered_pickups[::-1]
#     tsp = traveling_salesman(graph,ordered_pickups,sink)
#     xvals = np.ones(ne);
#     lamval = tsp['cost'];
#     return {'x':xvals,'lam':lamval,'tsp':tsp,'pickups':pickups} #,'v':vvals}      


# def plan_travel_drive(start,end,plan=None):    
#     if plan == None:
#         path = nx.shortest_path(graph, source=start, target=end,weight = 'c'); #, weight=None);
#         cost = cost + nx.shortest_path_length(graph, source=start_node, target=end_node, weight='c')
#     costs.append(cost)
#     route = route + path
#     shortest_path #drive
    
# def plan_travel_ondemand(start,end,schedule):
#     plan = schedule[start];
#     trip = plan['trip']
#     cost = plan['cost']
#     return {'trip':trip,'cost':cost}

# def plan_travel_walktransit():
#     shortest_path #walk
#     access_transit_schedule
# def plan_travel_ondemandtransit():
#     access_ondemand_schedule
#     access_transit_schedule

#### WORLD STATE #### WORLD STATE #### WORLD STATE #### WORLD STATE #### WORLD STATE 
#### WORLD STATE #### WORLD STATE #### WORLD STATE #### WORLD STATE #### WORLD STATE 
#### WORLD STATE #### WORLD STATE #### WORLD STATE #### WORLD STATE #### WORLD STATE 


        #print(WORLD['ondemand']['costs'].keys())
    #pickups = WORLD['ondemand']['sources']
#     pickups = WORLD['ondemand']['people']    

#     resegment = False;
#     if resegment:
#         temp = segment_pickups(PEOPLE,WORLD,DELIVERY); #graph,pickups,num);
#         clients = temp['clients'];
#         sources = temp['sources'];
#         targets = temp['targets'];
#     else:
#         deliveries = list(DELIVERY.keys());
#         clients = [DELIVERY[delivery]['clients'] for j,delivery in enumerate(deliveries)]
#         sources = [DELIVERY[delivery]['sources'] for j,delivery in enumerate(deliveries)]
#         targets = [DELIVERY[delivery]['targets'] for j,delivery in enumerate(deliveries)]
        
#     planned_trips = [];
#     for i,group in enumerate(clients):
#         nodes = sources[i] + targets[i];
#         sink = delivery_nodes[i];
#         nx.set_edge_attributes(graph,{k:v for k,v in zip(graph.edges,np.ones(len(graph.edges)))},'c')        
#         ordered_pickups = order_pickups(graph,nodes,sink)
#         ordered_pickups = ordered_pickups[::-1];
#         tsp = traveling_salesman(graph,ordered_pickups,sink) #######
#         xvals = np.ones(ne);
#         lamval = tsp['cost'];
#         planned_trips.append({'x':xvals,'lam':lamval,'tsp':tsp,'pickups':pickups}) #,'v':vvals}          
#     return planned_trips

# def update_costs(PEOPLE,WORLD):
#     for i,node in enumerate(people_nodes):
#         PERSON = PEOPLE[node];
#         for o,opt in enumerate(PERSON['opts']):
#             for j,factor in enumerate(PERSON['factors']):
#                 PERSON['costs'][opt][factor] = WORLD[opt][node][factor]['cost'];
    



