
import os
#### path to folder with VRP solver module
path_to_VRPsolver = '/Users/dan/Documents/transit_webapp/' ### UPDATE FOR YOUR SYSTEM...
path_to_multimodalsim = '/Users/dan/Documents/multimodal/'

os.chdir(path_to_VRPsolver)
from core import db, optimizer
import config
from common.helpers import _internalrequest
os.chdir(path_to_multimodalsim)


import plotly.express as px
import plotly.graph_objects as go
import skimage.io as sio
from PIL import Image, ImageSequence
from plotly.subplots import make_subplots



import numpy as np
import numpy.linalg as mat
import scipy as sp
import scipy.linalg as smat
# import cvxpy as cp

import datetime
from datetime import datetime

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
# from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn import metrics
# from sklearn.cluster import DBSCAN

from scipy.spatial import ConvexHull, convex_hull_plot_2d

import time
import warnings
warnings.filterwarnings('ignore')

import random
import sys


import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pygris
import os


# os.chdir('/Users/dan/Documents/transit_webapp/')
# from core import db, optimizer
# import config
# from common.helpers import _internalrequest


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


def applyProgressiveWeight(value,progressive_weights):
    if value < 0:
        weight = progressive_weights[0];
    elif value < len(progressive_weights):
        weight = progressive_weights[int(np.floor(value))];
    else: 
        weight = progressive_weights[-1]
    return weight;

def setup_ondemand_service(PRE, NODES, LOCS, SIZES, GRAPHS, params, people_tags):
    
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
    num_people = len(people_tags);
    num_targets = num_people;

    if 'num_deliveries' in params:
        num_deliveries = params['num_deliveries']['delivery1'];
        num_deliveries2 = params['num_deliveries']['delivery2'];
    else:
        num_deliveries =  int(num_people/10);
        num_deliveries2 = int(num_people/10);


    ##### k-means clustering of the population locations to see where the different ondemand vehicles should go
    ### CHANGED TO RUN FOR PILOT REGION....
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
    delivery_transit_nodes = sample(list(bus_graph.nodes()), num_deliveries2)
    end_time = time.time();
    print('time to setup origins & dests: ',end_time - start_time)
    
    print("Done setting up on-demand service")
    
    return PRE, NODES, LOCS, SIZES


def generate_driver_dataframe(group, start_time, end_time, am_capacity=8, wc_capacity=2):
    # Create a DataFrame with driver information
    driver_data = {
        'group': group,
        'driver_start_time': [start_time],
        'driver_end_time': [end_time],
        'am_capacity': [am_capacity],
        'wc_capacity': [wc_capacity],
    }
    
    driver_df = pd.DataFrame(driver_data)
    return driver_df


def add_logistic_values (PRE, tag, 
                         DWT = 1, DWM = 0, DWC = 0, DWS = 0,
                         WWT = 1, WWM = 0, WWC = 0, WWS = 0,
                         TWT = 1 , TWM = 0, TWC = 0, TWS = 0, 
                         OWT = 1, OWM = 0, OWC = 0, OWS = 0):

            # Add fields for 'drive'
            PRE[tag]['drive_weight_time'] = DWT
            PRE[tag]['drive_weight_money'] = DWM
            PRE[tag]['drive_weight_conven'] = DWC
            PRE[tag]['drive_weight_switches'] = DWS

            # Add fields for 'walk'
            PRE[tag]['walk_weight_time'] = WWT
            PRE[tag]['walk_weight_money'] = WWM
            PRE[tag]['walk_weight_conven'] = WWC
            PRE[tag]['walk_weight_switches'] = WWS

            # Add fields for 'ondemand'
            PRE[tag]['ondemand_weight_time'] = OWT
            PRE[tag]['ondemand_weight_money'] = OWM
            PRE[tag]['ondemand_weight_conven'] = OWC
            PRE[tag]['ondemand_weight_switches'] = OWS

            # Add fields for 'transit'
            PRE[tag]['transit_weight_time'] = TWT
            PRE[tag]['transit_weight_money'] = TWM
            PRE[tag]['transit_weight_conven'] = TWC
            PRE[tag]['transit_weight_switches'] = TWS
            
            return PRE




########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 
########### BASIC ########### BASIC ########### BASIC ########### BASIC ########### BASIC 

class DASHBOARD:
    def __init__(self,dash_folder,version='version0'): #,params,folder):
        # self.GRAPHS = params['GRAPHS']
        # self.CONVERTER = params['CONVERTER'];
        # self.ONDEMAND = params['ONDEMAND'];
        # self.NETWORKS = params['NETWORKS'];
        # self.SIZES = params['SIZES']
        # self.size = (0,0);

        self.DATAINDS = {}
        self.folder = dash_folder + '/'



        self.colors = {'drive': 'rgb(255,0,0)', 'walk': 'rgb(255,255,0)',
                       'ondemand': 'rgb(0,0,255)','gtfs': 'rgb(255,128,0)'};

        self.edgecolors = {'drive': 'rgb(255,0,0)', 'walk': 'rgb(0,0,0)',
                       'ondemand': 'rgb(0,0,255)','gtfs': 'rgb(255,128,0)'};



        self.opacities = {'all':0.3,'grp':0.3,'run':0.3,'indiv':0.3}
        

        self.version = version; #'version0'
        if self.version == 'version0':

            self.width = 800; self.height = 900; 
            self.numx = 14; self.numy = 12;
            # self.row_heights = [0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083];
            # self.column_widths = [0.25,0.025,0.13,0.025,0.07,0.025,0.25,0.025,0.13,0.025,0.07,0.025];

            # pads = 't','b','l','r';
            gpads = [0.03,0.03,0.,0.]

            pp = gpads[0]+gpads[1]; nn = 4;
            dy = 1.0/12;

            self.row_heights = [dy+gpads[0]-pp/nn,dy-pp/nn,dy-pp/nn,dy+gpads[1]-pp/nn,
                                dy+gpads[0]-pp/nn,dy-pp/nn,dy-pp/nn,dy+gpads[1]-pp/nn,
                                dy+gpads[0]-pp/nn,dy-pp/nn,dy-pp/nn,dy+gpads[1]-pp/nn]
            self.column_widths = [0.3,0.0,0.12,0.0,0.164,0.0,
                                  0.16,0.0,0.16,0.0,0.12,0.0,0.0,0.0]
            # list(0.15*np.ones(self.numx))
            self.SLIDERS = {};
            self.list_of_sliders = [];
            self.slider_names = ['drive','walk','ondemand','gtfs','ondemand_grp'];#,'gtfs_grp']; #'gtfs_run','ondemand_grp','ondemand_run']
            # self.slider_locs = [(0.7,0.3),(0.7,0.5),(0.9,0.1),(0.9,0.5)]; 
            self.slider_grid_locs = [[4,8],[8,8],[11.99,0],[11.99,6],[11.99,2]];#,[11.9,8]];
            self.slider_hpads = [[-0.1,0.],[-0.1,0.05],[0.0,0.0],[-0.1,0.1],[-0.1,0.1],[0.05,0.1]];
            self.slider_vpads = [[0.,0.],[0.,0.2],[0.,0.],[0.,0.],[0.,0.]];
            self.slider_grid_lengths = [3,3,1,3,1,1];
            # self.slider_locs = [(0.7,0.7),(0.7,0.5),(0.,0.),(0.5,0.)];
            self.slider_lengths = []
            # self.slider_col_lengths = [[0],[7],[8,]
                                # (0.1,0.1),(0.1,0.1),(0.1,0.1),(0.1,0.1)]; 
            self.BUTTONS = {};                                
            self.button_names = ['lines','drive','walk','ondemand','gtfs','source','target'];
            self.button_display_names = ['lines','drive','walk','ondemand','gtfs','source','target'];
            lns = [0.15,0.15,0.15,0.2,0.15,0.15];
            self.button_lengths = lns;
            self.button_locs = [[np.sum(lns[:0]),1.1],[np.sum(lns[:1]),1.1],
                                [np.sum(lns[:2]),1.1],[np.sum(lns[:3]),1.1],
                                [np.sum(lns[:4]),1.1],[np.sum(lns[:5]),1.1],[np.sum(lns[:6]),1.1],]; #,(0.7,0.5),(0.9,0.1),(0.9,0.5)]; 
            self.button_hpads = [[]]
            



            self.imgpad = [0.01,0.01,0.01,0.01]


            
            # pads = 't','b','l','r';
            rpads = [0.005,0.005,0.01,0.01]
            rmargs = [0,0,0,0];
            rpads2 = [0,0,0,0.01];
            rmargs2 = [0,0,0,0];

            modes = ['drive','walk','ondemand','gtfs'];
            factors = ['dist','time','money','switches'];
            self.factors = factors;
            inds1 = [0,1,2,3]
            inds2 = [0,1,2,3,4,5]
            rinds = {}; dinds = {}; 
            addtags = ['bar','box','bar_grp','box_grp','bar_run','box_run'];
            for i,mode in enumerate(modes):
                for j,factor in enumerate(factors):
                    for k,tag in enumerate(addtags):
                        fulltag = mode+'_'+factor+'_'+tag
                        rinds[fulltag] = [inds1[j],inds2[k]];                        
                        if mode in ['drive','walk']:
                            if tag == 'box': rinds[fulltag][1] = inds2[k+2]

                        temp = (1,1);
                        if mode in ['drive','walk'] and tag == 'bar': temp = (1,3);
                        if mode in ['gtfs'] and tag == 'bar': temp = (1,3);
                        dinds[fulltag] = temp;

            
            self.pltgrps = {};
            self.pltgrps['fig'] = {'tag':'fig','loc':(1,1),'inds':(1,1),'dinds':(1,1),'subplts':[]};
            modes = ['drive','walk','ondemand','gtfs'];
            # factors 
            inds1 = [1,5,9,9]; inds2 = [9,9,1,7];
            for i,mode in enumerate(modes):
                ind = (inds1[i],inds2[i])
                self.pltgrps[mode] = {'tag':mode,'inds':ind,'dinds':(4,6)};
                subplt_tags = [mode+'_'+factor+'_bar' for factor in factors]
                subplt_tags = subplt_tags + [mode+'_'+factor+'_box' for factor in factors]
                if mode in ['ondemand','gtfs']:
                    subplt_tags = subplt_tags +  [mode+'_'+factor+'_bar_grp' for factor in factors]
                    subplt_tags = subplt_tags + [mode+'_'+factor+'_box_grp' for factor in factors]
                    subplt_tags = subplt_tags + [mode+'_'+factor+'_bar_run' for factor in factors]
                    subplt_tags = subplt_tags + [mode+'_'+factor+'_box_run' for factor in factors]
                self.pltgrps[mode]['subplts'] = subplt_tags;
                # print(mode)
                # print(subplt_tags)

            self.subplts = {}
            self.subplts['fig'] = {'rloc'};
            modes = ['drive','walk','ondemand','gtfs'];
            factors = ['dist','time','money','switches'];

            

            # for temp in rinds: print(temp)
            # for temp in dinds: print(temp)
            
            for mode in modes:
                for factor in factors:
                    addtags = ['bar','box']
                    for tag in addtags:
                        fulltag = mode + '_' + factor + '_' + tag

                        rind = rinds[fulltag]; dind = dinds[fulltag];
                        rpad = rpads.copy(); rmarg = rmargs;
                        if tag == 'box': rpad = rpads2;

                        if factor == 'dist': rpad[0] = gpads[0];
                        if factor == 'switches': rpad[1] = gpads[1];
                        
                        self.subplts[fulltag] = {'rinds':rind,'dinds':dind,'rpads':rpad,'rmargs':rmarg}
                                                            # 'rloc':rlocs[factor + tag],
                                                            # 'rdims':rdims[factor + tag],

                    if mode in ['ondemand','gtfs']:
                        addtags = ['bar_grp','box_grp','bar_run','box_run'];
                        for tag in addtags:
                            fulltag = mode + '_' + factor + '_' + tag
                            rind = rinds[fulltag]; dind = dinds[fulltag];
                            rpad = rpads.copy(); rmarg = rmargs;
                            if tag == 'box_run' or tag == 'box_grp': rpad = rpads2;

                            if factor == 'dist': rpad[0] = gpads[0];
                            if factor == 'switches': rpad[1] = gpads[1];

                            self.subplts[fulltag] = {'rinds':rind,'dinds':dind,'rpads':rpad,'rmargs':rmarg}
                                                            # 'rloc':rlocs[factor + tag],
                                                            # 'rdims':rdims[factor + tag],


    def makeGrid(self):
        self.grid = [[{} for _ in range(self.numx)] for j in range(self.numy)]
        self.plts = {};

        self.plts['img1'] = {'inds':[1,1],'dinds':[8,8],'pads':self.imgpad};

        for pltgrp in self.pltgrps:
            PLTGRP = self.pltgrps[pltgrp];
            # gloc = PLTGRP['loc'];
            gind = PLTGRP['inds'];
            # gpad = PLTGRP['pads'];
            # gidims = PLTGRP['dinds'];
            subplts = PLTGRP['subplts'];
            for subplt in subplts:
                SUBPLT = self.subplts[subplt];
                # rlocs = SUBPLT['rloc']
                # rdims = SUBPLT['rdims'];
                rind = SUBPLT['rinds'];
                dind = SUBPLT['dinds'];
                ind = (gind[0]+rind[0],gind[1]+rind[1])
                rpad = SUBPLT['rpads'];
                pad = rpad
                self.plts[subplt] = {};
                self.plts[subplt]['inds'] = ind
                self.plts[subplt]['dinds'] = dind
                self.plts[subplt]['pads'] = pad;
                # self.plts[subplt]['loc'] = [gloc[0] + gidims[0]*rlocs[0],gloc[1]+gidims[1]*rlocs[1]];
                # self.plts[subplt]['dims'] = [gidims[0]*rdims[0],gidims[1]*rdims[1]];

        for plt in self.plts:
            PLT = self.plts[plt]
            ind = PLT['inds'];
            dind = PLT['dinds']
            pad = PLT['pads']; #marg = PLT['margs'];            
            self.grid[ind[0]-1][ind[1]-1] = {"rowspan":dind[0], "colspan":dind[1],"t": pad[0],"b": pad[1],"l": pad[2],"r": pad[3]}; #,"type": "image"}] + [None for _ in range(2*num_fig_cols-1)] + [{},{}]


    def show(self):
        fig = make_subplots(rows=self.numy,cols=self.numx,
                            column_widths = self.column_widths,
                            row_heights = self.row_heights,
                            horizontal_spacing=0.0,#list(0.02*np.ones(12)),
                            vertical_spacing=0.0,
                            specs=self.grid);
                            # print_grid=True); 


        # from plotly.subplots import make_subplots
        # import plotly.graph_objects as go

        # # Initialize figure with subplots
        # fig = make_subplots(
        #     rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
        # )

# Valid properties:

    # anchor
    #         If set to an opposite-letter axis id (e.g. `x2`, `y`),
    #         this axis is bound to the corresponding opposite-letter
    #         axis. If set to "free", this axis' position is
    #         determined by `position`.
    #     automargin
    #         Determines whether long tick labels automatically grow
    #         the figure margins.
    #     autorange
    #         Determines whether or not the range of this axis is
    #         computed in relation to the input data. See `rangemode`
    #         for more info. If `range` is provided, then `autorange`
    #         is set to False.
    #     autoshift
    #         Automatically reposition the axis to avoid overlap with
    #         other axes with the same `overlaying` value. This
    #         repositioning will account for any `shift` amount
    #         applied to other axes on the same side with `autoshift`
    #         is set to true. Only has an effect if `anchor` is set
    #         to "free".
    #     autotypenumbers
    #         Using "strict" a numeric string in trace data is not
    #         converted to a number. Using *convert types* a numeric
    #         string in trace data may be treated as a number during
    #         automatic axis `type` detection. Defaults to
    #         layout.autotypenumbers.
    #     calendar
    #         Sets the calendar system to use for `range` and `tick0`
    #         if this is a date axis. This does not set the calendar
    #         for interpreting data on this axis, that's specified in
    #         the trace or via the global `layout.calendar`
    #     categoryarray
    #         Sets the order in which categories on this axis appear.
    #         Only has an effect if `categoryorder` is set to
    #         "array". Used with `categoryorder`.
    #     categoryarraysrc
    #         Sets the source reference on Chart Studio Cloud for
    #         `categoryarray`.
    #     categoryorder
    #         Specifies the ordering logic for the case of
    #         categorical variables. By default, plotly uses "trace",
    #         which specifies the order that is present in the data
    #         supplied. Set `categoryorder` to *category ascending*
    #         or *category descending* if order should be determined
    #         by the alphanumerical order of the category names. Set
    #         `categoryorder` to "array" to derive the ordering from
    #         the attribute `categoryarray`. If a category is not
    #         found in the `categoryarray` array, the sorting
    #         behavior for that attribute will be identical to the
    #         "trace" mode. The unspecified categories will follow
    #         the categories in `categoryarray`. Set `categoryorder`
    #         to *total ascending* or *total descending* if order
    #         should be determined by the numerical order of the
    #         values. Similarly, the order can be determined by the
    #         min, max, sum, mean or median of all the values.
    #     color
    #         Sets default for all colors associated with this axis
    #         all at once: line, font, tick, and grid colors. Grid
    #         color is lightened by blending this with the plot
    #         background Individual pieces can override this.
    #     constrain
    #         If this axis needs to be compressed (either due to its
    #         own `scaleanchor` and `scaleratio` or those of the
    #         other axis), determines how that happens: by increasing
    #         the "range", or by decreasing the "domain". Default is
    #         "domain" for axes containing image traces, "range"
    #         otherwise.
    #     constraintoward
    #         If this axis needs to be compressed (either due to its
    #         own `scaleanchor` and `scaleratio` or those of the
    #         other axis), determines which direction we push the
    #         originally specified plot area. Options are "left",
    #         "center" (default), and "right" for x axes, and "top",
    #         "middle" (default), and "bottom" for y axes.
    #     dividercolor
    #         Sets the color of the dividers Only has an effect on
    #         "multicategory" axes.
    #     dividerwidth
    #         Sets the width (in px) of the dividers Only has an
    #         effect on "multicategory" axes.
    #     domain
    #         Sets the domain of this axis (in plot fraction).
    #     dtick
    #         Sets the step in-between ticks on this axis. Use with
    #         `tick0`. Must be a positive number, or special strings
    #         available to "log" and "date" axes. If the axis `type`
    #         is "log", then ticks are set every 10^(n*dtick) where n
    #         is the tick number. For example, to set a tick mark at
    #         1, 10, 100, 1000, ... set dtick to 1. To set tick marks
    #         at 1, 100, 10000, ... set dtick to 2. To set tick marks
    #         at 1, 5, 25, 125, 625, 3125, ... set dtick to
    #         log_10(5), or 0.69897000433. "log" has several special
    #         values; "L<f>", where `f` is a positive number, gives
    #         ticks linearly spaced in value (but not position). For
    #         example `tick0` = 0.1, `dtick` = "L0.5" will put ticks
    #         at 0.1, 0.6, 1.1, 1.6 etc. To show powers of 10 plus
    #         small digits between, use "D1" (all digits) or "D2"
    #         (only 2 and 5). `tick0` is ignored for "D1" and "D2".
    #         If the axis `type` is "date", then you must convert the
    #         time to milliseconds. For example, to set the interval
    #         between ticks to one day, set `dtick` to 86400000.0.
    #         "date" also has special values "M<n>" gives ticks
    #         spaced by a number of months. `n` must be a positive
    #         integer. To set ticks on the 15th of every third month,
    #         set `tick0` to "2000-01-15" and `dtick` to "M3". To set
    #         ticks every 4 years, set `dtick` to "M48"
    #     exponentformat
    #         Determines a formatting rule for the tick exponents.
    #         For example, consider the number 1,000,000,000. If
    #         "none", it appears as 1,000,000,000. If "e", 1e+9. If
    #         "E", 1E+9. If "power", 1x10^9 (with 9 in a super
    #         script). If "SI", 1G. If "B", 1B.
    #     fixedrange
    #         Determines whether or not this axis is zoom-able. If
    #         true, then zoom is disabled.
    #     gridcolor
    #         Sets the color of the grid lines.
    #     griddash
    #         Sets the dash style of lines. Set to a dash type string
    #         ("solid", "dot", "dash", "longdash", "dashdot", or
    #         "longdashdot") or a dash length list in px (eg
    #         "5px,10px,2px,2px").
    #     gridwidth
    #         Sets the width (in px) of the grid lines.
    #     hoverformat
    #         Sets the hover text formatting rule using d3 formatting
    #         mini-languages which are very similar to those in
    #         Python. For numbers, see:
    #         https://github.com/d3/d3-format/tree/v1.4.5#d3-format.
    #         And for dates see: https://github.com/d3/d3-time-
    #         format/tree/v2.2.3#locale_format. We add two items to
    #         d3's date formatter: "%h" for half of the year as a
    #         decimal number as well as "%{n}f" for fractional
    #         seconds with n digits. For example, *2016-10-13
    #         09:15:23.456* with tickformat "%H~%M~%S.%2f" would
    #         display "09~15~23.46"
    #     labelalias
    #         Replacement text for specific tick or hover labels. For
    #         example using {US: 'USA', CA: 'Canada'} changes US to
    #         USA and CA to Canada. The labels we would have shown
    #         must match the keys exactly, after adding any
    #         tickprefix or ticksuffix. For negative numbers the
    #         minus sign symbol used (U+2212) is wider than the
    #         regular ascii dash. That means you need to use −1
    #         instead of -1. labelalias can be used with any axis
    #         type, and both keys (if needed) and values (if desired)
    #         can include html-like tags or MathJax.
    #     layer
    #         Sets the layer on which this axis is displayed. If
    #         *above traces*, this axis is displayed above all the
    #         subplot's traces If *below traces*, this axis is
    #         displayed below all the subplot's traces, but above the
    #         grid lines. Useful when used together with scatter-like
    #         traces with `cliponaxis` set to False to show markers
    #         and/or text nodes above this axis.
    #     linecolor
    #         Sets the axis line color.
    #     linewidth
    #         Sets the width (in px) of the axis line.
    #     matches
    #         If set to another axis id (e.g. `x2`, `y`), the range
    #         of this axis will match the range of the corresponding
    #         axis in data-coordinates space. Moreover, matching axes
    #         share auto-range values, category lists and histogram
    #         auto-bins. Note that setting axes simultaneously in
    #         both a `scaleanchor` and a `matches` constraint is
    #         currently forbidden. Moreover, note that matching axes
    #         must have the same `type`.
    #     minexponent
    #         Hide SI prefix for 10^n if |n| is below this number.
    #         This only has an effect when `tickformat` is "SI" or
    #         "B".
    #     minor
    #         :class:`plotly.graph_objects.layout.yaxis.Minor`
    #         instance or dict with compatible properties
    #     mirror
    #         Determines if the axis lines or/and ticks are mirrored
    #         to the opposite side of the plotting area. If True, the
    #         axis lines are mirrored. If "ticks", the axis lines and
    #         ticks are mirrored. If False, mirroring is disable. If
    #         "all", axis lines are mirrored on all shared-axes
    #         subplots. If "allticks", axis lines and ticks are
    #         mirrored on all shared-axes subplots.
    #     nticks
    #         Specifies the maximum number of ticks for the
    #         particular axis. The actual number of ticks will be
    #         chosen automatically to be less than or equal to
    #         `nticks`. Has an effect only if `tickmode` is set to
    #         "auto".
    #     overlaying
    #         If set a same-letter axis id, this axis is overlaid on
    #         top of the corresponding same-letter axis, with traces
    #         and axes visible for both axes. If False, this axis
    #         does not overlay any same-letter axes. In this case,
    #         for axes with overlapping domains only the highest-
    #         numbered axis will be visible.
    #     position
    #         Sets the position of this axis in the plotting space
    #         (in normalized coordinates). Only has an effect if
    #         `anchor` is set to "free".
    #     range
    #         Sets the range of this axis. If the axis `type` is
    #         "log", then you must take the log of your desired range
    #         (e.g. to set the range from 1 to 100, set the range
    #         from 0 to 2). If the axis `type` is "date", it should
    #         be date strings, like date data, though Date objects
    #         and unix milliseconds will be accepted and converted to
    #         strings. If the axis `type` is "category", it should be
    #         numbers, using the scale where each category is
    #         assigned a serial number from zero in the order it
    #         appears.
    #     rangebreaks
    #         A tuple of
    #         :class:`plotly.graph_objects.layout.yaxis.Rangebreak`
    #         instances or dicts with compatible properties
    #     rangebreakdefaults
    #         When used in a template (as
    #         layout.template.layout.yaxis.rangebreakdefaults), sets
    #         the default property values to use for elements of
    #         layout.yaxis.rangebreaks
    #     rangemode
    #         If "normal", the range is computed in relation to the
    #         extrema of the input data. If *tozero*`, the range
    #         extends to 0, regardless of the input data If
    #         "nonnegative", the range is non-negative, regardless of
    #         the input data. Applies only to linear axes.
    #     scaleanchor
    #         If set to another axis id (e.g. `x2`, `y`), the range
    #         of this axis changes together with the range of the
    #         corresponding axis such that the scale of pixels per
    #         unit is in a constant ratio. Both axes are still
    #         zoomable, but when you zoom one, the other will zoom
    #         the same amount, keeping a fixed midpoint. `constrain`
    #         and `constraintoward` determine how we enforce the
    #         constraint. You can chain these, ie `yaxis:
    #         {scaleanchor: *x*}, xaxis2: {scaleanchor: *y*}` but you
    #         can only link axes of the same `type`. The linked axis
    #         can have the opposite letter (to constrain the aspect
    #         ratio) or the same letter (to match scales across
    #         subplots). Loops (`yaxis: {scaleanchor: *x*}, xaxis:
    #         {scaleanchor: *y*}` or longer) are redundant and the
    #         last constraint encountered will be ignored to avoid
    #         possible inconsistent constraints via `scaleratio`.
    #         Note that setting axes simultaneously in both a
    #         `scaleanchor` and a `matches` constraint is currently
    #         forbidden.
    #     scaleratio
    #         If this axis is linked to another by `scaleanchor`,
    #         this determines the pixel to unit scale ratio. For
    #         example, if this value is 10, then every unit on this
    #         axis spans 10 times the number of pixels as a unit on
    #         the linked axis. Use this for example to create an
    #         elevation profile where the vertical scale is
    #         exaggerated a fixed amount with respect to the
    #         horizontal.
    #     separatethousands
    #         If "true", even 4-digit integers are separated
    #     shift
    #         Moves the axis a given number of pixels from where it
    #         would have been otherwise. Accepts both positive and
    #         negative values, which will shift the axis either right
    #         or left, respectively. If `autoshift` is set to true,
    #         then this defaults to a padding of -3 if `side` is set
    #         to "left". and defaults to +3 if `side` is set to
    #         "right". Defaults to 0 if `autoshift` is set to false.
    #         Only has an effect if `anchor` is set to "free".
    #     showdividers
    #         Determines whether or not a dividers are drawn between
    #         the category levels of this axis. Only has an effect on
    #         "multicategory" axes.
    #     showexponent
    #         If "all", all exponents are shown besides their
    #         significands. If "first", only the exponent of the
    #         first tick is shown. If "last", only the exponent of
    #         the last tick is shown. If "none", no exponents appear.
    #     showgrid
    #         Determines whether or not grid lines are drawn. If
    #         True, the grid lines are drawn at every tick mark.
    #     showline
    #         Determines whether or not a line bounding this axis is
    #         drawn.
    #     showspikes
    #         Determines whether or not spikes (aka droplines) are
    #         drawn for this axis. Note: This only takes affect when
    #         hovermode = closest
    #     showticklabels
    #         Determines whether or not the tick labels are drawn.
    #     showtickprefix
    #         If "all", all tick labels are displayed with a prefix.
    #         If "first", only the first tick is displayed with a
    #         prefix. If "last", only the last tick is displayed with
    #         a suffix. If "none", tick prefixes are hidden.
    #     showticksuffix
    #         Same as `showtickprefix` but for tick suffixes.
    #     side
    #         Determines whether a x (y) axis is positioned at the
    #         "bottom" ("left") or "top" ("right") of the plotting
    #         area.
    #     spikecolor
    #         Sets the spike color. If undefined, will use the series
    #         color
    #     spikedash
    #         Sets the dash style of lines. Set to a dash type string
    #         ("solid", "dot", "dash", "longdash", "dashdot", or
    #         "longdashdot") or a dash length list in px (eg
    #         "5px,10px,2px,2px").
    #     spikemode
    #         Determines the drawing mode for the spike line If
    #         "toaxis", the line is drawn from the data point to the
    #         axis the  series is plotted on. If "across", the line
    #         is drawn across the entire plot area, and supercedes
    #         "toaxis". If "marker", then a marker dot is drawn on
    #         the axis the series is plotted on
    #     spikesnap
    #         Determines whether spikelines are stuck to the cursor
    #         or to the closest datapoints.
    #     spikethickness
    #         Sets the width (in px) of the zero line.
    #     tick0
    #         Sets the placement of the first tick on this axis. Use
    #         with `dtick`. If the axis `type` is "log", then you
    #         must take the log of your starting tick (e.g. to set
    #         the starting tick to 100, set the `tick0` to 2) except
    #         when `dtick`=*L<f>* (see `dtick` for more info). If the
    #         axis `type` is "date", it should be a date string, like
    #         date data. If the axis `type` is "category", it should
    #         be a number, using the scale where each category is
    #         assigned a serial number from zero in the order it
    #         appears.
    #     tickangle
    #         Sets the angle of the tick labels with respect to the
    #         horizontal. For example, a `tickangle` of -90 draws the
    #         tick labels vertically.
    #     tickcolor
    #         Sets the tick color.
    #     tickfont
    #         Sets the tick font.
    #     tickformat
    #         Sets the tick label formatting rule using d3 formatting
    #         mini-languages which are very similar to those in
    #         Python. For numbers, see:
    #         https://github.com/d3/d3-format/tree/v1.4.5#d3-format.
    #         And for dates see: https://github.com/d3/d3-time-
    #         format/tree/v2.2.3#locale_format. We add two items to
    #         d3's date formatter: "%h" for half of the year as a
    #         decimal number as well as "%{n}f" for fractional
    #         seconds with n digits. For example, *2016-10-13
    #         09:15:23.456* with tickformat "%H~%M~%S.%2f" would
    #         display "09~15~23.46"
    #     tickformatstops
    #         A tuple of :class:`plotly.graph_objects.layout.yaxis.Ti
    #         ckformatstop` instances or dicts with compatible
    #         properties
    #     tickformatstopdefaults
    #         When used in a template (as
    #         layout.template.layout.yaxis.tickformatstopdefaults),
    #         sets the default property values to use for elements of
    #         layout.yaxis.tickformatstops
    #     ticklabelmode
    #         Determines where tick labels are drawn with respect to
    #         their corresponding ticks and grid lines. Only has an
    #         effect for axes of `type` "date" When set to "period",
    #         tick labels are drawn in the middle of the period
    #         between ticks.
    #     ticklabeloverflow
    #         Determines how we handle tick labels that would
    #         overflow either the graph div or the domain of the
    #         axis. The default value for inside tick labels is *hide
    #         past domain*. Otherwise on "category" and
    #         "multicategory" axes the default is "allow". In other
    #         cases the default is *hide past div*.
    #     ticklabelposition
    #         Determines where tick labels are drawn with respect to
    #         the axis Please note that top or bottom has no effect
    #         on x axes or when `ticklabelmode` is set to "period".
    #         Similarly left or right has no effect on y axes or when
    #         `ticklabelmode` is set to "period". Has no effect on
    #         "multicategory" axes or when `tickson` is set to
    #         "boundaries". When used on axes linked by `matches` or
    #         `scaleanchor`, no extra padding for inside labels would
    #         be added by autorange, so that the scales could match.
    #     ticklabelstep
    #         Sets the spacing between tick labels as compared to the
    #         spacing between ticks. A value of 1 (default) means
    #         each tick gets a label. A value of 2 means shows every
    #         2nd label. A larger value n means only every nth tick
    #         is labeled. `tick0` determines which labels are shown.
    #         Not implemented for axes with `type` "log" or
    #         "multicategory", or when `tickmode` is "array".
    #     ticklen
    #         Sets the tick length (in px).
    #     tickmode
    #         Sets the tick mode for this axis. If "auto", the number
    #         of ticks is set via `nticks`. If "linear", the
    #         placement of the ticks is determined by a starting
    #         position `tick0` and a tick step `dtick` ("linear" is
    #         the default value if `tick0` and `dtick` are provided).
    #         If "array", the placement of the ticks is set via
    #         `tickvals` and the tick text is `ticktext`. ("array" is
    #         the default value if `tickvals` is provided). If
    #         "sync", the number of ticks will sync with the
    #         overlayed axis set by `overlaying` property.
    #     tickprefix
    #         Sets a tick label prefix.
    #     ticks
    #         Determines whether ticks are drawn or not. If "", this
    #         axis' ticks are not drawn. If "outside" ("inside"),
    #         this axis' are drawn outside (inside) the axis lines.
    #     tickson
    #         Determines where ticks and grid lines are drawn with
    #         respect to their corresponding tick labels. Only has an
    #         effect for axes of `type` "category" or
    #         "multicategory". When set to "boundaries", ticks and
    #         grid lines are drawn half a category to the left/bottom
    #         of labels.
    #     ticksuffix
    #         Sets a tick label suffix.
    #     ticktext
    #         Sets the text displayed at the ticks position via
    #         `tickvals`. Only has an effect if `tickmode` is set to
    #         "array". Used with `tickvals`.
    #     ticktextsrc
    #         Sets the source reference on Chart Studio Cloud for
    #         `ticktext`.
    #     tickvals
    #         Sets the values at which ticks on this axis appear.
    #         Only has an effect if `tickmode` is set to "array".
    #         Used with `ticktext`.
    #     tickvalssrc
    #         Sets the source reference on Chart Studio Cloud for
    #         `tickvals`.
    #     tickwidth
    #         Sets the tick width (in px).
    #     title
    #         :class:`plotly.graph_objects.layout.yaxis.Title`
    #         instance or dict with compatible properties
    #     titlefont
    #         Deprecated: Please use layout.yaxis.title.font instead.
    #         Sets this axis' title font. Note that the title's font
    #         used to be customized by the now deprecated `titlefont`
    #         attribute.
    #     type
    #         Sets the axis type. By default, plotly attempts to
    #         determined the axis type by looking into the data of
    #         the traces that referenced the axis in question.
    #     uirevision
    #         Controls persistence of user-driven changes in axis
    #         `range`, `autorange`, and `title` if in `editable:
    #         true` configuration. Defaults to `layout.uirevision`.
    #     visible
    #         A single toggle to hide the axis while preserving
    #         interaction like dragging. Default is true when a
    #         cheater plot is present on the axis, otherwise false
    #     zeroline
    #         Determines whether or not a line is drawn at along the
    #         0 value of this axis. If True, the zero line is drawn
    #         on top of the grid lines.
    #     zerolinecolor
    #         Sets the line color of the zero line.
    #     zerolinewidth
    #         Sets the width (in px) of the zero line.
        


        # Update xaxis properties
        # fig.update_xaxes(title_text="xaxis 1 title", row=10, col=1)
        # fig.update_xaxes(title_text="xaxis 2 title", range=[10, 50], row=1, col=2)
        # fig.update_xaxes(title_text="xaxis 3 title", showgrid=False, row=2, col=1)
        # fig.update_xaxes(title_text="xaxis 4 title", type="log", row=2, col=2)

        # Update yaxis properties
        self.fig = fig;

        self.fig.update_xaxes(showticklabels=False) 
        self.fig.update_yaxes(showticklabels=False)

        tickfont = {'size':10}; 
        tickfont2 = {'size':10}

        titlefont = {'size':12};
        titlefont2 = {'size':14}




        
            #     "y": y_pos,
            #     "xref": "paper",
            #     "x": x_pos,
            #     "yref": "paper",
            #     "text": title,
            #     "showarrow": False,
            #     "font": dict(size=16),
            #     "xanchor": "center",
            #     "yanchor": "bottom",
            # } for x_pos,y_pos,title in subplot_titles]


# annotations = [{
#                 "y": y_pos,
#                 "xref": "paper",
#                 "x": x_pos,
#                 "yref": "paper",
#                 "text": title,
#                 "showarrow": False,
#                 "font": dict(size=16),
#                 "xanchor": "center",
#                 "yanchor": "bottom",
#             } for x_pos,y_pos,title in subplot_titles]

        # if True: 
        #     self.fig.update_title(title={'text':"Ondemand" ,'font':titlefont2,'standoff':0},row=9,col=1); 
        #     self.fig.update_title(title={'text':"DEMAND" ,'font':titlefont2,'standoff':0},row=9,col=1); 
        #     self.fig.update_title(title={'text':"ONDEMAND" ,'font':titlefont2,'standoff':0},row=9,col=1); 
        #     self.fig.update_title(title={'text':"ONDEMAND" ,'font':titlefont2,'standoff':0},row=9,col=1); 
            # self.fig.update_xaxes(title={'text':"groups" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=3, tickfont=tickfont2, side='bottom'); #,ticks='inside')
            # self.fig.update_xaxes(title={'text':"trip segments" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=7, tickfont=tickfont2, side='bottom'); #,ticks='inside')
            # self.fig.update_xaxes(title={'text':"" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=7, tickfont=tickfont2, side='bottom'); #,ticks='inside')



        if True: 
            ### yaxes 
            self.fig.update_yaxes(title={'text':"dist" ,'font':titlefont}, row=9 , col=1,showticklabels=True, tickfont=tickfont,side='left'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"time" ,'font':titlefont}, row=10, col=1, showticklabels=True, tickfont=tickfont,side='left'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"money",'font':titlefont}, row=11, col=1, showticklabels=True, tickfont=tickfont,side='left'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"switch",'font':titlefont}, row=12, col=1, showticklabels=True, tickfont=tickfont,side='left');#,ticks='inside')

            self.fig.update_yaxes(title={'text':""}, row=9 , col=3,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':""}, row=10, col=3, showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':""}, row=11, col=3, showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':""}, row=12, col=3, showticklabels=True, tickfont=tickfont,side='right');#,ticks='inside')


            self.fig.update_yaxes(title={'text':"dist" ,'font':titlefont}, row=9 , col=7,showticklabels=True, tickfont=tickfont,side='left'); #,ticks='inside')        
            self.fig.update_yaxes(title={'text':"time" ,'font':titlefont}, row=10, col=7, showticklabels=True, tickfont=tickfont,side='left'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"money",'font':titlefont}, row=11, col=7, showticklabels=True, tickfont=tickfont,side='left'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"switch",'font':titlefont}, row=12, col=7, showticklabels=True, tickfont=tickfont,side='left');#,ticks='inside')



            self.fig.update_yaxes(title={'text':"dist" ,'font':titlefont}, row=1 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"time" ,'font':titlefont}, row=2 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"money" ,'font':titlefont}, row=3 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"switch" ,'font':titlefont}, row=4 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')        


            self.fig.update_yaxes(title={'text':"dist" ,'font':titlefont}, row=5 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"time" ,'font':titlefont}, row=6 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"money" ,'font':titlefont}, row=7 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            self.fig.update_yaxes(title={'text':"switch" ,'font':titlefont}, row=8 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')        


        # annotations = [dict(text='Ondemand',y=0,x=0,yref='y9',xref='x1',font=dict(size=16),xanchor='center',yanchor='top',showarrow=False)]
                       # dict(text='Ondemand',y=0,x=0,yref='y9',xref='x1',font=dict(size=16),showarrow-False)]
        # self.fig.update_layout(annotations=annotations)                       
        # if True: 
        #     self.fig.update_xaxes(title={'text':"trip segments" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=1, tickfont=tickfont2, side='bottom'); #,ticks='inside')
        #     self.fig.update_xaxes(title={'text':"groups" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=3, tickfont=tickfont2, side='bottom'); #,ticks='inside')
        #     self.fig.update_xaxes(title={'text':"trip segments" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=7, tickfont=tickfont2, side='bottom'); #,ticks='inside')
        #     self.fig.update_xaxes(title={'text':"" ,'font':titlefont2,'standoff':50}, showticklabels=True, row=12 , col=7, tickfont=tickfont2, side='bottom'); #,ticks='inside')

            # self.fig.update_xaxes(title={'text':"trip segments" ,'font':titlefont}, row=12 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            # self.fig.update_xaxes(title={'text':"trip segments" ,'font':titlefont}, row=12 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')
            # self.fig.update_xaxes(title={'text':"trip segments" ,'font':titlefont}, row=12 , col=9,showticklabels=True, tickfont=tickfont,side='right'); #,ticks='inside')



        # fig.update_yaxes(anchor='free',ticks='inside',title={'text':"dist" ,'font':{'size':10}}, row=1 , col=13)
        # fig.update_yaxes(title={'text':"time" ,'font':{'size':10}}, row=10, col=1)
        # fig.update_yaxes(title={'text':"money",'font':{'size':10}}, row=11, col=1)
        # fig.update_yaxes(title={'text':"switch",'font':{'size':10}}, row=12, col=1)


        # fig.update_yaxes(title={'text':"dist" ,'font':{'size':10}}, row=9 , col=7)
        # fig.update_yaxes(title={'text':"time" ,'font':{'size':10}}, row=10, col=7)
        # fig.update_yaxes(title={'text':"money",'font':{'size':10}}, row=11, col=7)


        # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
        # fig.update_yaxes(title_text="yaxis 3 title", showgrid=False, row=2, col=1)
        # fig.update_yaxes(title_text="yaxis 4 title", row=2, col=2)



        # for plt in self.plts:
        #     PLT = self.plts[plt]
        #     inds = PLT['inds']
        #     fig.add_trace(go.Scatter(x=[300, 400, 500], y=[600, 700, 800]),row=inds[0],col=inds[1]);


        self.TRACES = {};
        self.addSliders();
        self.addButtons();

        self.addTraces();
        self.connectControls();

        
        
        self.fig.update_layout(width=self.width,height=self.height,showlegend=False); #dims[0],height=dims[1],boxmode='group'); #, xaxis_visible=False, yaxis_visible=False)
        temp = [SLIDER.dict for SLIDER in self.SLIDERS.values()]
        self.fig.update_layout(sliders=temp)
        temp = [BUTTON.dict for BUTTON in self.BUTTONS.values()]
        self.fig.update_layout(updatemenus=temp)


        

        self.fig.update_layout(barmode='overlay'); #layout = go.Layout(barmode='overlay')); #'stack'))
        # fig = dict(data = data, layout = layout)
        # iplot(fig, show_link=False)




            # self.list_of_sliders);
        # fig.update_layout(updatemenus=BUTTONS)
        # fig.update_layout(sliders=SLIDERS)
# width = np.sum(column_widths); height = 1.2*np.sum(row_heights); #+padb1*3)
# fig.update_layout(width=width, height=height, xaxis_visible=False, yaxis_visible=False)

        fig.show()

    def addImages(self,layers ):
        self.image_names = [];
        for layer in layers:
            LAYER = layers[layer];
            self.image_names.append(LAYER['name']);

    def addTraces(self):
        DFs = self.dataDFs;
        factors = ['dist','time','money','switches']
        factors2 = ['distance','travel_time','money','switches']
        show_boxes = False; 

        IMGPLT = self.plts['img1'];


        img_names = ['base','lines','source','target','drive','walk','ondemand','gtfs']
        for name in img_names:
            filename =  self.folder + name + '.png'
            if os.path.isfile(filename): 
                params = {}; 
                params['name'] = name
                params['filename'] = filename;
                params['loc'] = IMGPLT['inds'];
                # blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                if not(name == 'base'):
                    params['buttons'] = [name]; #{mode: blarg};
                params['typ'] = 'image'
                params['init_visible'] = True;
                self.TRACES[name] = TRACE(self.fig,params)                                         
                self.TRACES[name].add();




        ####### 
        for mode in DFs:

            if mode == 'ondemand':
                sortby = 'sorted_by_group_and_travel_time';
                sortbygrp = 'sorted_by_travel_time_order_in_group'
            else:
                sortby = 'sorted_by_travel_time';
                sortbygrp = 'sorted_by_travel_time_order_in_group'

            DF = DFs[mode];
            for i,factor in enumerate(factors):
                
                params = {};
                plt = mode + '_' + factor + '_bar'
                name = mode + '_bar_' + 'all'; params['name'] = name
                PLT = self.plts[plt]
                factor2 = factors2[i]
                xx = list(DF[sortby]); yy = list(DF[factor2])
                xx = np.array(xx); yy = np.array(yy)
                temp = np.argsort(xx);
                xx = xx[temp]; yy = yy[temp]
                params['loc'] = PLT['inds'];
                params['x'] = xx; params['y'] = yy
                params['color'] = self.colors[mode]; params['opacity'] = 0.3;
                # params['sliders'] = {mode:blarg};
                params['typ'] = 'bar'
                params['init_visible'] = True;
                self.TRACES[name] = TRACE(self.fig,params)
                self.TRACES[name].add();

                plt = mode + '_' + factor + '_bar'
                PLT = self.plts[plt]

                # if show_boxes:
                #     plt = mode + '_' + factor + '_box'
                #     name = mode + '_box_' + 'all'; params['name'] = name
                #     PLT = self.plts[plt]
                #     name = mode + '_box_' + 'all'; params['name'] = name
                #     params['typ'] = 'box'
                #     params['loc'] = PLT['inds'];
                #     self.TRACES[name] = TRACE(self.fig,params)


                ########################################################
                if mode in ['ondemand']:#,'gtfs']:

                    datainds_temp = {};
                    datainds_img = {'groups':{},'runs':{}};

                    if mode == 'ondemand': grp_tag = 'group_id'; run_tag = 'run_id'
                    if mode == 'gtfs': grp_tag = 'line_id'; run_tag = 'bus_trip_id';

                    plt1 = mode + '_' + factor + '_bar'
                    plt2 = mode + '_' + factor + '_bar_grp'
                    PLT1 = self.plts[plt1]
                    PLT2 = self.plts[plt2]
                    factor2 = factors2[i]


                    tag = 'sorted_by_' + factor2 + '_order_in_group';

                    group_ids = DF[grp_tag].unique()
                    # print('group_ids: ', group_ids)
                    for j,group_id in enumerate(group_ids):
                        datainds_temp[group_id] = {};
                        datainds_img['groups'][group_id] = {};
                        datainds_img['runs'][group_id] = {};

                        DF2 = DF[DF[grp_tag] == group_id]
                        xx2 = np.array(list(DF2[sortby]));

                        xx3 = np.array(list(DF2[sortbygrp]));
                        yy2 = np.array(list(DF2[factor2]))
                        # print(group_id)
                        # print(len(yy2))

                        params = {};
                        if not(isinstance(group_id,str)): group_id = str(group_id)
                        params = {};
                        name = mode + '_' + factor + '_bar_all_grp_' + group_id;
                        params['name'] = name
                        params['loc'] = PLT['inds'];
                        params['x'] = xx2; params['y'] = yy2
                        params['color'] = self.colors[mode]; #'rgb(0,0,0)'; params['opacity'] = 0.5;
                        params['opacity'] = 0.2;
                        blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                        params['sliders'] = {mode+'_grp':blarg};
                        params['typ'] = 'bar'
                        params['init_visible'] = False;
                        if j==0:  params['init_visible'] = True;
                        else:  params['init_visible'] = False;
                        self.TRACES[name] = TRACE(self.fig,params)
                        self.TRACES[name].add();

                        name = mode + '_' + factor + '_bar_grp_' + group_id;
                        params['name'] = name
                        params['loc'] = PLT2['inds'];
                        params['x'] = xx3; #np.array(list(range(len(xx2))))+1;
                        params['y'] = yy2
                        params['color'] = self.colors[mode]; #'rgb(0,0,0)';
                        params['opacity'] = 0.3;
                        blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                        params['sliders'] = {mode+'_grp':blarg};
                        params['typ'] = 'bar'
                        if j==0:  params['init_visible'] = True;
                        else:  params['init_visible'] = False;
                        self.TRACES[name] = TRACE(self.fig,params)
                        self.TRACES[name].add();

                        if factor == 'time':
                            name = 'img_' + mode + '_' + group_id;
                            filename =  self.folder + 'groups/' + group_id + '.png';
                            if os.path.isfile(filename) and not(name in self.TRACES):
                                params = {}; 
                                params['name'] = name
                                params['filename'] = filename;
                                params['loc'] = IMGPLT['inds'];
                                params['typ'] = 'image'
                                blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                                params['sliders'] = {mode+'_grp':blarg};
                                if j==0:  params['init_visible'] = True;
                                else:  params['init_visible'] = False;
                                self.TRACES[name] = TRACE(self.fig,params)
                                dataind = self.TRACES[name].add();
                                datainds_img['groups'][group_id] = dataind


                        run_ids = DF[DF[grp_tag]==group_id]['run_id'].unique();
                        # print('run_ids:', run_ids)
                        for k,run_id in enumerate(run_ids):

                            # print(run_id)
                            mask1 = DF[grp_tag] == group_id
                            mask2 = DF[run_tag] == run_id
                            DF3 = DF[mask1 & mask2];
                            # DF3 = DF2[DF2[run_tag] == run_id]
                            xx4 = np.array(list(DF3[sortby]));
                            yy4 = np.array(list(DF3[factor2]))
                            # print(xx4)

                            name = mode + '_' + factor + '_bar_grp_' + group_id + '_run' + str(int(run_id));
                            params = {};
                            params['name'] = name
                            params['loc'] = PLT['inds'];
                            # if len(xx3)>0:
                            #     params['x'] = np.array(xx3[0:]); #np.array(list(range(len(xx2))))+1;
                            #     params['y'] = np.array(yy3[0:1]);
                            # else:
                            params['x'] = np.array(xx4); #np.array(list(range(len(xx2))))+1;
                            params['y'] = np.array(yy4);

                            # if factor == 'time':
                            #     print(params['x'])
                            #     print(params['y'])

                            # print(params['x'])
                            # print(params['y'])
                            # print(xx3)
                            # print(yy3)

                            params['color'] = self.colors[mode]; #'rgb(0,0,0)';
                            params['opacity'] = 0.8;
                            # print(datainds_temp)
                            # blarg = {'turnoninds':[datainds_temp[group_id][run_id]]}; #int(10*np.random.rand(1))}
                            # # params['sliders'] = {mode+'_grp':blarg};
                            params['typ'] = 'bar'
                            # if j==0:  params['init_visible'] = True;
                            params['init_visible'] = False;
                            self.TRACES[name] = TRACE(self.fig,params)
                            dataind = self.TRACES[name].add();
                            datainds_temp[group_id][run_id] = dataind;
                            
                            if factor == 'time':
                                name = 'img_' + mode + '_' + group_id + '_run' + str(int(run_id));
                                # img_name = 'img_' + mode + '_' + group_id + '_run' + str(int(run_id));
                                filename =  self.folder + 'runs/' + group_id + '_run' + str(int(run_id)) + '.png';
                                if os.path.isfile(filename) and not(name in self.TRACES):
                                    params = {}; 
                                    params['name'] = name
                                    params['filename'] = filename;
                                    params['loc'] = IMGPLT['inds'];
                                    params['typ'] = 'image'
                                    if j==0:  params['init_visible'] = True;
                                    else:  params['init_visible'] = False;
                                    self.TRACES[name] = TRACE(self.fig,params)
                                    dataind = self.TRACES[name].add();
                                    datainds_img['runs'][group_id][run_id] = dataind                         



        # for mode in DFs:

        #     if mode == 'ondemand':
        #         sortby = 'sorted_by_group_and_travel_time';
        #         sortbygrp = 'sorted_by_travel_time_order_in_group'
        #     else:
        #         sortby = 'sorted_by_travel_time';
        #         sortbygrp = 'sorted_by_travel_time_order_in_group'

        #     DF = DFs[mode];
        #     for i,factor in enumerate(factors):


                if True: #factor == 'time':
                    for j,x in enumerate(xx):#[:30]):

                        blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                        params = {};                    
                        name = mode + '_'+factor+'_bar_' + str(x); params['name'] = name
                        params['loc'] = PLT['inds'];
                        params['x'] = [x]; params['y'] = [yy[j]]
                        params['color'] = self.colors[mode]
                        params['color'] = 'rgb(0,0,0)';
                        params['opacity'] = 0.8;
                        params['sliders'] = {mode:blarg};
                        params['typ'] = 'bar'
                        if j==0:  params['init_visible'] = True;
                        else:  params['init_visible'] = False;
                        self.TRACES[name] = TRACE(self.fig,params)
                        self.TRACES[name].add();

                        if factor == 'time':
                            start_node = DF.iloc[j]['start_node']
                            end_node = DF.iloc[j]['end_node']
                            # group_id = DF.iloc[j]['group_id']
                            # run_id = DF.iloc[j]['run_id']

                            if isinstance(start_node,float): start_node = int(start_node);
                            if isinstance(end_node,float): end_node = int(end_node);
                            if isinstance(start_node,int): start_node = str(start_node)
                            if isinstance(end_node,int): end_node = str(end_node)
                            name = 'img_' + mode + '_' + start_node + '_' + end_node
                            filename =  self.folder + mode + '_trips/';
                            filename = filename + mode + '_' + start_node + '_' + end_node + '.png'
                            if os.path.isfile(filename) and not(name in self.TRACES):
                                params = {}; 
                                params['name'] = name
                                params['filename'] = filename;
                                params['loc'] = IMGPLT['inds'];
                                blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                                params['sliders'] = {mode: blarg};
                                params['typ'] = 'image'
                                if j==0:  params['init_visible'] = True;
                                else:  params['init_visible'] = False;
                                self.TRACES[name] = TRACE(self.fig,params)                                         
                                self.TRACES[name].add();

                            # name = 'img_' + mode + '_' + start_node + '_' + end_node
                            # filename =  self.folder + mode + '_trips/';
                            # filename = filename + mode + '_' + start_node + '_' + end_node + '.png'
                            # if os.path.isfile(filename): 
                            #     params = {}; 
                            #     params['name'] = name
                            #     params['filename'] = filename;
                            #     params['loc'] = IMGPLT['inds'];
                            #     blarg = {'turnoninds':[j]}; #int(10*np.random.rand(1))}
                            #     params['sliders'] = {mode: blarg};
                            #     params['typ'] = 'image'
                            #     if j==0:  params['init_visible'] = True;
                            #     else:  params['init_visible'] = False;

                            #     self.TRACES[name] = TRACE(self.fig,params)                                         
                            #     self.TRACES[name].add();






                # print(list(DF[sortby]))
                DF4 = DF.sort_values(by=[sortby])
                if mode == 'ondemand':
                    for j in range(len(DF4)):#enumerate(xx): #range(len(DF)):#,x in enumerate(xx):#[:30]):
                        # print('x:',x)
                        # print('j:',j)
                        row = DF4.iloc[j]
# run_ids = DF[DF[grp_tag]==group_id]['run_id'].unique();

                        group_id = row['group_id']
                        run_id = row['run_id'];
                        # print(group_id)
                        # print(run_id)

                        # print(datainds_temp)
                        # print(datainds_img)

                        # print(group_id)
                        # print(run_id)                        
                        # dataind = datainds_temp[group_id][run_id]
                        trace_name = mode + '_' + factor + '_bar_grp_' + group_id + '_run' + str(int(run_id));
                        
                        turnonind = j;
                        # print(trace_name)

                        # print(turnonind)
                        # print(dataind)
                        # print('')
                        # print(name)
                        # self.TRACES[name].turnoninds = self.TRACES[name].turnoninds + [j]
                        # print(self.TRACES[name].sliders); # = self.TRACES[name].turnoninds + [j]); #turnonaddTrace(self,[j],[dataind])
                        slider = mode;
                        # print(self.TRACES[trace_name].sliders)
                        if trace_name in self.TRACES:
                            if not(slider in self.TRACES[trace_name].sliders):
                                self.TRACES[trace_name].sliders[slider] = {'turnoninds':[turnonind]}
                            else:
                                self.TRACES[trace_name].sliders[slider]['turnoninds'].append(turnonind)


                        if factor == 'time':
                            img_name = 'img_' + mode + '_' + group_id + '_run' + str(int(run_id));
                            # print(list(self.TRACES))
                            if img_name in self.TRACES:
                                if not(slider in self.TRACES[img_name].sliders):
                                    self.TRACES[img_name].sliders[slider] = {'turnoninds':[turnonind]}
                                else:
                                    self.TRACES[img_name].sliders[slider]['turnoninds'].append(turnonind)

                    # for j,x in enumerate(xx):#[:30]):
                    #     row = DF.iloc[j]
                    #     # run_ids = DF[DF[grp_tag]==group_id]['run_id'].unique();

                    #     group_id = row['group_id']
                    #     run_id = row['run_id'];
                    #     # print(group_id)
                    #     # print(run_id)
                    #     trace_name = mode + '_' + factor + '_bar_grp_' + group_id + '_run_' + str(int(run_id));

                        # print('place:',j)
                        # print('trace:',trace_name);
                        # print(self.TRACES[trace_name].sliders[slider]['turnoninds'])


                        # print(self.TRACES[trace_name].sliders[slider]['turnoninds'])

                        # print('turnoninds...')
                        # print(self.SLIDERS[slider].turnoninds)
                        # print(self.SLIDERS[slider].datainds)
                        # self.addButtons();                            
                        # if not(slider in self.TRACES[trace_name].sliders):
                        #     self.TRACES[trace_name].sliders[slider] = {'turnoninds':[]}
                        # # print(self.TRACES[trace_name].sliders[slider])
                        # self.TRACES[trace_name].sliders[slider]['turnoninds'].append(j)


    # def addTrace(self,turnoninds,dataind):
    #     maxturnonind = int(np.max(turnoninds))
    #     if self.num_steps < maxturnonind + 1:
    #         new_num_steps = maxturnonind + 1;
    #         diff_size = new_num_steps - self.num_steps;
    #         self.turnoninds = self.turnoninds + [[] for _ in range(diff_size)];
    #         self.num_steps = new_num_steps
    #     for turnonind in turnoninds:
    #         self.turnoninds[turnonind].append(len(self.datainds));
    #         self.datainds.append(dataind)    


                            # print(self.TRACES[name].sliders[slider]['turnoninds'])


                        # if True: mode == 'drive' or mode == 'walk' or mode == 'ondemand':

                    # if show_boxes:
                    #     plt = mode + '_' + factor + '_box_grp'
                    #     PLT = self.plts[plt]
                    #     name = mode + '_box_' + 'all'; params['name'] = name
                    #     params['typ'] = 'box'
                    #     params['loc'] = PLT['inds']
                    #     self.TRACES[name] = TRACE(self.fig,params)

                    # if False:
                    #     plt = mode + '_' + factor + '_bar_run'
                    #     PLT = self.plts[plt]
                    #     factor2 = factors2[i]
                    #     xx = list(DF[sortby]); yy = list(DF[factor2])
                    #     params = {};
                    #     name = mode + '_bar_' + 'all'; params['name'] = name
                    #     params['loc'] = PLT['inds'];
                    #     params['x'] = xx; params['y'] = yy
                    #     params['color'] = 'rgb(0,0,0)'; params['opacity'] = 0.5;
                    #     params['sliders'] = {mode:blarg};
                    #     params['typ'] = 'bar'
                    #     self.TRACES[name] = TRACE(self.fig,params)
                    #     self.TRACES[name].add();

                    # if show_boxes:
                    #     plt = mode + '_' + factor + '_box_run'
                    #     PLT = self.plts[plt]
                    #     name = mode + '_box_' + 'all'; params['name'] = name
                    #     params['loc'] = PLT['inds'];
                    #     params['typ'] = 'box'
                    #     self.TRACES[name] = TRACE(self.fig,params)

    def connectControls(self):
        for trace in self.TRACES:
            TRACE = self.TRACES[trace];
            for slider in TRACE.sliders:
                slider_data = TRACE.sliders[slider];
                # print('slider':)
                # print(slider_data)
                dataind = TRACE.dataind;
                turnoninds = slider_data['turnoninds'];
                self.SLIDERS[slider].addTrace(turnoninds,dataind);

            for button in TRACE.buttons:
                # button_data = TRACE.buttons[button];
                dataind = TRACE.dataind;
                self.BUTTONS[button].addTrace(dataind);


        # for image in self.IMAGES:
        #     IMAGE = self.IMAGES[image];
        #     for slider in TRACE.sliders:
        #         slider_data = TRACE.sliders[slider];
        #         dataind = TRACE.dataind;
        #         turnoninds = slider_data['turnoninds'];
        #         self.SLIDERS[slider].addDataInd(turnoninds,dataind)                
        for slider in self.SLIDERS:
            SLIDER = self.SLIDERS[slider];
            SLIDER.addSteps();
            # self.list_of_sliders.append(SLIDER.dict);            
        for button in self.BUTTONS:
            BUTTON = self.BUTTONS[button];
            BUTTON.add();
            # print(BUTTON.dict)

        # for name in self.trace_names:            
        #     self.TRACES[name]
        # self.trace_names



    def addSliders(self):
        self.SLIDERS = {}
        self.list_of_sliders = [];
        
        for i,name in enumerate(self.slider_names):

            params = {};
            grid_len = self.slider_grid_lengths[i];
            grid_loc = self.slider_grid_locs[i]
            hpad = self.slider_hpads[i]
            vpad = self.slider_vpads[i]

            gloc0 = int(np.floor(grid_loc[0])); gloc0d = grid_loc[0]-gloc0;
            gloc1 = int(np.floor(grid_loc[1])); gloc1d = grid_loc[1]-gloc1;

            loc0 = np.sum(self.row_heights[:gloc0]) + self.row_heights[gloc0]*gloc0d;
            loc1 = np.sum(self.column_widths[:gloc1]) + self.column_widths[gloc1]*gloc1d;

            glen1 = int(np.floor(grid_len)); glen1d = grid_len - glen1;
            slider_len = np.sum(self.column_widths[gloc1:gloc1+glen1]) + self.column_widths[glen1]*glen1d


            col_wid = self.column_widths[gloc1]*self.width;
            row_hgt = self.column_widths[gloc0]*self.width;
            # print(loc0)
            # print()

            params['loc'] = [loc1,1.-float(loc0)]
            params['length'] = slider_len
            params['pad'] = {'t':vpad[0]*row_hgt,'b':vpad[1]*row_hgt,'l':hpad[0]*col_wid,'r':hpad[1]*col_wid};
            params['name'] = name;
            params['active'] = True;
            self.SLIDERS[name] = SLIDER(params)

    def addButtons(self):
        self.BUTTONS = {}
        self.list_of_buttons = [];
        for i,name in enumerate(self.button_names):
            params = {};
            # length = self.button_lengths[i];
            loc = self.button_locs[i];
            # hpad = self.button_hpads[i]
            # print( self.button_display_names[i])

            params['loc'] = loc; 
            # params['length'] = length;
            # params['pad'] = {'t':0,'b':0,'l':hpad[0],'r':hpad[1]};
            params['name'] = name;
            params['active'] = True;
            params['display_name'] = self.button_display_names[i];

            self.BUTTONS[name] = BUTTON(params)

    def addOutputs(self,OUTPUTS,use_active=False,filter_for_zeros=False):
        # self.OUTPUTS = OUTPUTS;
        factors = ['distance','travel_time','money','switches']
        modes = ['drive','walk','ondemand','gtfs'];
        DFs = {};
        main_cols = ['distance','travel_time','money','switches','start_node','end_node']

        for mode in modes:
            if mode in ['drive','walk']: cols = main_cols
            if mode in ['ondemand']: cols = main_cols + ['group_id','run_id']
            if mode in ['gtfs']: cols = main_cols + ['line_id','bus_trip_id']
            df = OUTPUTS['by_mode'][mode]
            
            if use_active:
                df = df[df['active']==True];
            if mode == 'ondemand' and filter_for_zeros:
                df = df[~df['travel_time'].isna()];
            df = df[cols]
            for factor in factors:
                df = df.sort_values(by=[factor])
                df['sorted_by_' + factor] = list(range(len(df)+1))[1:]
                if mode == 'ondemand':
                    df = df.sort_values(by=['group_id',factor])
                    df['sorted_by_group_and_'+factor] = list(range(len(df)+1))[1:];

                    df['sorted_by_'+factor+'_order_in_group'] = np.nan;
                    group_ids = df['group_id'].unique()
                    for group in group_ids:
                        nms = np.array(list(range(np.sum(df['group_id'] == group)))).astype(int);
                        df.loc[df['group_id'] == group,'sorted_by_'+factor+'_order_in_group'] = nms
            DFs[mode] = df.copy();
        self.dataDFs = DFs



    # def addButtons(self):
    #     list_of_buttons = [];
    #     for i,name in enumerate(self.button_names):
    #         params = {};
    #         params['loc'] = self.button_locs[i];
    #         params['name'] = name; 
    #         self.BUTTONS[name] = BUTTON(params)
    #         list_of_buttons.append(self.BUTTONS[name].dict);
    #     self.fig.update_layout(updatemenus=list_of_buttons)
        # fig.update_layout(sliders=SLIDERS)
        # width = np.sum(column_widths); height = 1.2*np.sum(row_heights); #+padb1*3)
        # fig.update_layout(width=width, height=height, xaxis_visible=False, yaxis_visible=False)





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






class TRACE:
    def __init__(self,fig,params = {}):

        self.fig = fig;
        self.image = None;
        self.name = None
        self.x = None
        self.y = None
        self.sliders = {};#'':{},
        self.buttons = [];#'':{},
        self.loc = None;
        self.dataind = None;
        self.filename = None;
        self.color = 'rgb(0,0,0)';
        self.opacity = 0.7;
        self.typ = None;
        self.init_visible = True;
        self.bar_width = 1;
        self.linecolor = 'rgb(0,0,0)'


        if 'name' in params: self.name = params['name'];
        if 'image' in params: self.image = params['image'];
        if 'x' in params: self.x = params['x'];
        if 'y' in params: self.y = params['y'];
        if 'filename' in params: self.filename = params['filename']
        if 'loc' in params: self.loc = params['loc']
        # if 'dataind' in params: self.dataind = params['dataind'];
        if 'sliders' in params: self.sliders = params['sliders'];
        if 'buttons' in params: self.buttons = params['buttons'];
        if 'color' in params: self.color = params['color'];
        if 'linecolor' in params: self.color = params['linecolor'];
        if 'opacity' in params: self.opacity = params['opacity'];
        if 'typ' in params: self.typ = params['typ']
        if 'init_visible' in params: self.init_visible = params['init_visible'];

    def add(self):
        if self.typ == 'bar':
            self.fig.add_trace(go.Bar(x=self.x,y=self.y, base = 'overlay',width=1.,
                            marker = {'color':self.color,'opacity':self.opacity,'line':dict(width=0.1,color=self.linecolor)}),row=self.loc[0],col=self.loc[1]);
        if self.typ == 'box':
            edgecolor = 'rgba(0,0,0)'
            # boxpoints = 'all'
            self.fig.add_traces(go.Box(
                           y=self.y,fillcolor=self.color,opacity=self.opacity,
                           marker_color=edgecolor,width=0.4,line={'width':1},name=self.name),
                           self.loc[0],self.loc[1]);
        if self.typ == 'image':
            img = px.imshow(sio.imread(self.filename))
            self.fig.add_trace(img.data[0], self.loc[0],self.loc[1])

        if self.init_visible == False:
            self.fig.data[-1].visible = False;

        self.dataind = len(self.fig.data)-1;
        return self.dataind;


#                 fig.add_traces(go.Box(y=VALS,fillcolor=color,opacity=opac,marker_color=edgecolor,boxpoints=boxpoints,width=0.4,line={'width':1},name=series_name),inds[0],inds[1]);
#             ###########
    #         #     fig.add_trace(go.Bar(x=counts2,y=costs2,width=0.5,base ='overlay',marker = {'color' :color,'opacity':0.5}),inds2[0],inds2[1])

        # # for i in range(len(fig.data)): 
# #     if not(i in all_start_inds):
# #         fig.data[i].visible = False;



    # def show(self,fig):
        #     fig.add_trace(go.Bar(x=counts2,y=costs2,width=0.5,base ='overlay',marker = {'color' :color,'opacity':0.5}),inds2[0],inds2[1])





class SLIDER:
    def __init__(self,params):
        self.loc = [0,0];
        self.xanchor = "left"
        self.yanchor = "middle"
        self.active = 0;
        self.length = 0.4; 
        self.pad = {'t':0,'b':0,'r':100,'l':-100};
        self.currentvalue = {};

        if 'loc' in params: self.loc = params['loc'];
        if 'length' in params: self.length = params['length']
        if 'xanchor' in params: self.xanchor = params['xanchor']
        if 'yanchor' in params: self.yanchor = params['yanchor']
        if 'active' in params: self.active = params['active'];
        if 'pad' in params: self.pad = params['pad'];
        if 'currentvalue' in params: self.currentvalue = params['currentvalue'];

        self.steps = [];
        self.datainds = [];
        self.turnoninds = [];
        self.num_steps = 0;
        # self.add();

    # def add(self):
    #     ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS

    def addTrace(self,turnoninds,dataind):
        maxturnonind = int(np.max(turnoninds))
        if self.num_steps < maxturnonind + 1:
            new_num_steps = maxturnonind + 1;
            diff_size = new_num_steps - self.num_steps;
            self.turnoninds = self.turnoninds + [[] for _ in range(diff_size)];
            self.num_steps = new_num_steps

        if dataind in self.datainds:
            dataindloc = np.where(dataind == np.array(self.datainds))[0][0]
            for turnonind in turnoninds:
                self.turnoninds[turnonind].append(dataindloc);
        else:
            self.datainds.append(dataind)
            for turnonind in turnoninds:
                self.turnoninds[turnonind].append(len(self.datainds)-1);
            

    def addSteps(self):
        self.steps = [];
        for i in range(self.num_steps):
            STEP = dict(method="update",label='',args=[{'visible':[False for _ in range(len(self.datainds))]},{},self.datainds]);
            # for j,on_ind in enumerate(self.sliderinds[i]):
            for turnonind in self.turnoninds[i]:
                STEP['args'][0]['visible'][turnonind] = True;
            self.steps.append(STEP.copy())
        # self.add()
        self.dict = dict(x=self.loc[0],y=self.loc[1],len=self.length,
                         xanchor=self.xanchor,yanchor=self.yanchor,
                         active=self.active,pad=self.pad,
                         steps=self.steps,currentvalue=self.currentvalue)        


    # def addImage(self,turnoninds,dataind):
    #     maxturnonind = int(np.max(turnoninds))
    #     if self.num_steps < maxturnonind + 1:
    #         new_num_steps = maxturnonind + 1;
    #         diff_size = new_num_steps - self.num_steps;
    #         self.turnoninds = self.turnoninds + [[] for _ in range(diff_size)];
    #         self.num_steps = new_num_steps
    #     for turnonind in turnoninds:
    #         self.turnoninds[turnonind].append(len(self.datainds));
    #     self.datainds.append(dataind)



    # def addTrace(self,turnoninds,dataind):
    #     maxturnonind = int(np.max(turnoninds))
    #     if self.num_steps < maxturnonind + 1:
    #         new_num_steps = maxturnonind + 1;
    #         diff_size = new_num_steps - self.num_steps;
    #         self.turnoninds = self.turnoninds + [[] for _ in range(diff_size)];
    #         self.num_steps = new_num_steps

    #     for turnonind in turnoninds:
    #         self.turnoninds[turnonind].append(len(self.datainds));
    #     self.datainds.append(dataind)
        



    # def addStep(self,sliderind,dataind):
    #     if self.num_steps < sliderind + 1:
    #         new_num_steps = sliderind + 1;
    #         diff_size = new_num_steps - self.num_steps;

    #         val = None

    #         self.datainds = self.datainds + [val for _ in range(diff_size)];
    #         for i in range(new_num_steps):
    #             if i < self.num_steps:
    #                 STEP = self.steps[i]
    #                 STEP['args'][0]['visible'] = STEP['args'][0]['visible'] + [False for _ in range(diff_size)];
    #                 STEP['args'][2] = STEP['args'][2] + [val for _ in range(diff_size)];
    #             else:
    #                 STEP = dict(method="update",label='',args=[{'visible':[False for _ in range(new_num_steps)]},{},[val for _ in range(new_num_steps)]]);
    #                 self.steps.append(STEP)
    #         self.num_steps = new_num_steps;
            
    #     # for i in range(self.num_steps):
    #     STEP = self.steps[sliderind]
    #     STEP['args'][0]['visible'][sliderind] = True;
    #     STEP['args'][2][sliderind] = dataind;
    #     self.datainds[sliderind] = dataind
        


    # def addStep(self,sliderind,dataind):
    #     if self.num_steps < sliderind + 1:
    #         new_num_steps = sliderind + 1;
    #         diff_size = new_num_steps - self.num_steps;
    #         self.datainds = self.datainds + [None * diff_size];
    #         for i in range(new_num_steps):
    #             if i < self.num_steps:
    #                 STEP = self.steps[i]
    #                 STEP['args'][0]['visible'] = STEP['args'][0]['visible'] + [False * diff_size];
    #                 STEP['args'][2] = STEP['args'][2] + [None for _ in range(diff_size)];
    #             else:
    #                 STEP = dict(method="update",label='',args=[{'visible':[False*new_num_steps]},{},[None for _ in range(new_num_steps)]]);
    #                 self.steps.append(STEP)
    #     self.num_steps = new_num_steps;
    #     for i in range(self.num_steps):
    #         STEP = self.steps[i]
    #         STEP['args'][0]['visible'][sliderind] = True;
    #         STEP['args'][2][sliderind] = dataind;
    #     self.datainds[sliderind] = dataind













class BUTTON:
    def __init__(self,params):
        self.loc = [0,0];
        self.xanchor = "left"
        self.yanchor = "top"
        self.active = 0;
        self.name = 'blarg'
        self.display_name = 'blarg';
        self.typ = 'toggle'
        self.pad = {'t':0,'b':0,'r':0,'l':0};
        self.currentvalue = {};
        self.direction = 'down';

        if 'name' in params: self.name = params['name'];
        if 'typ' in params: self.typ = params['typ'];        
        if 'loc' in params: self.loc = params['loc'];
        if 'length' in params: self.length = params['length']
        if 'xanchor' in params: self.xanchor = params['xanchor']
        if 'yanchor' in params: self.yanchor = params['yanchor']
        if 'active' in params: self.active = params['active'];
        if 'pad' in params: self.pad = params['pad'];
        if 'currentvalue' in params: self.currentvalue = params['currentvalue'];
        if 'direction' in params: self.direction = params['direction'];
        if 'display_name' in params: self.display_name = params['display_name'];

        self.steps = [];
        self.datainds = [];
        self.num_steps = 0;
        # self.add();

    def addTrace(self,dataind):
        self.datainds.append(dataind);

    def add(self):
        if self.typ == 'toggle':
            # print('creating button...')
            # print(self.name)
            # print(self.datainds)
            # print(self.display_name)
            # print('')
            self.dict = dict(buttons=list([
                        dict(args=[{"visible":[True]*len(self.datainds)},{},self.datainds],label = self.display_name + " ON" ,method="update"),
                        dict(args=[{"visible":[False]*len(self.datainds)},{},self.datainds],label= self.display_name + " OFF",method="update")]),
                        # type = "buttons",
                        direction=self.direction,
                        pad=self.pad,
                        # height= 100,
                        # style={'font-size': '12px', 'width': '140px', 'display': 'inline-block', 'margin-bottom': '10px', 'margin-right': '5px', 'height':'37px', 'verticalAlign': 'top'},
                        showactive=True,
                        x=self.loc[0],y=self.loc[1],
                        xanchor=self.xanchor,
                        yanchor=self.yanchor);






# fig.update_layout(updatemenus=BUTTONS)

# fig.update_layout(sliders=SLIDERS)
# width = np.sum(column_widths); height = 1.2*np.sum(row_heights); #+padb1*3)
# fig.update_layout(width=width, height=height, xaxis_visible=False, yaxis_visible=False)



# class TRACE:
#     def __init__(self):
#         self.image = None;
#         self.tags = None;
#         self.datas = {'data1':{'tag':None,'x':None,'y':None},
#                       'data2':{'tag':None,'x':None,'y':None}}
#         self.datax = None;
#         self.datay = None;
#         self.dataind = None;
#         self.DF = None;
#         self.sliders 


# class BUTTON:
#     def __init__(self):
#         pass





def generateRandomOutputs():


    OUTPUTS = {};
    OUTPUTS['by_mode'] = {};

    num_data = 4;
    bnds = np.array([[-85.3394,  34.9458],[-85.2494,  35.0658]]);    
    people = ['person'+str(int(400*num)) for num in np.random.rand(num_data)];
    trip_ids = ['trip'+str(int(1000*num)) for num in np.random.rand(3*num_data)];
    seg_ids = {}; seg_num = 0;
    networks = ['gtfs','ondemand','drive','walk'];
    num_segs = 10;

    grp_ids = ['group'+str(int(5*num)) for num in np.random.rand(num_segs)]
    run_ids = ['run'+str(int(5*num)) for num in np.random.rand(num_segs)]


    for network in networks:
        seg_ids[network] = ['seg' + str(int(num+seg_num)) for num in range(num_segs)]
        seg_num = len(seg_ids[network])

    trip_types = [['drive'],['ondemand'],['walk'],
                  ['walk','gtfs','walk'],
                  ['ondemand','gtfs','walk'],
                  ['walk','gtfs','ondemand'],
                  ['ondemand','gtfs','ondemand']];

    time_wind = [21600,36000]
    time_win_diff = 200;
    time_trip_diff = 1000;

    for network in networks:


        num_segs = len(seg_ids[network]);
        num_data = num_segs
        mode = network;
        DATA = {}
        DATA['seg_id'] = seg_ids[network]; #[mode+'_seg'+str(int(1000*num)) for num in np.random.rand(num_data)];

        DATA['trip_ids'] = [];        
        for i in range(num_segs):
            if np.random.rand(1)<0.1: DATA['trip_ids'].append(sample(trip_ids,3))
            elif np.random.rand(1)<0.2: DATA['trip_ids'].append(sample(trip_ids,2))
            else: DATA['trip_ids'].append(sample(trip_ids,1))

        if network == 'gtfs':
            DATA['start_node'] = [str(int(1000*num)) for num in np.random.rand(num_segs)];
            DATA['end_node'] = [str(int(1000*num)) for num in np.random.rand(num_segs)];
        else:
            DATA['start_node'] = [int(1000*num) for num in np.random.rand(num_segs)];
            DATA['end_node'] = [int(1000*num) for num in np.random.rand(num_segs)];

        DATA['start_loc'] = [loc for loc in zip(np.random.uniform(low=bnds[0][0],high=bnds[1][0],size=(num_data)),np.random.uniform(low=bnds[0][1],high=bnds[1][1],size=(num_data)))]
        DATA['end_loc'] = [loc for loc in zip(np.random.uniform(low=bnds[0][0],high=bnds[1][0],size=(num_data)),np.random.uniform(low=bnds[0][1],high=bnds[1][1],size=(num_data)))]
        DATA['people'] = [];
        add_people = [];
        for i in range(num_segs):
            if np.random.rand(1)<0.05: DATA['people'].append(sample(people,2))
            else: DATA['people'].append(sample(people,1))

        DATA['mode'] = [mode] * num_data
        DATA['distance'] = [200*num for num in np.random.rand(num_data)];
        DATA['travel_time'] = [1000*num for num in np.random.rand(num_data)];
        DATA['money'] = [10*num for num in np.random.rand(num_data)];
        DATA['switches'] = [int(3*num+1) for num in np.random.rand(num_data)];

        DATA['uncongested_distance'] = [200*num for num in np.random.rand(num_data)];
        DATA['uncongested_travel_time'] = [1000*num for num in np.random.rand(num_data)];

        if mode == 'ondemand' or mode == 'gtfs':
            DATA['group_ids'] = grp_ids;
            DATA['run_ids'] = run_ids;


        if mode == 'ondemand':

            DATA['pickup_time_start'] = [];
            DATA['pickup_time_end'] = [];
            DATA['dropoff_time_start'] = [];
            DATA['dropoff_time_end'] = [];
            DATA['pickup_time_scheduled'] = []; 
            DATA['dropoff_time_scheduled'] = []; 
            DATA['pickup_time'] = []; 
            DATA['dropoff_time'] = [];

            for i in range(num_segs):

                pickup_start_time = np.random.uniform(low=time_wind[0],high=time_wind[1]-time_win_diff-time_trip_diff,size=1)[0];
                pickup_end_time = pickup_start_time + time_win_diff;
                dropoff_start_time = pickup_start_time + time_trip_diff;
                dropoff_end_time = dropoff_start_time + time_trip_diff;

                scheduled_pickup_time = pickup_start_time + 0.3*time_win_diff;
                scheduled_dropoff_time = pickup_start_time + 0.3*time_win_diff + time_trip_diff;
                pickup_time = pickup_start_time + 0.5*time_win_diff;
                dropoff_time = pickup_start_time + 0.5*time_win_diff + time_trip_diff;

                DATA['pickup_time_start'].append(pickup_start_time)
                DATA['pickup_time_end'].append(pickup_end_time)
                DATA['dropoff_time_start'].append(dropoff_start_time)
                DATA['dropoff_time_end'].append(dropoff_end_time)

                DATA['pickup_time_scheduled'].append(scheduled_pickup_time)
                DATA['dropoff_time_scheduled'].append(scheduled_dropoff_time)
                DATA['pickup_time'].append(pickup_time)
                DATA['dropoff_time'].append(dropoff_time)

            DATA['num_other_passengers'] = [int(4*num) for num in np.random.rand(num_data)]
        DF = pd.DataFrame(DATA)
        OUTPUTS['by_mode'][mode] = DF
        
    # DATA = {};
    # DATA['person'] = people;
    # DATA['trip_ids'] = [];
    # DATA['modes'] = [];
    # DATA['seg_ids'] = [];
    # # DATA['start_node'] = [];
    # # DATA['end_node'] = [];
    # # DATA['start_loc'] = [];
    # # DATA['end_loc'] = [];    
    # for person in people:
    #     DATA['trip_ids'].append(sample(trip_ids,3));
    #     DATA['modes'].append(sample(trip_types,3))
    #     DATA['seg_ids'].append([])
    #     for trip in DATA['modes'][-1]:
    #         DATA['seg_ids'][-1].append([])
    #         for mode in trip:
    #             DATA['seg_ids'][-1][-1].append(sample(seg_ids[mode],1)[0]);

    # num_data = len(people)
    # DATA['start_node'] = [int(1000*num) for num in np.random.rand(len(people))];
    # DATA['end_node'] = [int(1000*num) for num in np.random.rand(len(people))];
    # DATA['start_loc'] = [loc for loc in zip(np.random.uniform(low=bnds[0][0],high=bnds[1][0],size=(num_data)),np.random.uniform(low=bnds[0][1],high=bnds[1][1],size=(num_data)))]
    # DATA['end_loc'] = [loc for loc in zip(np.random.uniform(low=bnds[0][0],high=bnds[1][0],size=(num_data)),np.random.uniform(low=bnds[0][1],high=bnds[1][1],size=(num_data)))]

    # DF = pd.DataFrame(DATA)
    # OUTPUTS['by_person'] = DF;


    # line_ids = ['line'+str(int(40*num)) for num in np.random.rand(40)];
    # bus_ids = ['bus' + str(int(400*num)) for num in np.random.rand(400)];

    # DATA = {};
    # DATA['bus_trip'] = []; #append(gtrip);
    # tags = ['bus_id','line_id','num']
    # DATA
    # for gtrip in gtfs_trips:
    #     DATA['bus_trip'].append(gtrip);




    return OUTPUTS

# Dataframe2: Ondemand info:
# Index: driver_run_id
# Columns: 
# driver_run
# vehicle miles traveled
# passenger miles traveled (desired)
# VMT/PMT
# group: ondemand delivery group
# distance: distance of driver run
# time: travel time of driver run
# total_passengers: 
# time_wpassengers:  list of time spent with each number of passengers
# ex. [time with 0 passengers, time with 1 passenger, time with 2 passengers, …]
# distance_wpassengers: list of distance traveled with each number of passengers..
# ex. [distance with 0 passengers, distance with 1 passenger, distance with 2 passengers,…]

# Dataframe3: Fixed line info: (second priority)
# origin and destination
# time between origin and destination
# transit route number
# trip id gtfs

# cost 
# Dataframe’s for each mode separately 








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



def CONVERT_CSV_TO_PRE_FORMAT(GRAPHS,VEHS, SEG_TYPES, minz, maxz, csv_file_path,max_num_people=None): #,carta_money_costs = {}):

    #     CSV has columns:
    #     index
    #     orig_lon,orig_lat : origin coordinates
    #     dest_lon,dest_lat : destination coordinates 
    #     take_car
    #     take_transit
    #     take_walk
    #     take_ondemand
    #     mass: number of people
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    # df = df.iloc[np.random.permutation(len(df))]

    # print(df)
    # asdfasdfasdf

    # print(type(list(df['median_income'])[0]))
    if not(max_num_people == None):
        samp_inds = np.array(sample(list(range(len(df))),max_num_people));
        df = df.iloc[samp_inds]        
        # df = df.iloc[:max_num_people]

        # PRE[tag] = {};
        # take_car = row['take_car']
        # take_transit = row['take_transit']
        # take_walk = row['take_walk']
        # take_ondemand = row['take_ondemand']
        # mass = 1.;#row['mass']
        # leave_time_start = row['leave_time_start']
        # leave_time_end = row['leave_time_end']
        # arrival_time_start = row['arrival_time_start']
        # arrival_time_end = row['arrival_time_end']

        # orig_loc_str = row['orig_loc'].strip('[]')
        # orig_loc_elements = orig_loc_str.split()        
        # orig_loc = [float(num) for num in orig_loc_elements]

    if not('leave_time_start' in df.columns): df['leave_time_start'] = df['start_pickup_window']
    if not('leave_time_end' in df.columns): df['leave_time_end'] = df['end_pickup_window']
    if not('arrival_time_start' in df.columns): df['arrival_time_start'] = df['start_dropoff_window']
    if not('arrival_time_end' in df.columns): df['arrival_time_end'] = df['end_dropoff_window']





# Index(['tag', 'h_geocode', 'w_geocode', 'total_jobs',
          # 'origin_loc_lat',
#        'origin_loc_lon', 'dest_loc_lat', 'dest_loc_lon', 'pickup_time_0',
#        'pickup_time_0_secs', 'pickup_time_0_str', 'origin_geom', 'dest_geom',
#        'walk_time', 'drive_time', 'origin_lat', 'origin_lon',
#        'destination_lat', 'destination_lon', 'transit_time', 'transfers',
#        'trip1', 'trip2', 'trip3', 'trip4', 'trip5', 'trip6', 'route1',
#        'route2', 'route3', 'route4', 'route5', 'route6', 'mode',
#        'bus_capacity_upto_2hours', 'GEOID', 'median_income', 'margin',
#        'start_pickup_window', 'end_pickup_window', 'start_dropoff_window',
#        'end_dropoff_window', 'mass', 'take_car', 'take_transit', 'take_walk',
#        'take_ondemand', 'MODE', 'TT_car', 'TT_transit', 'TT_walk',
#        'TT_ondemand', 'p_t_car', 'p_t_transit', 'p_t_walk', 'p_t_ondemand',
#        'r_t_car', 'r_t_transit', 'r_t_walk', 'r_t_ondemand', 'k_t_car',
#        'k_t_transit', 'k_t_walk', 'k_t_ondemand', 'd_t_car', 'd_t_transit',
#        'd_t_walk', 'd_t_ondemand', 'cost_car', 'cost_transit', 'cost_walk',
#        'cost_ondemand'],
#       dtype='object')

    
    # socio_economic_fp = './data/census_data_hamiliton/2021_census_tract_hamilton.geojson' # Socio economic data path
    # socio_economic_df = gpd.read_file(socio_economic_fp) # Reading the socio economic data
    # socio_economic_df = socio_economic_df[['geometry','median_income_last12months']] # Getting only the required columns
    # median_income = socio_economic_df['median_income_last12months'].median()
    # median_income = 50000;
    
    NODES = {}

    NODES['orig'] = []; NODES['dest'] = [];
    NODES['delivery1'] = [];  NODES['delivery2'] = [];
    NODES['delivery1_transit'] = []; NODES['delivery2_transit'] = [];
    NODES['drive_transit'] = [];

    LOCS = {};
    LOCS['orig'] = []; LOCS['dest'] = [];
    LOCS['delivery1'] = []; LOCS['delivery2'] = []
    SIZES = {};
    # Initialize the PRE object
    PRE = {}
    home_locs = []
    work_locs = []
    people_tags = []
    home_nodes = []
    work_nodes = []
    home_sizes = {}
    work_sizes = {}
    
    # cost_transit = 1.5  # Cost of transit trip
    # cost_microtransit = 8  # Cost of microtransit trip
    
    # Convert the DataFrame to the PRE format
    for index, row in df.iterrows():

        try: 
            median_income = row['median_income']; #50000;
            if isinstance(median_income,str):
                median_income = float(median_income)
        except:
            median_income = 30000.
    
        tag = f"person{index}"
        people_tags.append(tag)
        PRE[tag] = {};
        take_car = row['take_car']
        take_transit = row['take_transit']
        take_walk = row['take_walk']
        take_ondemand = row['take_ondemand']
        mass = 1.;#row['mass']



        leave_time_start = row['leave_time_start']
        leave_time_end = row['leave_time_end']
        arrival_time_start = row['arrival_time_start']
        arrival_time_end = row['arrival_time_end']


        # orig_loc_str = row['orig_loc'].strip('[]')
        # orig_loc_elements = orig_loc_str.split()        
        # orig_loc = [float(num) for num in orig_loc_elements]
        # dest_loc_str = row['dest_loc'].strip('[]')
        # dest_loc_elements = dest_loc_str.split()
        # dest_loc = [float(num) for num in dest_loc_elements]

        if True: #'orig_loc' in df.columns:
            orig_loc = eval(row['orig_loc']);
            dest_loc = eval(row['dest_loc']);
        else:
            orig_loc = [row['origin_loc_lon'],row['origin_loc_lat']]
            dest_loc = [row['dest_loc_lon'],row['dest_loc_lat']]
        
        # print(eval(orig_loc))
        # print(eval(dest_loc))
        # print(type(orig_loc))
        # print(type(dest_loc))

        PRE[tag]['take_car'] = take_car
        PRE[tag]['take_transit'] = take_transit
        PRE[tag][ 'take_ondemand'] = take_ondemand
        PRE[tag]['take_walk'] = take_walk
        PRE[tag]['leave_time_start'] = leave_time_start
        PRE[tag]['leave_time_end'] = leave_time_end
        PRE[tag]['arrival_time_start'] = arrival_time_start
        PRE[tag]['arrival_time_end'] = arrival_time_end
        
#         orig_loc = np.array([row['orig_long'], row['orig_lat']])
#         dest_loc = np.array([row['dest_long'], row['dest_lat']])
        
    
        home_loc = orig_loc;    
        work_loc = dest_loc;
        PRE[tag]['orig_loc'] = home_loc
        PRE[tag]['dest_loc'] = work_loc;
        
        VALS = np.abs(VEHS['lon']-home_loc[0])+np.abs(VEHS['lat']-home_loc[1]);
        mask1 = VALS == np.min(VALS);

        perc_wcars = list(VEHS[mask1]['percent_w_cars'])[0]
        perc_wnocars = list(VEHS[mask1]['percent_wout_cars'])[0]
        
        home_size = mass; 
        work_size = mass; 
        

        test1 = (maxz[0]>=home_loc[0]) and (maxz[1]>=home_loc[1]);
        test2 = (minz[0]<=home_loc[0]) and (minz[1]<=home_loc[1]);
        test3 = (maxz[0]>=work_loc[0]) and (maxz[1]>=work_loc[1]);
        test4 = (minz[0]<=work_loc[0]) and (minz[1]<=work_loc[1]);
        
        if(test1 and test2 and test3 and test4 == False):
            PRE.pop(tag, None)
        else:
        ##### Adding populations.... 
         
        #### CASE 1: If the population has a car
            if perc_wcars > 1e-6:
            
                LOCS['orig'].append(home_loc);
                LOCS['dest'].append(work_loc);

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

                ## seg_types: list of different travel modes... 
                # [('drive',),
                #  ('ondemand',),
                #  ('walk','gtfs','walk'),
                #  ('walk','gtfs','ondemand'),
                #  ('ondemand','gtfs','walk'),
                #  ('ondemand','gtfs','ondemand')
                #  ];

                else: 
                    seg_types = SEG_TYPES['car_only'] #[('drive',)]

                PRE[tag]['seg_types'] = seg_types

            #### Case 2: If the population doesn't have a car. 
            if perc_wnocars > 1e-6:
        
                home_node = ox.distance.nearest_nodes(GRAPHS['drive'],home_loc[0],home_loc[1]);
                work_node = ox.distance.nearest_nodes(GRAPHS['drive'], work_loc[0],work_loc[1]);


                if home_node in home_sizes: home_sizes[home_node] = home_sizes[home_node] + home_size;
                else: home_sizes[home_node] = home_size;
                if work_node in work_sizes: work_sizes[work_node] = work_sizes[work_node] + work_size;
                else: work_sizes[work_node] = work_size;

                home_nodes.append(home_node);
                work_nodes.append(work_node);
                NODES['orig'].append(home_node);
                NODES['dest'].append(work_node);

                PRE[tag]['home_node'] = home_node;        
                PRE[tag]['work_node'] = work_node;

                PRE[tag]['pop'] = home_size*perc_wnocars;
                    
                seg_types = SEG_TYPES['car_no']
                ## seg_types: list of different travel modes... 
                # [('drive',),
                #  ('ondemand',),
                #  ('walk','gtfs','walk'),
                #  ('walk','gtfs','ondemand'),
                #  ('ondemand','gtfs','walk'),
                #  ('ondemand','gtfs','ondemand')
                #  ];   
        
      

        lat = PRE[tag]['orig_loc'][1]
        lon = PRE[tag]['orig_loc'][0]
        # Create a pandas DataFrame with a single row
        df = pd.DataFrame({'Latitude': [lat], 'Longitude': [lon]})

        # Convert the pandas DataFrame into a GeoPandas DataFrame with a Point geometry
        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

        # Converting to Geopandas 
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        # Perform a spatial join between the MultiPolygon GeoPandas DataFrame and the Point GeoPandas DataFrame
        # result = gpd.sjoin(gdf, socio_economic_df, how="inner", op='intersects')
        result = gdf;
        result.reset_index(drop=True,inplace=True)

        # Storing the median income - have a try and except block incase points don't lie in the census data
        try:
            PRE[tag]['median_income'] = result['median_income_last12months'][0]
        except:
            PRE[tag]['median_income'] = median_income
            
        #Weights assigned to convenience and number of switches in Drive mode.
        DWC = 0; 
        #Weights assigned to convenience and number of switches in Walk mode.
        WWC = 0; 
        #Weight assigned to convenience in Transit mode.
        TWC = 0
        #Weight assigned to convenience in Ondemand mode.
        OWC = 0




        #Weight assigned to time in all modes.
        DWT = WWT = TWT = OWT = 1
        
        # # Weekly Income Calculation
        # daily_income = PRE[tag]['median_income'] / (52 * 5)  # Convert annual income to weekly income

        # # Assigning weights based on weekly income
        # DWM =  1/daily_income #weight for drive
        # TWM = cost_transit / daily_income  # Weight for transit
        # OWM = cost_microtransit / daily_income  # Weight for microtransit
        # WWM = 0


        DWS = 0;
        WWS = 0;




        #### DAN update...
        money_weight = (1./PRE[tag]['median_income'])*(52./1.)*(40./1.)*(3600./1.)
        TWM = money_weight;
        OWM = money_weight;
        WWM = money_weight; # not applied.
        DWM = money_weight;
        # OWM = (1./PRE[tag]['median_income'])*(52./1.)*(40./1.)*(3600./1.)


        #Weight assigned according to the number of switches in Transit mode. Upper bound on the number of transfers --> 7 (5 bus transfers and 2 otherwise)
        TWS = [  0.0,
                 0.14285714285714285,
                 0.2857142857142857,
                 0.42857142857142855,
                 0.5714285714285714,
                 0.7142857142857143,
                 0.8571428571428571,
                 1.0 ]

        #Weight assigned according to the number of transfers in Ondemand mode. Upper bound on the number of transfers --> 7 (5 bus transfers and 2 otherwise)
        OWS= [   0.0,
                 0.14285714285714285,
                 0.2857142857142857,
                 0.42857142857142855,
                 0.5714285714285714,
                 0.7142857142857143,
                 0.8571428571428571,
                 1.0 ]
        DWS = [0]
        WWS = [0]


        DWS = [20.*DWM*fac for fac in DWS];
        WWS = [20.*WWM*fac for fac in WWS];
        TWS = [20.*TWM*fac for fac in TWS];        
        OWS = [20.*OWM*fac for fac in OWS];
        

        #Adding logistic choice weights to PRE object.
        PRE = add_logistic_values (PRE, tag, DWT, DWM, DWC, DWS,
                             WWT , WWM, WWC, WWS,
                             TWT, TWM, TWC, TWS, 
                             OWT, OWM, OWC, OWS)




    SIZES['home_sizes'] = home_sizes
    SIZES['work_sizes'] = work_sizes


    # def setup_ondemand_service(PRE, NODES, LOCS, SIZES, GRAPHS, params, people_tags):
    
    start_time = time.time()
    # print('starting delivery1 sources...')
    # for i,node in enumerate(NODES['delivery1']):
    #     if np.mod(i,200)==0: print(i)
    #     addNodeToDF(node,'drive',GRAPHS,NDF)
        
    # print('starting delivery2 sources...')
    # for i,node in enumerate(NODES['delivery2']):
    #     if np.mod(i,200)==0: print(i)
    #     addNodeToDF(node,'drive',GRAPHS,NDF)
            
    
            
    # # end_time = time.time()
    # # print('time to create nodes...: ',end_time-start_time)
    # num_people = len(people_tags);
    # num_targets = num_people;

    # if 'num_deliveries' in params:
    #     num_deliveries = params['num_deliveries']['delivery1'];
    #     num_deliveries2 = params['num_deliveries']['delivery2'];
    # else:
    num_people = len(people_tags);
    num_deliveries =  int(num_people/10);
    num_deliveries2 = int(num_people/10);


    ##### k-means clustering of the population locations to see where the different ondemand vehicles should go
    ### CHANGED TO RUN FOR PILOT REGION....
    node_group = NODES['orig'] + NODES['dest']
    out = kmeans_nodes(num_deliveries,'ondemand',GRAPHS,node_group); 
    LOCS['delivery1'] = out['centers']
    out = kmeans_nodes(num_deliveries,'ondemand',GRAPHS,node_group); 
    LOCS['delivery2'] = out['centers']
    
    
    for i,loc in enumerate(LOCS['delivery1']):
        NODES['delivery1'].append(ox.distance.nearest_nodes(GRAPHS['ondemand'],loc[0],loc[1]));
    for i,loc in enumerate(LOCS['delivery2']):
        NODES['delivery2'].append(ox.distance.nearest_nodes(GRAPHS['ondemand'],loc[0],loc[1]));
    bus_graph = GRAPHS['gtfs']; #bus_graph_wt'];
    delivery_transit_nodes = sample(list(bus_graph.nodes()), num_deliveries2)
    end_time = time.time();
    print('time to setup origins & dests: ',end_time - start_time)
    
    print("Done setting up on-demand service")
    
    # return PRE, NODES, LOCS, SIZES    
    # print("Using cost: ", cost_microtransit)
    # directory = 'Scenario_1_new'
    # os.makedirs(directory, exist_ok=True)
    # sample_output_path = os.path.join(directory, 'PRE_micro_8.csv')
    # save_PRE(PRE, sample_output_path)
    # print("Saving PRE ...")
    OUT = {}
    OUT['PRE'] = PRE
    OUT['NODES'] = NODES
    OUT['LOCS'] = LOCS
    OUT['SIZES'] = SIZES
    OUT['VEHS'] = VEHS;



    return OUT #PRE, NODES, LOCS, SIZES, people_tags


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

    # if 'num_deliveries' in params:
    #     num_deliveries = params['num_deliveries']['delivery1'];
    #     num_deliveries2 = params['num_deliveries']['delivery2'];
    # else:
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


# class DASHBOARD:
#     def __init__(self): pass
#     def sortData(self): pass
#     def generate(self): pass


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

# def convertNode(selg , node,from_mode,to_mode,from_type = 'graph',to_type = 'graph',verbose = False):

#     """
#     description -- converts nodes (at approx the same location) between two different graph types 
#     inputs --
#            node: node to convert
#            from_type: initial mode node is given in
#            to_type: desired node type
#            NODES: node conversion dict - contains dataframes with conversion information
#     returns --
#            node in desired mode
#     """
#     #######
#     from_mode2 = from_mode
#     to_mode2 = to_mode;
#     if from_type == 'feed': from_mode2 = 'feed_'+from_mode;
#     if to_type == 'feed': to_mode2 = 'feed_'+to_mode
#     # if not(from_mode2 in self.NINDS): self.addNodeToConverter(node,from_mode2,node_type=from_type):
#     if not(node in selg.NINDS[from_mode2]): selg.addNodeToConverter(node,from_mode2,node_type=from_type)

#     node_index = selg.NINDS[from_mode2][node];
#     to_node = selg.NDF.loc[node_index][to_mode2]
#     return to_node            

class CONVERTER: 
    def __init__(self,params={}):

        self.modes = params['modes'];
        self.GRAPHS = params['GRAPHS']
        self.FEEDS = params['FEEDS']
        self.NINDS = {};
        for mode in self.GRAPHS: self.NINDS[mode] = {};
        for mode in self.FEEDS: self.NINDS['feed_'+mode] = {};        

 

        self.NDF = pd.DataFrame({mode:[] for mode in self.modes},index=[]);
        if 'NDF' in params: self.NDF = params['NDF']
        # else: self.NDF = pd.DataFrame({mode:[] for mode in self.modes},index=[])
        if 'NINDS' in params: self.NINDS = params['NINDS'];
        # else: self.NINDS = {}
        


    def convertNode(self , node,from_mode,to_mode,from_type = 'graph',to_type = 'graph',verbose = False):

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
            self.csv_file_path = params['csv_file_path'];

        else: 
            self.verbose = self.LOADED['OTHER']['verbose']
            self.inputparams = self.LOADED['OTHER']['inputparams']
            self.main = self.LOADED['OTHER']['main']
            self.factors = self.LOADED['OTHER']['factors']
            self.modes = self.LOADED['OTHER']['modes']
            self.bnds = self.LOADED['OTHER']['bnds'];



        self.load_fits = False;
        self.save_fits = False;
        if 'load_ondemand_fits' in params: self.load_fits = True; 

        if 'load_ondemand_fits' in params: self.load_fits_file = params['load_ondemand_fits'];
        if 'save_ondemand_fits' in params: self.save_fits_file = params['save_ondemand_fits'];

        self.num_drivers_per_group = {}
        if 'num_drivers_per_group' in params: self.num_drivers_per_group = params['num_drivers_per_group']




        if 'max_num_people' in params: self.max_num_people = params['max_num_people']
        else: self.max_num_people = None


        self.start_time = 0.
        self.end_time = 86400.

        if 'time_window' in params: self.start_time = params['time_window'][0];
        if 'time_window' in params: self.end_time = params['time_window'][1];

        self.gtfs_feed_file = params['gtfs_feed_file'];
        self.gtfs_precomputed_file = params['gtfs_precomputed_file'];
        self.groups_regions_geojson = params['groups_regions_geojson']
        if 'background_congestion_file' in params: self.preloads_file = params['background_congestion_file'];

        self.monetary_costs = {'ondemand':0,'gtfs':0,'drive':0,'walk':0};
        if 'monetary_costs' in params: 
            for mode in self.modes:
                if mode in params['monetary_costs']: self.monetary_costs[mode] = params['monetary_costs'][mode];

        for mode in self.modes:
            print('monetary cost of',mode,'segment:',self.monetary_costs[mode], '$')
        


        ####### 
        self.NETWORKS = {}

        self.all_trip_ids = [];

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
            # self.initUNCONGESTED()





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
                self.GRAPHS[mode] = ox.graph_from_place('chattanooga',network_type='drive'); #


                #ox.graph_from_polygon(graph_boundary,network_type='drive') ### UPDATE


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
            params2['time_window'] = [self.start_time,self.end_time];

            if mode == 'ondemand' or mode == 'gtfs':
                params2['monetary_cost'] = self.monetary_costs[mode]

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


    def initSTATS(self,version='agrima_code'):

        path2 = self.groups_regions_geojson
        group_polygons = generate_polygons('from_geojson',path = path2);
        self.group_polygons = group_polygons
        self.grpsDF, = self.createGroupsDF(group_polygons);
        # params3 = {'num_drivers':16,
        #             'num_drivers_per_group': self.num_drivers_per_group,
        #             'am_capacity':8, 'wc_capacity':2,
        #            'start_time' : self.start_time,'end_time' : self.end_time}

        self.driversDF = {};
        for group in list(self.grpsDF['group']):
            num_drivers = 10;
            if group in self.num_drivers_per_group:
                num_drivers = self.num_drivers_per_group[group];
            print('adding',num_drivers,'to',group,'...')
            params3 = {'num_drivers':num_drivers,
                    'am_capacity':8, 'wc_capacity':2,
                   'start_time' : self.start_time,'end_time' : self.end_time}
            self.driversDF[group] = self.createDriversDF(params3)
        self.DELIVERYDF = {'grps':self.grpsDF,'drivers':self.driversDF}


        # print(self.driversDF)


        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 

        start = self.start_time; end = self.end_time; #8*60*60; end = 9*60*60;
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

        if version == 'dan_code':
            OUT = SETUP_POPULATIONS_CHATTANOOGA(self.GRAPHS,cutoff_bnds = cutoff_bnds, params=params);
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- #### ---- 
        if version == 'agrima_code':






            #### initial parameters...
            if 'pop_cutoff' in  params: pop_cutoff = params['pop_cutoff'];
            else: pop_cutoff = 30;

            if 'OD_version' in  params: OD_version = params['OD_version']
            else: OD_version = 'basic'

            SEG_TYPES = params['SEG_TYPES'];
            

            # for i,samp in enumerate(samps):
            #     tag = 'person'+str(i);
            zzgraph = self.GRAPHS['all']
            
            temp = [Point(n['x'],n['y']) for i,n in zzgraph.nodes(data=True)]
            temp2 = np.array([[n['x'],n['y']] for i,n in self.GRAPHS['all'].nodes(data=True)])
            minz = np.min(temp2,0); maxz = np.max(temp2,0);
                # zzgraph.nodes(data=True)])
            # use_box = True;
            # if use_box:
            #     minz = np.min(temp2,0); maxz = np.max(temp2,0);
            #     dfz = maxz-minz; centerz = minz + 0.5*dfz;
            #     skz = 0.9;
            #     pts = 0.5*np.array([[dfz[0],dfz[1]],[-dfz[0],dfz[1]],[-dfz[0],-dfz[1]],[dfz[0],-dfz[1]]]) + centerz;
            #     points = [Point(zz[0],zz[1]) for i,zz in enumerate(pts)]
            #     temp = temp + points;
            # graph_boundary = gpd.GeoSeries(temp).unary_union.convex_hull
            
            # cornersz = np.array([[maxz[0],maxz[1]],[minz[0],maxz[1]],[minz[0],minz[1]],[maxz[0],minz[1]]]);
            # #corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
            # divx = 40; divy = int(divx*(dfz[1]/dfz[0]));
            # ptBnds = ptsBoundary(cornersz,[divx,divy,divx,divy])
            # # plt.plot(ptBnds[:,0],ptBnds[:,1],'o')











            ### Reading in the loads data set... (pandas dataframes)
            #asdf0 = pd.read_parquet('data/pop/lodes_combinations_upd.parquet')
            # asdf0.head()
            
            ##### forget what this does.... 
            BGDEFS = pygris.block_groups(state = "TN", county="Hamilton", cb = True, cache=True)
            BGDEFS['pt']  = BGDEFS['geometry'].representative_point()
            BGDEFS['lon'] = BGDEFS['pt'].x;
            BGDEFS['lat'] = BGDEFS['pt'].y;

            #### Reading American Commuter Survey data set (pandas dataframes)
            ### information about vehicle ussage 




            #### Reading American Commuter Survey data set (pandas dataframes)
            ### information about vehicle ussage 
            VEHS = pd.read_csv('data/pop/ACSDT5Y2020.B992512-Data.csv')
            # BGDEFS['AFFGEOID']
            #VEHS = VEHS.rename(columns={'B992512_001E':'from_cbg','home_geo':'from_geo','w_geocode':'to_cbg','work_geo':'to_geo'}).drop(columns=['return_time'])[['from_cbg', 'to_cbg', 'total_jobs', 'go_time', 'from_geo', 'to_geo']]
            VEHS = VEHS.rename(columns={'GEO_ID':'AFFGEOID','B992512_001E':'workers','B992512_002E':'wout_cars','B992512_003E':'w_cars'}).drop(columns=['B992512_001EA','B992512_002EA','B992512_003EA','Unnamed: 8'])
            VEHS = VEHS.drop([0])
            ### computing the percentage of workers with and without cars... (within pandas)
            VEHS['workers'] = pd.to_numeric(VEHS['workers'],errors='coerce')
            VEHS['wout_cars'] = pd.to_numeric(VEHS['wout_cars'],errors='coerce')
            VEHS['w_cars'] = pd.to_numeric(VEHS['w_cars'],errors='coerce')
            VEHS['percent_w_cars'] = VEHS['w_cars']/VEHS['workers'];
            VEHS['percent_wout_cars'] = VEHS['wout_cars']/VEHS['workers'];
            VEHS = VEHS.merge(BGDEFS,how='left',on='AFFGEOID')
            self.VEHS = VEHS
            self.SEG_TYPES = generate_segtypes('reg6') # reg1,reg2,bg

            # carta_money_costs = {'ondemand':self.cost_of_ondemand,'transit':self.cost_of_transit}
            OUT = CONVERT_CSV_TO_PRE_FORMAT(self.GRAPHS,self.VEHS, self.SEG_TYPES, minz, maxz, self.csv_file_path,max_num_people = self.max_num_people); #carta_money_costs = carta_money_costs)





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
        params2['group_polygons'] = self.group_polygons

        # if hasattr(self,'load_fits_file'): params2['load_fits_file'] = self.load_fits_file
        # if hasattr(self,'save_fits_file'): params2['save_fits_file'] = self.save_fits_file

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

            # print(self.start_time);
            # print(self.end_time);

            self.PEOPLE[person] = PERSON(person,
                self.CONVERTER,
                self.GRAPHS,
                self.FEEDS,
                self.NETWORKS,
                self.ONDEMAND,
                self.PRE,
                {'modes':self.modes,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'factors':self.factors},
                len(self.all_trip_ids)
                )
            self.all_trip_ids = self.all_trip_ids + self.PEOPLE[person].trip_ids;

    def initBACKGROUND(self):

        if hasattr(self,'preloads_file'):
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

    def createGroupsDF(self,polygons,types0 = []):  ### ADDED TO CLASS
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
        return GROUPS,

    # def createDriversDF(self): 
    def createDriversDF(self,params): #WORLD):  ### ADDED TO CLASS



        num_drivers = params['num_drivers']
        if not('start_time' in params): start_times = [self.start_time for _ in range(num_drivers)]
        elif not(isinstance(params['start_time'],list)): start_times = [params['start_time'] for _ in range(num_drivers)]

        if not('end_time' in params): end_times = [self.end_time for _ in range(num_drivers)]
        elif not(isinstance(params['end_time'],list)): end_times = [params['end_time'] for _ in range(num_drivers)]

        if not('am_capacity' in params): am_capacities = [8 for _ in range(num_drivers)];
        elif not(isinstance(params['am_capacity'],list)): am_capacities = [params['am_capacity'] for _ in range(num_drivers)]        
        if not('wc_capacity' in params): wc_capacities = [2 for _ in range(num_drivers)];
        elif not(isinstance(params['wc_capacity'],list)): wc_capacities = [params['wc_capacity'] for _ in range(num_drivers)]
        OUT = pd.DataFrame({'start_time':start_times, 'end_time':end_times,'am_capacity':am_capacities, 'wc_capacity':wc_capacities})
        return OUT


    def plotPRELIMINARIES(self,include_ods = True,include_grp_regions = True, include_demand_curves=False,save_file=None):
        #fig,axs = plt.subplots(1,3,figsize=(12,4));


        if include_ods and include_grp_regions:
            GROUPS = self.ONDEMAND.groups
            colors = [GROUPS[group].polygon_color for group in GROUPS]
            plotODs(self.GRAPHS,self.SIZES,self.NODES,scale=100.,figsize=(5,5),with_regions=True,group_polygons=self.group_polygons,colors = colors,save_file=save_file); #,ax=axs[0])

        elif include_ods:
            plotODs(self.GRAPHS,self.SIZES,self.NODES,scale=100.,figsize=(5,5),save_file=save_file); #,ax=axs[0])
        elif include_grp_regions:
            plotShapesOnGraph(self.GRAPHS,self.group_polygons,figsize=(5,5),save_file=save_file); #,ax=axs[2]); #axs[1]);
        # fig,ax = plt.subplots(1,1,figsize=(5,5))
        if include_demand_curves:
            self.ONDEMAND.plotCongestionModels(); #ax=ax); #s[2])

    def fitModels(self,counts = {'num_counts':2,'num_per_count':1}):
        
        already_loaded = False;
        if self.load_fits:
            if hasattr(self,'load_fits_file'):               
                if os.path.exists(self.load_fits_file):
                    print('loading demand functions...')
                    print('(if you want to regenerate them set WORLD.load_fits = False and rerun...)')
                    print('')
                    self.loadFits();
                    already_loaded = True;
                else:
                    print('load file does not exist.')
                    print('continuing to regenerate fits...')
            else:
                print('load fits file not specified.')
                print('continuing to regenerate fits...')
        else:
            print('load fits not specified...')
            print('generating fits...')

        if not(already_loaded):
            print('updating individual choices...')
            for i,person in enumerate(self.PEOPLE):
                PERSON = self.PEOPLE[person]
                if np.mod(i,200)==0: print(person,'...')
                PERSON.UPDATE(self.NETWORKS,takeall=True)
            NETWORK = self.NETWORKS['ondemand'];
            self.ONDEMAND.generateCongestionModels(NETWORK,counts = counts,verbose=self.verbose); 

    def saveFits(self):
        fits_data = {};
        GROUPS = self.ONDEMAND.groups
        for i,group in enumerate(self.ONDEMAND.groups):
            GROUP = GROUPS[group];
            polygon = GROUP.polygon
            centroid = np.mean(np.array(polygon),0)
            fits_data[group] = {'centroid':centroid,'polygon':polygon,'fit':GROUP.fit}
        if self.save_fits == True: #hasattr(self,'save_fits_file'):
            handle = open(self.save_fits_file,'wb');
            pickle.dump(fits_data,handle)

    def loadFits(self):
        
        handle = open(self.load_fits_file,'rb')
        fitsload = pickle.load(handle);
        GROUPS = self.ONDEMAND.groups;
        loaded_group_tags = list(fitsload)
        loaded_fits = [fitsload[grp]['fit'] for grp in fitsload];
        loaded_centroids = [list(fitsload[grp]['centroid']) for grp in fitsload]
        loaded_centroids = np.array(loaded_centroids)
        for group in GROUPS:
            GROUP = GROUPS[group]
            polygon = np.array(GROUP.polygon)
            centroid = np.mean(polygon,0)
            diffs = loaded_centroids - centroid 
            mags = np.array([mat.norm(diff) for diff in diffs])
            ind = np.where(mags==np.min(mags))[0][0]
            fit = loaded_fits[ind];
            GROUP.fit = fit.copy();
            print('loading poly for',group,':',GROUP.fit['poly']);


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


    ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA 
    ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA ###### LOADING BACKGROUND EDGE DATA 

    def SIM(self,num_iters = 1,restart=True,compute_uncongested = True, force_takeall=False):


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
            pop_guess = 10.;
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


            takeall=False;
            if force_takeall==True:
                takeall=True;
            

            print('updating individual choices...')


            for i,person in enumerate(self.PEOPLE):
                PERSON = self.PEOPLE[person]
                if np.mod(i,200)==0: print(person,'...')
                PERSON.UPDATE(self.NETWORKS,takeall=takeall)
                # self.PEOPLE.UPDATE(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=False);
            # update_choices(PEOPLE, DELIVERY, NDF, GRAPHS,WORLD,takeall=False);


            end_time = time.time()
            
            print('iteration time: ',end_time-start_time)
            self.main['iter'] = self.main['iter'] + 1.;
            self.main['alpha'] = 10./(self.main['iter']+1.);
            # WORLD['main']['alpha'] = 10.;

        if compute_uncongested:
            self.initUNCONGESTED()

    def makeDashboard(self):

        self.DASH = DASHBOARD();
        self.DASH.makeGrid();
        self.DASH.addOutputs(self.OUTPUTS)


    def generateLayers(self,use_outputs=False,use_all_trips=False):

        params = {};
        params['bgcolor'] = [1,1,1,1]


        GRAPHS = self.GRAPHS;
        ONDEMAND = self.ONDEMAND;
        NETWORKS = self.NETWORKS;
        NODES = self.NODES;
        SIZES = self.SIZES;



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


        wids = { 'base': 4, 'lines':4,
                 'drive': 4.,'walk':2,'gtfs':2,'ondemand':2,
                 'group': 4.,  
                 'direct':1,'ondemand_indiv':10, 
                  'drive_all':4,
                  'walk_all':4,
                  'gtfs_all':4,
                  'ondemand_all':4,
                  'drive_trips':10,
                  'walk_trips':10,
                  'gtfs_trips':10,
                  'ondemand_trips':10,
                  'ondemand_groups':7,
                  'ondemand_runs':7,

                 };
        maxwids = {'drive': 4.,'walk':2,'transit':6,'lines':4,'gtfs':8,'ondemand':2,'direct':1,'ondemand_indiv':10,'base':4.}
        minwids = {'drive': 4.,'walk':2,'transit':6,'lines':4,'gtfs':8,'ondemand':2,'direct':1,'ondemand_indiv':10,'base':4.}
        # mxpop1 = 1.
        mxpops = {'drive': 0.3,'walk':0.01,'transit':0.001,'lines':0.00001,'gtfs':0.001,'ondemand':0.001}

        # params['colors']['shuttle'] = [0.,0.,1.]
        # params['set_alphas'] = {'direct':0.6,'shuttle':0.6}
        # params['set_wids'] = {'direct':4,'shuttle':4}    


        colors = {'base':[0,0,0,1],
                  'drive':[1,0,0,1.],
                  'walk':[1.,1.,0.,1.], #[0.7,0.7,0.7],
                  'lines':[0.,0.,0.,1.],'transit':[1.,0.,1.,1.],'gtfs': [1.,0.5,0.,1.],   
                  'ondemand':[0.,0.,1.,1.],'direct':[0.6,0.,1.,1.],'shuttle':[0.,0.,1.,1.],          
                  'source':[0.,0.,1.,0.5],'target':[1.,0.,0.,0.5], #[0.8,0.8,0.8],
                  'groups':[[0.,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5],[0.7,0.,1.,0.5]],
                  'shuttle_nodes':[1.,0.5,0.,0.5],
                  'ondemand_indiv':[1.,0.,1.0,1.],
                  'ondemand1':[0.,0.,1.0,1.],
                  'ondemand2':[1.,0.,1.,1.],
                  'ondemand_trips1':[0.,0.,0.,1.],
                  'default_edge':[1,1,1,1],

                  'drive_all':[1,0,0,1.],
                  'walk_all':[1,1.,0,1.],
                  'gtfs_all':[1,0.5,0,1.],
                  'ondemand_all':[0.,0.,1.,0.3],
                  'drive_trips':[1,0,0,1.],
                  'walk_trips':[1.,1.,0,1.],
                  'gtfs_trips':[1.,0.5,0.,1.],
                  'ondemand_trips':[0.,0.,1.,1.0],
                  'ondemand_groups':[0.,0.,1.,0.3],
                  'ondemand_runs':[0.,0.,1.,0.3]
                  }

        colors2 = {'drive_all':[1,0,0,0.3],'walk_all':[1,1.,0,1.],'gtfs_all':[1,0.5,0,0.2],'ondemand_all':[0.,0.,1.,0.5],
                   'drive_trips':[0,0,0,1.],'walk_trips':[1.,1.,0,1.],'gtfs_trips':[1.,0.5,0.,1.],'ondemand_trips':[0.,0.,0.,1.0]}

        colorB = np.array([0.,0.,1.])
        colorA = np.array([0.6,0.,1.]); 
        if len(list(ONDEMAND.groups))>0: groups = list(ONDEMAND.groups)
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
        ondemand_group_tags = [];
        # try: 
        for _,group in enumerate(ONDEMAND.groups):
            GROUP = ONDEMAND.groups[group]
            tags.append(group);
            ondemand_group_tags.append(group)

            payload = GROUP.current_payload;
            manifest = GROUP.current_manifest;
            PDF = GROUP.payloadDF(payload,GRAPHS,include_drive_nodes = True);
            MDF = GROUP.manifestDF(manifest,PDF)
            ZZ = MDF[MDF['run_id']==1]
            zz = list(ZZ.sort_values(['scheduled_time'])['drive_node'])
            PLANS = GROUP.routesFromManifest(); #MDF,GRAPHS)
            # PATHS = routeFromStops(zz,GRAPHS['drive'])
            ONDEMAND_EDGES[group] = [PLANS[plan]['edges'] for _,plan in enumerate(PLANS)]
            # DELIVERY_TAGS[group] = []
            ONDEMAND_RUNS[group] = {};
            for _,plan in enumerate(PLANS):
                PLAN = PLANS[plan]
                ONDEMAND_RUNS[group][plan] = PLAN['edges']
                runtag = group + '_run' + str(plan);
                ondemand_run_tags.append(runtag)
            # for i,plan in enumerate(PLANS):
            #     tags.append(group+'_delivery'+str(i))

        # except:
        #     pass


        ONDEMAND_TRIPS = {};
        ondemand_trip_tags = [];

        # try: 
        for _,group in enumerate(ONDEMAND.groups):
            tags.append(group);
            GROUP = ONDEMAND.groups[group]
            payload = GROUP.current_payload;
            manifest = GROUP.current_manifest;
            PDF = GROUP.payloadDF(payload,GRAPHS,include_drive_nodes = True);
            MDF = GROUP.manifestDF(manifest,PDF)
            PLANS = GROUP.singleRoutesFromManifest(); # MDF,GRAPHS)
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

                    # tag = 'ondemand'+'_'+group+'_'+str(node1)+'_'+str(node2);
                    tag = 'ondemand'+'_'+str(node1)+'_'+str(node2);
                    ONDEMAND_TRIPS[tag] = TRIP['edges'];
                    ondemand_trip_tags.append(tag);
        # except:
        #     pass        

        # print(ondemand_run_tags)

        ACTIVE_TRIPS = {}
        for mode in ['drive','walk','gtfs','ondemand']:
            NETWORK = NETWORKS[mode];
            if use_all_trips: active_trips = list(NETWORK.segs)
            else: active_trips = NETWORK.active_segs;
            ACTIVE_TRIPS[mode] = active_trips.copy();


        # print(ACTIVE_TRIPS)
        walk_trip_tags = []; drive_trip_tags = []; gtfs_trip_tags = [];
        TRIP_EDGES = {};
        for mode in ['drive','walk','gtfs']:
            NETWORK = NETWORKS[mode];
            # print(mode)
            TRIP_EDGES[mode] = {}
            active_trips = ACTIVE_TRIPS[mode]
            for trip in active_trips:
                tag = mode + '_' + str(int(trip[0])) + '_' + str(int(trip[1]))
                if trip in NETWORK.segs:
                    SEG = NETWORK.segs[trip]
                    # print(SEG.current_path)
                    if not(SEG.current_path == None):
                        edges = self.edgesFromPath(SEG.current_path)
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
        transit_edges = list(GRAPHS['gtfs'].edges())

        DRIVE = NETWORKS['drive'];
        TRANSIT = NETWORKS['gtfs'];
        ONDEMAND = NETWORKS['ondemand'];
        WALK = NETWORKS['walk'];

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


            if node in GRAPHS['gtfs'].nodes:
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
        transit_edges = list(GRAPHS['gtfs'].edges)

        transit_edges2 = [];
        for i,edge in enumerate(transit_edges):
            transit_edges2.append((edge[0],edge[1],));


        for k,edgex in enumerate(FGRAPH.edges):        
            edge = (edgex[0],edgex[1]);
            edge2 = (edgex[0],edgex[1],0);


            if edge2 in transit_edges:
                tag = 'lines'
                edges[tag].append(edge2)
                edge_colors[tag].append(colors['lines']); #tag])
                edge_widths[tag].append(wids['lines']); #tag]); #SIZES['work_sizes'][node]*node_scale)



            if edge2 in drive_edges:
                tag = 'drive'            
                # drive_mass = WORLD['drive']['current_edge_masses'][edge]
                if hasattr(NETWORKS['drive'],'base_edge_masses'): #.base_edge_masses[edge2]):
                    drive_mass = NETWORKS['drive'].base_edge_masses[edge2]
                    if drive_mass > 0: 
                        edges[tag].append(edge2)
                        edge_colors[tag].append(colors['drive_all']); #[:3]+[0.3*thresh(drive_mass,threshs[tag])])
                        edge_widths[tag].append(wids['drive_all']); #maxwids[tag]*thresh(drive_mass,threshs[tag]))


            if edge2 in walk_edges:
                tag = 'walk'
                walk_mass = NETWORKS[tag].current_edge_masses[edge2]
                if walk_mass > 0: 
                    edges[tag].append(edge2)
                    edge_colors[tag].append(colors['walk_all']); #[:3]+[thresh(walk_mass,threshs[tag])])
                    edge_widths[tag].append(wids['walk_all']); #maxwids[tag]*thresh(walk_mass,threshs[tag]))




            if edge in transit_edges2:
                tag = 'gtfs'            
                gtfs_mass = NETWORKS[tag].current_edge_masses[edge2]
                if gtfs_mass > 0: 
                    edges[tag].append(edge2)
                    edge_colors[tag].append(colors['gtfs_all']); #tag])
                    edge_widths[tag].append(wids['gtfs_all']); #smaxwids[tag]*thresh(gtfs_mass,threshs[tag]))

            for gg,group in enumerate(ONDEMAND_EDGES):
                edge_lists4 = ONDEMAND_EDGES[group];
                for kk,edge_list in enumerate(edge_lists4):
                    if edge in edge_list:
                        tag = 'ondemand'
                        edges[tag].append(edgex)
                        edge_colors[tag].append(colors['ondemand_groups']);#colors['groups'][gg])
                        edge_widths[tag].append(wids['ondemand_groups']); #4.)

                        tag = group;
                        edges[tag].append(edgex)
                        # edge_colors[tag].append(colors['groups'][gg])
                        edge_colors[tag].append(colors['ondemand_groups']); #colors2['groups'][group])
                        edge_widths[tag].append(wids['ondemand_groups'])


                        # tag = group + '_delivery' + str(kk);
                        # edges[tag].append(edgex)
                        # # edge_colors[tag].append(colors['groups'][gg][::3]+[1.])
                        # edge_colors[tag].append(colors2['groups'][group][:3]+[1.])
                        # edge_widths[tag].append(10.)


            for _,trip in enumerate(ONDEMAND_TRIPS):
                edge_list = ONDEMAND_TRIPS[trip];
                # for kk,edge_list in enumerate(edge_lists4):
                if edge in edge_list:
                    tag = trip
                    edges[tag].append(edgex)
                    edge_colors[tag].append(colors['ondemand_trips']); #ondemand_trips1'])
                    edge_widths[tag].append(wids['ondemand_trips']); #10.)

            for _,group in enumerate(ONDEMAND_RUNS):
                RUNS = ONDEMAND_RUNS[group]
                for _,runid in enumerate(RUNS):
                    edge_list = RUNS[runid]
                    # for kk,edge_list in enumerate(edge_lists4):
                    if edge in edge_list:
                        tag = group + '_run' + str(runid)
                        edges[tag].append(edgex)
                        edge_colors[tag].append(colors['ondemand_runs']); #2['groups'][group])
                        edge_widths[tag].append(wids['ondemand_runs']); #10.)



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
                        edge_colors[trip_tag].append(colors[mode+'_trips'])
                        edge_widths[trip_tag].append(wids[mode+'_trips']); #10.)
            
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
        for i,name in enumerate(all_tags):
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
                if name in ondemand_group_tags: layers[name]['subfolder'] = 'groups/'
                if name in ondemand_run_tags: layers[name]['subfolder'] = 'runs/'
# if name in ondemand_trip_tags: layers[name]['subfolder'] = 'ondemand_trips/'
                if 'subfolder' in layers[name]: layers[name]['subpath'] = layers[name]['subfolder'] + layers[name]['name'] + '.png' 
                else: layers[name]['subpath'] = layers[name]['name'] + '.png' 

        # for i,name in enumerate(ondemand_trip_tags):
        #     if len(nodes[name]) > 0  or len(edges[name]) > 0:
        #         layers[name] = {}
        #         layers[name]['name'] = name;
        #         layers[name]['bgcolor'] = [1,1,1,0];
        #         layers[name]['nodes'] = nodes[name].copy();
        #         layers[name]['edges'] = edges[name].copy();
        #         layers[name]['node_colors'] = node_colors[name].copy();
        #         layers[name]['edge_colors'] = edge_colors[name].copy();
        #         layers[name]['node_sizes'] = node_sizes[name].copy();
        #         layers[name]['edge_widths'] = edge_widths[name].copy();
        #         layers[name]['node_edge_colors'] = node_edge_colors[name].copy();
        #         layers[name] = {**basic_layer,**layers[name]}
        #         layers[name]['subfolder'] = 'ondemand_trips/'
        #         layers[name]['subpath'] = layers[name]['subfolder'] + layers[name]['name'] + '.png' 


        # for i,name in enumerate(ondemand_run_tags):
        #     if len(nodes[name]) > 0  or len(edges[name]) > 0:
        #         layers[name] = {}
        #         layers[name]['name'] = name;
        #         layers[name]['bgcolor'] = [1,1,1,0];
        #         layers[name]['nodes'] = nodes[name].copy();
        #         layers[name]['edges'] = edges[name].copy();
        #         layers[name]['node_colors'] = node_colors[name].copy();
        #         layers[name]['edge_colors'] = edge_colors[name].copy();
        #         layers[name]['node_sizes'] = node_sizes[name].copy();
        #         layers[name]['edge_widths'] = edge_widths[name].copy();
        #         layers[name]['node_edge_colors'] = node_edge_colors[name].copy();
        #         layers[name] = {**basic_layer,**layers[name]}
        #         layers[name]['subfolder'] = 'runs/'
        #         layers[name]['subpath'] = layers[name]['subfolder'] + layers[name]['name'] + '.png' 




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
        layers[tag]['edge_widths'] = wids[tag]*np.ones(ne)
        layers[tag] = {**basic_layer,**layers[tag]}
        layers[tag]['subpath'] = layers[tag]['name'] + '.png'



        tag = 'drive'
        if tag in layers:
            layers[tag]['name'] = tag;
            layers[tag]['node_sizes'] = 0.;
            layers[tag]['node_colors'] = [0,0,0,0];
            layers[tag]['node_edge_colors'] = [0,0,0,0];
            layers[tag]['subpath'] = layers[tag]['name'] + '.png' 



        end_time = time.time()
        print('time to create layers:',end_time-start_time)


        # LAYERS = generateLayers(GRAPHS,NODES,SIZES,DELIVERY,WORLD,params,use_all_trips=True);
        # # tag = 'base'
        # # plotLayer(GRAPHS,layers[tag])
        # folder = 'figs/interact/layers/'
        # tags = ['drive_trips/','walk_trips/','gtfs_trips/','ondemand_trips/']
        # for tag in tags:
        #     path = folder + tag
        #     if not os.path.exists(path):
        #         os.mkdir(path)

        # path = folder + 'groups/';
        # if not os.path.exists(path): os.mkdir(path)

        # path = folder + 'runs/';
        # if not os.path.exists(path): os.mkdir(path)
        # plotLAYERS(GRAPHS,folder,LAYERS);
        self.layers = layers
        # return layers

    def plotLAYERS(self,folder='DASH/dash',overwrite=False,verbose=True):
        print('starting layers...')
        GRAPHS = self.GRAPHS
        layers = self.layers;
        self.dash_folder = folder; #'DASH/'+folder


        write_layers = overwrite;
        if not(os.path.isdir(self.dash_folder)):
            os.mkdir(self.dash_folder)
            write_layers = True;
        # if not(os.path.isdir(self.dash_folder+'/groups')):
        #     os.mkdir(self.dash_folder+'/groups')
        if not(os.path.isdir(self.dash_folder+'/runs')):
            os.mkdir(self.dash_folder+'/runs')
        if not(os.path.isdir(self.dash_folder+'/groups')):
            os.mkdir(self.dash_folder+'/groups')


        if not(os.path.isdir(self.dash_folder+'/walk_trips')):
            os.mkdir(self.dash_folder+'/walk_trips');
        if not(os.path.isdir(self.dash_folder+'/ondemand_trips')):
            os.mkdir(self.dash_folder+'/ondemand_trips');
        if not(os.path.isdir(self.dash_folder+'/drive_trips')):
            os.mkdir(self.dash_folder+'/drive_trips');
        if not(os.path.isdir(self.dash_folder+'/gtfs_trips')):
            os.mkdir(self.dash_folder+'/gtfs_trips');


        # os.mkdir(self.dash_folder+'/drive_trips')
        # os.mkdir(self.dash_folder+'/walk_trips')
        # os.mkdir(self.dash_folder+'/ondemand_trips')                

        if write_layers: 
            for i,layer in enumerate(layers):
                if np.mod(i,20)==0: print('PLOTTING LAYER',i,'...');
                if verbose:  print('drawing layer: ',layer)
                LAYER = layers[layer]
                self.plotLayer(LAYER); #folder=self.dash_folder);
        print('finished.')


    def plotLayer(self,data): #,folder=''):
        GRAPHS = self.GRAPHS
        GRAPH = GRAPHS[data['graph']];
        folder = self.dash_folder + '/'

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
        # if data['name'] == 'walk':
        #     print(NODE_SIZES[:10])

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




# class TRACE:
#     def __init__(self,params = {}):
#         self.image = None;
#         self.name = None
#         self.x = None
#         self.y = None
#         self.sliders = {};#'':{},
#         self.loc = None;
#         self.dataind = None
#         if 'name' in params: self.name = params['name'];
#         if 'image' in params: self.image = params['image'];
#         if 'x' in params: self.x = params['x'];
#         if 'y' in params: self.y = params['y'];
#         if 'loc' in params: self.loc = params['loc']
#         if 'dataind' in params: self.dataind = params['dataind'];
#         if 'sliders' in params: self.sliders = params['sliders'];

#     def show(self,fig):
#         fig.add_trace(go.Bar(x=self.x,y=self.y),row=loc[0],col=loc[1]);
#         #     fig.add_trace(go.Bar(x=counts2,y=costs2,width=0.5,base ='overlay',marker = {'color' :color,'opacity':0.5}),inds2[0],inds2[1])





    def generate_graph_presets(self): #lims = [],other_edges = False):    

        lims = [];
        other_edges = False;
        fileName = 'current.pdf'

        shows = {'drive': True,
                 'walk':False,
                 'transit':False,
                 'ondemand': False,
                 'direct':False,
                 'shuttle':False,
                 'ondemand_indiv':False,
                 'lines':True,
                 'gtfs':False,
                 'source':False,
                 'target':False,
                 'legend':True,
                 'base':True}
        maxwids = {'drive': 10.,'walk':6,'transit':10,'lines':4,'gtfs':10,'ondemand':2,'direct':1,
                    'ondemand_indiv':10,'base':4.}

        colors = {'shuttle':[1,0.5,0.5]}

        # mxpop1 = 1.
        mxpops = {'drive': 1.,'walk':1,'transit':1,'lines':1,'gtfs':1,'ondemand':1}

        # params = generate_graph_presets(fileName,shows,WORLD,maxwids,mxpops,other_edges = True)
        # params['SIZES'] = SIZES;
        # params['colors']['shuttle'] = [0.,0.,1.]
        # params['set_alphas'] = {'direct':0.6,'shuttle':0.6}
        # params['set_wids'] = {'direct':4,'shuttle':4}
                                 
        # start_time = time.time()
        cmap = plt.get_cmap('autumn')
    # plot_multimode(GRAPHS,NODES,DELIVERY,WORLD,params);


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



        # if other_edges == True: 
        #     params['other_edges'] = {'grp1':{'edges':WORLD['drive']['base_edge_masses'],
        #                                      'thresh': threshs['drive'],
        #                                      'maxwid':maxwids['drive'],
        #                                      'color':[0.9,0.1,0.1],
        #                                      'maxalpha':0.3}}; #colors['drive']}}
        return params        




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

        # count = 0;
        # file_created=False;

        # month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',
        #                5:'May',6:'Jun',7:'Jul',8:'Aug',
        #                9:'Sept',10:'Oct',11:'Nov',12:'Dec'}
 
        # day = str(datetime.now().day)
        # month = month_names[datetime.now().month]
        # count = 0;
        # tag_to_add = '';
        # free_slot_found = False;
        # while  free_slot_found == False:
        #     print(count)
        #     possible_path = filename + tag_to_add; #'run' +str(count)+'_'+month+day
        #     if os.path.exists(possible_path):
        #         count = count + 1;
        #         tag_to_add = '_'+str(count+1);
        #     else:
        #         free_slot_found = True;
        #         output_path = possible_path;

    def saveOutputs(self,filename,typ='pickle',overwrite=False):

        output_path = filename;
        if not(os.path.isdir(filename)):
            os.mkdir(filename)
            overwrite=True;

        if overwrite:
            print('overwriting...')
            if typ=='parquet':
                for mode in self.modes:
                    self.OUTPUTS['by_mode'][mode].to_parquet(output_path+'/mode_' + mode + '.parquet')
                self.OUTPUTS['ondemand'].to_parquet(output_path+'/ondemand_driver_runs'+'.parquet')
            elif typ=='pickle':
                for mode in self.modes:
                    handle = open(output_path+'/mode_' + mode + '.pickle','wb')
                    pickle.dump(self.OUTPUTS['by_mode'][mode],handle);
                handle = open(output_path+'/ondemand_driver_runs'+'.pickle','wb')
                pickle.dump(self.OUTPUTS['ondemand'],handle)

    def printOutputs(self,dfs_to_show=['drive','walk','ondemand','gtfs','groups'],row_count =5, show_active=False):
        

        DFS = {}
        for mode in self.modes:
            DF = self.OUTPUTS['by_mode'][mode]
            if show_active: DF = DF[DF['active']==True]
            print(mode, 'statistics ... printing dataframe of length', len(DF))
            DFS[mode] = DF.copy()

        if 'groups' in dfs_to_show:
            DF = self.OUTPUTS['ondemand']
            # if show_active: DF = DF[DF['active']==True]
            print('ONDEMAND Statistics...printing dataframe of length',len(DF))
            print(DF.head(row_count)); 
            print(''); print(''); print('');

        for mode in self.modes:
            if mode in dfs_to_show:
                DF = DFS[mode]
                print(DF.head(row_count))
                print(''); print(''); print('')
        

    # def checkOutputs(self,mode):
    #     return self.OUTPUTS['by_mode'][mode]

    def generateOutputs(self,randomize=[]):


        maxvalues = {'distance':10000000,'travel_time':100000,'money':10000000,'switches':10000000}
        maxvalues_bymode = {'drive':{'distance':10000000,'travel_time':100000,'money':10000000,'switches':10000000},
                            'walk':{'distance':10000000,'travel_time':100000,'money':10000000,'switches':10000000},
                            'ondemand':{'distance':10000000,'travel_time':100000,'money':10000000,'switches':10000000},
                            'gtfs':{'distance':10000000,'travel_time':100000,'money':10000000,'switches':10000000}}
        ##### INITIALIZING 
        self.OUTPUTS = {};
        self.OUTPUTS['by_mode'] = {};
        for _,mode in enumerate(self.modes):
            colheads = []
            colheads = colheads + ['seg_id','mode','trip_ids','active']
            colheads = colheads + ['start_node','end_node','start_loc','end_loc'];
            colheads = colheads + ['people'];
            colheads = colheads + ['distance','travel_time','money','switches']
            colheads = colheads + ['uncongested_distance','uncongested_travel_time']
            if mode == 'ondemand':
                colheads = colheads + ['group_id','run_id'];
                colheads = colheads + ['pickup_time_start','pickup_time_end','dropoff_time_start','dropoff_time_end']
                colheads = colheads + ['pickup_time_scheduled','dropoff_time_scheduled']
                colheads = colheads + ['pickup_time','dropoff_time']
                colheads = colheads + ['num_passengers']

            if mode == 'gtfs':
                colheads = colheads + ['line_id','bus_trip_id'];
            self.OUTPUTS['by_mode'][mode] = pd.DataFrame({col:[] for col in colheads},index = [])
        
        for _,mode in enumerate(self.modes):
            NETWORK = self.NETWORKS[mode]
            GRAPH = self.GRAPHS[mode];
            active_segs = NETWORK.active_segs;
            all_segs = list(NETWORK.segs);
            SEGS = NETWORK.segs

            for seg in SEGS:

                SEG = SEGS[seg];



                DATA = {};
                DATA['seg_id'] = [SEG.seg_id];
                DATA['trip_ids'] = [SEG.trip_ids];
                DATA['mode'] = mode;
                if hasattr(SEG,'people'): DATA['people'] = [SEG.people]
                else: DATA['people'] = [None];
                    
                DATA['start_node'] = [seg[0]];
                DATA['end_node'] = [seg[1]];

                start_loc = (GRAPH.nodes[seg[0]]['x'],GRAPH.nodes[seg[0]]['y'])
                end_loc = (GRAPH.nodes[seg[1]]['x'],GRAPH.nodes[seg[1]]['y'])

                DATA['start_loc'] = [start_loc];
                DATA['end_loc'] = [end_loc];


                if seg in active_segs: DATA['active'] = True;
                else: DATA['active'] = False;


                # if 'current_dist' in SEG.costs:
                DATA['distance'] = [SEG.costs['current_dist']]
                # if 'current_conven' in SEG.costs:
                # DATA['current_conven'] = [SEG.costs['current_conven']]    
                DATA['travel_time'] = [SEG.costs['current_time']]
                DATA['money'] = [SEG.costs['current_money']]
                DATA['switches'] = [SEG.costs['current_switches']]
                # NEWDATA['conven'] = [SEG.costs['current_conven']]

                    
                    
                if hasattr(SEG,'uncongested'):
                    if 'costs' in SEG.uncongested:
                        if 'dist' in SEG.uncongested['costs']:
                            DATA['uncongested_distance'] = [SEG.uncongested['costs']['dist']]
                        DATA['uncongested_travel_time'] = [SEG.uncongested['costs']['time']]
                else:
                    if 'dist' in SEG.uncongested['costs']:
                        DATA['uncongested_distance'] = [DATA['dist']] 
                    DATA['uncongested_travel_time'] = [DATA['time']]
                
                if mode == 'gtfs':
                    if hasattr(SEG,'line_id'): DATA['line_id'] = SEG.line_id
                    if hasattr(SEG,'bus_trip_id'): DATA['bus_trip_id'] = SEG.bus_trip_id

                if mode == 'ondemand':
                    # if False:
                    DATA['run_id'] = [SEG.run_id];
                    DATA['group_id'] = [SEG.group];

                    if hasattr(SEG,'pickup_time_scheduled'):
                        pickup_time_scheduled = SEG.pickup_time_scheduled;
                        dropoff_time_scheduled = SEG.dropoff_time_scheduled;
                        num_passengers = SEG.num_passengers;
                    else:
                        pickup_time_scheduled = SEG.pickup_time_window_start;
                        dropoff_time_scheduled = SEG.dropoff_time_window_start;
                        num_passengers = 1;

                    DATA['pickup_time_scheduled'] = [pickup_time_scheduled];
                    DATA['dropoff_time_scheduled'] = [dropoff_time_scheduled];
                    DATA['pickup_time'] = [pickup_time_scheduled];
                    DATA['dropoff_time'] = [dropoff_time_scheduled];
                    DATA['num_passengers'] = [num_passengers];

                    DATA['group_id'] = [SEG.group];
                    DATA['booking_id'] = [SEG.booking_id];
                    DATA['pickup_time_start'] = [SEG.pickup_time_window_start];
                    DATA['pickup_time_end'] = [SEG.pickup_time_window_end];
                    DATA['dropoff_time_start'] = [SEG.dropoff_time_window_start];
                    DATA['dropoff_time_end'] = [SEG.dropoff_time_window_end];
                self.OUTPUTS['by_mode'][mode] = pd.concat([self.OUTPUTS['by_mode'][mode],pd.DataFrame(DATA)],ignore_index=True);

                for factor in maxvalues:
                    maxvalue = maxvalues_bymode[mode][factor];
                    mask = self.OUTPUTS['by_mode'][mode][factor] > maxvalue;
                    self.OUTPUTS['by_mode'][mode][mask] = 0;


            for factor in randomize:
                length = len(self.OUTPUTS['by_mode'][mode]);
                self.OUTPUTS['by_mode'][mode][factor] = np.random.rand(length)


            
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################
        ###########################################################################################


        tag = 'ondemand'
        colheads = [];
        colheads = colheads + ['group_id','run_id'];
        colheads = colheads + ['VMT','PMT','VMT/PMT'];
        colheads = colheads + ['VTT','PTT','VTT/PTT'];
        # colheads = colheads + ['distance','travel_time','money']
        colheads = colheads + ['total_passengers','max_num_passengers','ave_num_passengers'];
        self.OUTPUTS[tag] = pd.DataFrame({col:[] for col in colheads},index = [])


        GROUPS = self.ONDEMAND.groups;

        for group in GROUPS:
            GROUP = GROUPS[group];
            for run in GROUP.runs:
                RUN = GROUP.runs[run];
                DATA = {};
                DATA['group_id'] = [GROUP.group]
                DATA['run_id'] = [RUN.run_id]
                DATA['VMT'] = [RUN.current_VMT]
                DATA['PMT'] = [RUN.current_PMT]
                DATA['VMT/PMT'] = [RUN.current_VMTbyPMT]
                DATA['VTT'] = [RUN.current_VTT];
                DATA['PTT'] = [RUN.current_PTT];
                DATA['VTT/PTT'] = [RUN.current_VTTbyPTT];

                # DATA['distance'] = [RUN.costs['current_dist']];
                # DATA['travel_time'] = [RUN.costs['current_time']];
                # DATA['money'] = [RUN.costs['current_money']];

                DATA['total_passengers'] = [RUN.total_passengers];
                DATA['max_num_passengers'] = [RUN.max_num_passengers];
                DATA['ave_num_passengers'] = [RUN.ave_num_passengers];
                self.OUTPUTS[tag]= pd.concat([self.OUTPUTS[tag],pd.DataFrame(DATA)],ignore_index=True);


            # print(GROUP.driver_runs)
            # print('')

            # print(list(vars(GROUP)))

            # asdfasdfs



        # if False: 
        #     for _,mode in enumerate(self.modes):
        #         DF = pd.DataFrame(COLHEADERS,index=[])
        #         active_segs = self.NETWORKS[mode].active_segs;
        #         all_segs = list(self.NETWORKS[mode].segs);
        #         SEGS = self.NETWORKS[mode].segs
        #         for seg in SEGS:
        #             SEG = SEGS[seg];

        #             NEWDATA = {};
        #             if hasattr(SEG,'people'): NEWDATA['people'] = SEG['people']
        #             else: NEWDATA['people'] = [None];
        #             NEWDATA['start_node'] = [seg[0]];
        #             NEWDATA['end_node'] = [seg[1]];
        #             if seg in active_segs: NEWDATA['active'] = True;
        #             else: NEWDATA['active'] = False;

        #             NEWDATA['mode'] = [mode]
        #             NEWDATA['time'] = [SEG.costs['current_time']]
        #             NEWDATA['money'] = [SEG.costs['current_money']]
        #             NEWDATA['switches'] = [SEG.costs['current_switches']]
        #             # NEWDATA['conven'] = [SEG.costs['current_conven']]
        #             if hasattr(SEG,'uncongested'):
        #                 if 'costs' in SEG.uncongested:
        #                     NEWDATA['uncongested_time'] = [SEG.uncongested['costs']['time'][-1]]
        #                 # NEWDATA['uncongested_money'] = [SEG.uncongested['costs']['money']]
        #             else: 
        #                 NEWDATA['uncongested_time'] = NEWDATA['time']
        #                 # NEWDATA['uncongested_money'] = NEWDATA['money']
        #             if mode == 'ondemand':
        #                 if hasattr(SEG,'group'): NEWDATA['group'] = [SEG.group];
        #                 if hasattr(SEG,'run_id'): NEWDATA['runid'] = [SEG.run_id];
        #                 if hasattr(SEG,'num_passengers'): NEWDATA['num_passengers'] = [SEG.num_passengers];
        #                 if hasattr(SEG,'pickup_time_window_start'): NEWDATA['pickup_time'] = [SEG.pickup_time_window_start]
        #                 if hasattr(SEG,'dropoff_time_window_start'): NEWDATA['dropoff_time'] = [SEG.dropoff_time_window_start]
        #                 NEWDATA['num_other_passengers'] = 0; 



        #             NEWDF = pd.DataFrame(NEWDATA)
        #             DF = pd.concat([DF,NEWDF], ignore_index = True)            
        #     self.OUTPUTS['PEOPLE'] = DF.copy();


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################



    def generateOutputs2(self):

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
                    NEWDATA['time0'] = [NEWDATA['time']]
                    NEWDATA['money0'] = [NEWDATA['money']]
                
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

        # folder = 'runs/'
        # filename = 'data_version1.obj'
        # folder = 'runs/'
        # filename = 'data_version1.obj'
        # # fileObj = open(folder+filename, 'wb')
        # # pickle.dump(BLAH,fileObj)
        # # fileObj.close()

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


        # folder = 'runs/'
        # filename = 'data_version1.obj'
        # folder = 'runs/'
        # filename = 'data_version1.obj'
        # # fileObj = open(folder+filename, 'wb')
        # # pickle.dump(BLAH,fileObj)
        # # fileObj.close()        

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


    def generateDashboard(self):
        pass
    def printDashboard(self):
        pass



# class STATICPLOTS:
#     def __init__(self):

# %load_ext autoreload
# %autoreload 2
# from multimodal_functions import * 

# compute_UncongestedEdgeCosts(WORLD,GRAPHS)
# compute_UncongestedTripCosts(WORLD,GRAPHS)

# # datanames = ['full','2regions','4regions','small','tiny'];
# #datanames = ['large1']; #,'medium1','small1','tiny1'];
# datanames = ['regions2','regions4','regions7','tiny1']
# filenames = {name:name+'.obj' for name in datanames}
# folder = 'runs/'

# print('COMPUTING DATA FOR DASHBOARD...')
# # DATA = computeData(WORLD,GRAPHS,DELIVERY)
# DATAS = loadDataRuns(folder,filenames,GRAPHS);

# # mode = 'gtfs';
# # subplot_tags = datanames;
# data_tags = ['regions2','regions4','regions7','tiny1']
# use_active = True; 

# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# import skimage.io as sio
# from PIL import Image, ImageSequence
# import numpy as np
# import os

# from plotly.subplots import make_subplots
# import numpy as np
# from numpy import pi, sin, cos

# modes = ['drive','walk','gtfs','ondemand']

# high_colors = {'drive':[1,0,0,1],'walk':[1,1,0,1],'gtfs':[1,0.5,0.,1],'ondemand':[0,0,1,1]}
# colors = {'drive':[1,0,0,1],'walk':[1,1,0,1],'gtfs':[1,0.5,0.,1],'ondemand':[0,0,1,1],
#           'groups':[[0,0,1,1],[0.5,0,1,1],[0.5,0,1,1],[0.5,0,1,1],[0.5,0,1,1],[0.5,0,1,1],[0.5,0,1,1],[0.5,0,1,1]],
#           'time':[1,0,0,1],'money':[0,0.8,0.,1.],'conven':[0,0,1,1],'switches':[0,0,0,1]}
# row_inds = {'drive':1,'walk':2,'gtfs':3,'ondemand':4};



# for mode in modes:
#     dims = [1000,400];
#     if mode == 'ondemand':
#         dims = [1000,400];
#         SUBPLOTS = [{'tag':'time','title':"Travel Time",'inds':(1,1)},
#                     {'tag':'time/dtime','title':"Time/(Direct Time)",'inds':(1,2)},
#                     {'tag':'money','title':'Monetary Cost','inds':(1,3)},
#                     {'tag':'overlap','title':'Passenger Overlap','inds':(1,4)}]
#                     # {'tag':'VDM','title':'VDM','inds':(1,5)},
#                     # {'tag':'VMT/PMT','title':'VMT/PMT','inds':(1,6)}]
#     elif mode == 'gtfs':
#         dims = [1000,400];
#         SUBPLOTS = [{'tag':'time','title':"Travel Time",'inds':(1,1)},
#                     {'tag':'money','title':'Monetary Cost','inds':(1,2)},
#                     {'tag':'transfers','title':'Bus Transfers','inds':(1,3)}]
#     elif mode == 'walk':
#         dims = [400,400];
#         SUBPLOTS = [{'tag':'time','title':"Travel Time",'inds':(1,1)}]
#     else:
#         dims = [1000,400];
#         SUBPLOTS = [{'tag':'time','title':"Travel Time",'inds':(1,1)},
#                     {'tag':'time/dtime','title':"Time/(Direct Time)",'inds':(1,2)},
#                     {'tag':'money','title':'Monetary Cost','inds':(1,3)}]
        
#     subplot_titles = [SUBPLOT['title'] for SUBPLOT in SUBPLOTS];
    
#     num_rows = 1; num_cols = len(subplot_titles); 
#     bot_row_specs = [[{'t':0.01,'b':0.01,'r':0.02,'l':0.02} for _ in range(num_cols)] for _ in range(num_rows)];
#     subplots_shape =  bot_row_specs;
    
#     row_heights = list(0.1*np.ones(len(bot_row_specs))); col_widths = list(0.3*np.ones(len(bot_row_specs[0])))
    
    
    
#     fig = make_subplots(rows=len(subplots_shape), cols = len(subplots_shape[0]),
#                         column_widths = col_widths,row_heights = row_heights,
#                         horizontal_spacing=0.0,vertical_spacing=0.0,
#                         subplot_titles=subplot_titles,
#                         specs=subplots_shape,print_grid=True);
    
    
    
#     fig.update_layout(width=dims[0],height=dims[1],boxmode='group'); #, xaxis_visible=False, yaxis_visible=False)
#     color = px.colors.label_rgb(list(255*np.array(colors[mode])))
#     edgecolor = 'rgb(0,0,0)';    
    
#     for SUBPLOT in SUBPLOTS:
#         title = SUBPLOT['title'];
#         tag = SUBPLOT['tag'];
#         inds = SUBPLOT['inds'];
#         for i,name in enumerate(data_tags): #enumerate(DATAS):
#             DATA = DATAS[name]
#             DF0 = DATA[mode]['DF'];
#             opac = 0.7;
            
#             if tag == 'time':
#                 DF = DF0.copy();
#                 col = 'time';
#                 if use_active: DF = DF[DF['active']==True];
#                 DF = DF[DF[col]<=1000000];
#                 VALS = DF[col]; series_name = name; boxpoints = 'all'
#                 fig.add_traces(go.Box(y=VALS,fillcolor=color,opacity=opac,marker_color=edgecolor,boxpoints=boxpoints,width=0.4,line={'width':1},name=series_name),inds[0],inds[1]);
#             ###########
    
#             if tag == 'time/dtime':
#                 DF = DF0.copy();
#                 col = 'time';
#                 if use_active: DF = DF[DF['active']==True];
#                 DF = DF[DF[col]<=1000000];
#                 VALS = DF['time']/DF['time0']; series_name = name; boxpoints = 'all'
#                 fig.add_traces(go.Box(y=VALS,fillcolor=color,opacity=opac,marker_color=edgecolor,boxpoints=boxpoints,width=0.4,line={'width':1},name=series_name),inds[0],inds[1]);
#             ###########
#             if tag == 'money':
#                 DF = DF0.copy();
#                 col = 'money';
#                 if use_active: DF = DF[DF['active']==True];
#                 DF = DF[DF[col]<=1000000];
#                 VALS = DF['money']; series_name = name; boxpoints = 'all'
#                 fig.add_traces(go.Box(y=VALS,fillcolor=color,opacity=opac,marker_color=edgecolor,boxpoints=boxpoints,width=0.4,line={'width':1},name=series_name),inds[0],inds[1]);
#             ###########
#             if tag == 'overlap':
#                 DF = DF0.copy();
#                 col = 'num_passengers';
#                 if use_active: DF = DF[DF['active']==True];
#                 DF = DF[DF[col]<=1000000];
#                 VALS = DF[col]; series_name = name; boxpoints = 'all'
#                 fig.add_traces(go.Box(y=VALS,fillcolor=color,opacity=opac,marker_color=edgecolor,boxpoints=boxpoints,width=0.4,line={'width':1},name=series_name),inds[0],inds[1]);
#             ###########
    
#             if tag == 'transfers':
#                 DF = DF0.copy();
#                 col = 'switches';
#                 if use_active: DF = DF[DF['active']==True];
#                 DF = DF[DF[col]<=1000000];
#                 DF = DF[DF[col]>=0.0];
#                 VALS = DF[col]; series_name = name; boxpoints = 'all'
#                 fig.add_traces(go.Box(y=VALS,fillcolor=color,opacity=opac,marker_color=edgecolor,boxpoints=boxpoints,width=0.4,line={'width':1},name=series_name),inds[0],inds[1]);
#             ###########

    
    
#     fig.update_layout(showlegend=False)
#     fig.show()

#     fig.write_image('./figs/static/' + mode+"_static_data.png"); #,engine="kaleido")
#     # plotly.image.save_as(fig, filename='file.png')





# class TRACE:
#     def __init__(self):
#         self.image = None;
#         self.tags = None;
#         self.datas = {'data1':{'tag':None,'x':None,'y':None},
#                       'data2':{'tag':None,'x':None,'y':None}}
#         self.datax = None;
#         self.datay = None;
#         self.dataind = None;
#         self.DF = None;
#         self.sliders 


# class BUTTON:
#     def __init__(self):
#         pass

# class SLIDER:
#     def __init__(self):
        
#         self.x = 0;
#         self.y = 0;
#         self.length = None;
#         self.xanchor = "left"
#         self.yanchor = "top"
#         self.active = 0;
#         self.currentvalue = {};
#         self.pad = {'t':0,'b':0,'r':0,'l':0}
#         self.steps = []
#         self.datainds = []
#         self.num_steps = 0;
#         # for i in range(self.num_steps):
#         #     step = dict(method="update",label='',args=[{'visible':[]},{},[]]);
#         #     self.steps.append(step)

#     def addStep(self,sliderind,dataind):

#         if self.num_steps < sliderind + 1:
#             new_num_steps = sliderind + 1;
#             diff_size = new_num_steps - self.num_steps;
#             self.datainds = self.datainds + [None * diff_size];
#             for i in range(new_num_steps):
#                 if i < self.num_steps:
#                     STEP = self.steps[i]
#                     STEP['args'][0]['visible'] = STEP['args'][0]['visible'] + [False * diff_size];
#                     STEP['args'][2] = STEP['args'][2] + [None * diff_size];
#                 else:
#                     STEP = dict(method="update",label='',args=[{'visible':[False*new_num_steps]},{},[None*new_num_steps]]);
#                     self.steps.append(STEP)

#         self.num_steps = new_num_steps:
#         for i in range(self.num_steps):
#             STEP = self.steps[i]
#             STEP['args'][0]['visible'][sliderind] = True;
#             STEP['args'][2][sliderind] = dataind;
#         self.datainds[sliderind] = dataind            

#     def write(self):
#         out = dict(x=self.x,y=self.y,len=self.length,xanchor=self.xanchor,yanchor=self.yanchor,active=self.active,currentvalue={},pad=self.pad,steps=self.steps)
#         return out

#         pass

#     len2 = len(datainds2);
#     datainds = datainds1 + datainds2; 
#     tag2 = group + '_runs';
#     loc = CTRL_LOCS[tag2]
#     if len(loc)>=3: length = loc[2];
#     else: length = 0.2;
#     SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="top",active=0,currentvalue={},pad={"t": 0},steps=steps))


# for tag in groups: #['group0','group1']: 
#     ### NEW
#     steps = []    
#     datainds1 = []; datainds2 = [];
#     TAGS = DATAINDS2['imgs']['groups'][tag];
#     for tag2 in TAGS:
#         try: datainds1.append(DATAINDS2['imgs']['groups'][tag][tag2])
#         except: pass
#         try: datainds2.append(DATAINDS2['bars']['groups'][tag][tag2]);
#         except: pass
#     datainds = datainds1 + datainds2
#     len1 = len(datainds1); len2 = len(datainds2);

#     if len2>0:
#         print('len1: ',len1)
#         print('len2: ',len2)
#         for i in range(len2):
#             step = dict(method="update",label='',args=[{"visible": [False] * (len1+len2)},{},datainds])
#             step["args"][0]["visible"][i] = True  
#             step["args"][0]["visible"][len1+i] = True  
#             steps.append(step)
#         loc = CTRL_LOCS[tag]
#         if len(loc)>=3: length = loc[2];
#         else: length = 0.2;
#         SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="top",active=0,currentvalue={},pad={"t": 0},steps=steps))


# for tag0 in ['walk','drive','gtfs']: #,'ondemand_trips']:
#     # NEW
    
#     datainds1 = []; datainds2 = [];
#     dataindsn = {};
#     tag = tag0 + '_trips'
#     TAGS = DATAINDS2['imgs'][tag];
#     lenn = []
#     for tag2 in TAGS:
#         try: datainds1.append(DATAINDS2['imgs'][tag0 + '_trips'][tag2])
#         except: pass
#     for factor in use_factors:
#         dataindsn[factor] = []
#         for tag2 in TAGS:
#             dataindsn[factor].append(DATAINDS2['bars'][tag0 + '_trips_' + factor][tag2]);
#         lenn.append(len(dataindsn[factor]));

#     datainds = datainds1
#     for factor in use_factors:
#         datainds = datainds + dataindsn[factor]
#     len1 = len(datainds1);  len2 = len(datainds2);

#     # if lenn[0]>0
#     # print('1: ',datainds1)
#     # print('time: ' ,dataindsn['time'])
#     # print('money: ',dataindsn['money'])
#     # print(datainds)

#     if len1 > 0: 
#         steps = []
#         for i in range(len1):
#             step = dict(method="update",label='',args=[{"visible": [False] * (len1+np.sum(lenn))},{},datainds])
#             step["args"][0]["visible"][i] = True  
#             for k,factor in enumerate(use_factors):
#                 step["args"][0]["visible"][len1 + int(np.sum(lenn[:k])) + i] = True  
#             steps.append(step)
            
#         loc = CTRL_LOCS[tag]
#         if len(loc)>=3: length = loc[2];
#         else: length = 0.2;
#         SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="top",
#                             active=0,currentvalue={},pad={"t": 0},steps=steps))    


# print('building buttons...')
# BUTTONS = [];
# for tag in ['lines','source','target','drive','gtfs','walk','ondemand']: 
#     loc = CTRL_LOCS[tag];
#     # print(DATAINDS[tag])
#     BUTTONS.append(dict(
#             buttons=list([
#                 dict(args=[{"visible":True},{},[DATAINDS2['imgs'][tag]]],label=tag + " ON" ,method="update"),
#                 dict(args=[{"visible":False},{},[DATAINDS2['imgs'][tag]]],label=tag + " OFF",method="update")]),
#             # type = "buttons",
#             direction="down",
#             pad={"r": 0, "t": 0},
#             showactive=True,x=loc[0],xanchor="left",y=loc[1],yanchor="top"));



# print('adding controls...')






# fig.update_layout(showlegend=False)

# # fig.update_layout(xaxis_visible=False, yaxis_visible=False)

# # fig.update_layout(autosize=True,
# #     height=600,
# #     width=500,
# #     margin=dict(l=0,r=0,t=20,b=0),
# #     grid = {'rows': 2, 'columns': 1, 'pattern': "independent"})

# # fig.update_layout(column_widths = [400,300],row_heights = [800,300])



# print('showing...')
# fig.show()

# print('done showing.')

# import plotly.io as pio
# pio.write_html(fig, file='case3.html', auto_open=True)


# class SUBPLOT: 
#     def __init__(self,params={}):





#         self.subplot = None;
#         self.size = (100,100)
#         self.pads = (2,2)
#         self.grid_loc = ()
#         self.grid_exts = (1,1)
#         self.margins = (2,2)
#         if 'subplot' in params: self.subplot = params['subplot'];
#         if 'size' in params: self.size = params['size'];
#         if 'grid_loc' in params: self.grid_loc = params['grid_loc'];
#         if 'grid_exts' in params: self.grid_exts = params['grid_exts'];
#         if 'pads' in params: self.pads = params['pads'];
#         if 'margins' in params: self.margins = params['margins'];


#         sefl

#     def sort(self):
#         pass

# class SUBCTRL:
#     def __init__(self):
#         pass

# class SUBGROUP:
#     def __init__(self):
#         self.PLOTS = None
#         self.CTRLS = None
#         pass





# num_groups = len(groups);   

# num_fig_rows = 3;
# num_fig_cols = num_groups


# aspect_ratio = 1540./1400.;
# fac = 1.3;
# height = fac*800; width = fac*1000; fig_col_width_perc = 0.5;
# bot_row_height = 200; #height - fig_row_height;
# fig_col_width = fig_col_width_perc * width;
# fig_row_height = (1./aspect_ratio) * fig_col_width;


# right_col_width = width - fig_col_width;
# group_col_width = fig_col_width/num_groups;

# # main_col_wids = [fig_col_width/3,fig_col_width/3,fig_col_width/3,width/4]
# main_col_wids = [fig_col_width/num_fig_cols for _ in range(num_fig_cols)] + [width/4];
# main_row_heights = [fig_row_height/num_fig_rows for _ in range(num_fig_rows)] + [200,300]; #height-fig_row_height];

# # # column_widths = [main_col_wids[0]*0.7,main_col_wids[0]*0.3,
# # #                  main_col_wids[1]*0.7,main_col_wids[1]*0.3,
# # #                  main_col_wids[2]*0.7,main_col_wids[2]*0.3,
# # #                  main_col_wids[3]*0.7,main_col_wids[3]*0.3]



# column_widths = [];
# for i in range(len(main_col_wids)):
#     column_widths.append(main_col_wids[i]*0.7);
#     column_widths.append(main_col_wids[i]*0.3);

# row_heights = [];
# for i in range(len(main_row_heights)):
#     row_heights.append(main_row_heights[i]*0.5);
#     row_heights.append(main_row_heights[i]*0.5);
    
# # fig_col_width = np.sum(main_col_wids[:3])

# # width = fig_col_width + right_col_width;
# # height =  fig_row_height + bot_row_height;

# nrows1 = 4;
# ngroups = 2;
# row_height1 = fig_row_height/nrows1

# row1_loc = -0.05;
# row2_loc = -0.25;
# row3_loc = -0.45;
# row4_loc = -0.65;
# row5_loc = -0.85;

# col1_wid = 0.2;
# col2_wid = 0.2;
# col3_wid = 0.2;



# # padb1s = np.array([0.1,0.1,0.1]);
# # padl1s = np.array([0.04,0.04,0.04]);
# padb1s = np.array([0.05,0.05,0.05,0.05]);
# padl1s = np.array([0.02,0.02,0.02,0.02]);
# padb1s_px = height*padb1s
# padb2 = 0.05; padl2 = 0.02; padt2 = 0.02;
# padb3 = 0.01; padl3 = 0.02; padt3 = 0.02;


# INDS = {'fig':(1,1),
#         'bar_drive_time':(1,2*num_fig_cols+1),'box_drive_time':(1,2*num_fig_cols+2),
#         'bar_drive_money':(2,2*num_fig_cols+1),'box_drive_money':(2,2*num_fig_cols+2),
#         'bar_walk_time':(3,2*num_fig_cols+1),'box_walk_time':(3,2*num_fig_cols+2),
#         'bar_walk_money':(4,2*num_fig_cols+1),'box_walk_money':(4,2*num_fig_cols+2),
#         'bar_gtfs_time':(5,2*num_fig_cols+1),'box_gtfs_time':(5,2*num_fig_cols+2),
#         'bar_gtfs_money':(6,2*num_fig_cols+1),'box_gtfs_money':(6,2*num_fig_cols+2),
#         'bar_ondemand_time':(7,2*num_fig_cols+1),'box_ondemand_time':(7,2*num_fig_cols+2),
#         'bar_ondemand_money':(8,2*num_fig_cols+1),'box_ondemand_money':(8,2*num_fig_cols+2),
#         'bar_people_time':(2*num_fig_rows + 3,1),'box_people_time':(2*num_fig_rows + 3,2*num_fig_cols+1),
#         'bar_people_money':(2*num_fig_rows + 4,1),'box_people_money':(2*num_fig_rows + 4,2*num_fig_cols+1)}


# PADS = {'fig':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_drive_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_drive_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_drive_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_drive_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_walk_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_walk_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_walk_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_walk_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_gtfs_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_gtfs_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_gtfs_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_gtfs_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_ondemand_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_ondemand_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_ondemand_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_ondemand_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},

#         'bar_people_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_people_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'bar_people_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01},
#         'box_people_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01}
#        }


# button_locs = [0,1.1]; dx = 0.14; 
# CTRL_LOCS = {'lines': [button_locs[0],button_locs[1]],
#              'source':[button_locs[0]+1*dx,button_locs[1]],
#              'target':[button_locs[0]+2*dx,button_locs[1]],
#              'drive':[button_locs[0]+3*dx,button_locs[1]],
#              'walk':[button_locs[0]+4*dx,button_locs[1]],
#              'gtfs':[button_locs[0]+5*dx,button_locs[1]],
#              'ondemand':[button_locs[0]+6*dx,button_locs[1]],

#              'drive_trips':[0.,(1.-np.sum(row_heights[:num_groups*2+2])/height),right_col_width/width],
#              'walk_trips' :[0.,(1.-np.sum(row_heights[:num_groups*2+2])/height),right_col_width/width],
#              'gtfs_trips' :[0.,(1.-np.sum(row_heights[:num_groups*2+2])/height),right_col_width/width],
#              'ondemand_trips':[0.,(1.-np.sum(row_heights[:num_groups*2+2])/height),right_col_width/width],
             
#              # 'drive_trips':[fig_col_width/width,(bot_row_height+3.*row_height1)/height,right_col_width/width],
#              # 'walk_trips' :[fig_col_width/width,(bot_row_height+2.*row_height1)/height,right_col_width/width], 
#              # 'gtfs_trips' :[fig_col_width/width,(bot_row_height+row_height1)/height,right_col_width/width], 
#              # 'ondemand_trips' :[fig_col_width/width,bot_row_height/height,right_col_width/width], 
             
#              'group0':[0.*group_col_width/width,0.,group_col_width/width], 
#              'group1':[1.*group_col_width/width,0.,group_col_width/width],
#              'group2':[2.*group_col_width/width,0.,group_col_width/width],
#              'group3':[3.*group_col_width/width,0.,group_col_width/width],
#              'group4':[4.*group_col_width/width,0.,group_col_width/width],

#              'group0_runs':[0.*group_col_width/width,row1_loc,group_col_width/width], 
#              'group1_runs':[1.*group_col_width/width,row1_loc,group_col_width/width],
#              'group2_runs':[2.*group_col_width/width,row1_loc,group_col_width/width],
#              'group3_runs':[3.*group_col_width/width,row1_loc,group_col_width/width],
#              'group4_runs':[4.*group_col_width/width,row1_loc,group_col_width/width],

#               ########################################################################
             
#              'drive_costs':[0.*group_col_width/width,row2_loc,col2_wid],
#              'walk_costs':[1.*group_col_width/width,row2_loc,col2_wid],
#              'gtfs_costs':[2.*group_col_width/width,row2_loc,col2_wid],
#              'ondemand_costs':[3.*group_col_width/width,row2_loc,col2_wid],

#              'time_costs':[0.*group_col_width/width,row3_loc,col3_wid],
#              'money_costs':[1.*group_col_width/width,row3_loc,col3_wid],
#              'conven_costs':[2.*group_col_width/width,row3_loc,col3_wid],
#              'switches_costs':[3.*group_col_width/width,row3_loc,col3_wid]};


# # INDS = {'fig':(1,1),
# #         'bar_drive_time':(2*num_groups + 1,1),'box_drive_time':(2*num_groups + 1,2),
# #         'bar_drive_money':(2*num_groups + 2,1),'box_drive_money':(2*num_groups + 2,2),
# #         'bar_walk_time':(2*num_groups + 1,3),'box_walk_time':(2*num_groups + 1,4),
# #         'bar_walk_money':(2*num_groups + 2,3),'box_walk_money':(2*num_groups + 2,4),
# #         'bar_gtfs_time':(2*num_groups + 1,5),'box_gtfs_time':(2*num_groups + 1,6),
# #         'bar_gtfs_money':(2*num_groups + 2,5),'box_gtfs_money':(2*num_groups + 2,6),
# #         'bar_ondemand_time':(2*num_groups + 1,7),'box_ondemand_time':(2*num_groups + 1,8),
# #         'bar_ondemand_money':(2*num_groups + 2,7),'box_ondemand_money':(2*num_groups + 2,8),
# #         'bar_people_time':(2*num_groups + 3,1),'box_people_time':(2*num_groups + 3,7),
# #         'bar_people_money':(2*num_groups + 4,1),'box_people_money':(2*num_groups + 4,7)
# #        }


# NEWINDS = {'bar_'+group+'_time': (2*num_fig_rows+1,2*i+1) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}
# NEWINDS = {'bar_'+group+'_money': (2*num_fig_rows+2,2*i+1) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}
# NEWINDS = {'box_'+group+'_time':(2*num_fig_rows+1,2*i+2) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}
# NEWINDS = {'box_'+group+'_money':(2*num_fig_rows+2,2*i+1) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}


# # NEWINDS = {'bar_'+group+'_time': (2*i+1,7) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}
# # NEWINDS = {'bar_'+group+'_money': (2*i+2,7) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}
# # NEWINDS = {'box_'+group+'_time':(2*i+1,8) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}
# # NEWINDS = {'box_'+group+'_money':(2*i+2,8) for i,group in enumerate(groups)}; INDS = {**INDS,**NEWINDS}

# NEWPADS = {'bar_'+group+'_time': {'l':0.01,'r':0.01,'t':0.01,'b':0.01} for i,group in enumerate(groups)}; PADS = {**PADS,**NEWPADS}
# NEWPADS = {'bar_'+group+'_money': {'l':0.01,'r':0.01,'t':0.01,'b':0.01} for i,group in enumerate(groups)}; PADS = {**PADS,**NEWPADS}
# NEWPADS = {'box_'+group+'_time':{'l':0.01,'r':0.01,'t':0.01,'b':0.01} for i,group in enumerate(groups)}; PADS = {**PADS,**NEWPADS}
# NEWPADS = {'box_'+group+'_money':{'l':0.01,'r':0.01,'t':0.01,'b':0.01} for i,group in enumerate(groups)}; PADS = {**PADS,**NEWPADS}

        
        
# # top_row_specs = [[{"rowspan": num_groups, "colspan": 3,"type": "image"}]+[None for _ in range(5)] + 
# #            [{'b':padb1s[0],'l':padl1s[0]}], #"type": "bar"}],
# #            [None for _ in range(num_groups)] + [{'b':padb1s[1],'l':padl1s[1]}],#"type": "bar"}],
# #            [None for _ in range(num_groups)] + [{'b':padb1s[2],'l':padl1s[2]}],#"type": "bar"}],
# #            [None for _ in range(num_groups)] + [{'b':padb1s[3],'l':padl1s[3]}]]

# top_row_specs = [[{"rowspan": 2*num_fig_rows, "colspan": 2*num_fig_cols,"type": "image"}]+[None for _ in range(2*num_fig_cols-1)] + [{},{}]]
# top_row_specs = top_row_specs + [[None for _ in range(2*num_fig_cols)] + [{},{}] for _ in range(num_fig_rows*2-1)];

# bot_row_specs = [[{} for _ in range(2*num_fig_cols + 2)],
#                  [{} for _ in range(2*num_fig_cols + 2)],
#                  [{"rowspan": 1, "colspan": num_fig_cols}]+[None for _ in range(2*num_fig_cols-1)] + [{"rowspan": 1, "colspan": 2},None],
#                  [{"rowspan": 1, "colspan": num_fig_cols}]+[None for _ in range(2*num_fig_cols-1)] + [{"rowspan": 1, "colspan": 2},None]]
    
# subplot_specs = top_row_specs + bot_row_specs;

# for tag in INDS:
#     inds = INDS[tag];
#     if tag in PADS: subplot_specs[inds[0]-1][inds[1]-1] = {**subplot_specs[inds[0]-1][inds[1]-1],**PADS[tag]};

# subplots_locs = [];


# nrows2 = len(bot_row_specs);


# fig = make_subplots(
#     rows=len(subplot_specs), cols = len(subplot_specs[0]),    
#     column_widths = column_widths,
#     row_heights = row_heights,
#     horizontal_spacing=0.0,
#     vertical_spacing=0.0,
#     specs=subplot_specs,
#     print_grid=True); 
    
#     # {'b':padb2,'l':padl2,'t':padt2} for _ in range(num_groups)] + [{}],
#     #            [{'b':padb3,'l':padl3,'t':padt3} for _ in range(num_groups)] + [{}],
#     #            [{'b':padb3,'l':padl3,'t':padt3} for _ in range(num_groups)] + [{}],
#     #            [{'b':padb3,'l':padl3,'t':padt3} for _ in range(num_groups)] + [{}],
#     #            [{'b':padb3,'l':padl3,'t':padt3} for _ in range(num_groups)] + [{}],
#     #            [{'b':padb3,'l':padl3,'t':padt3} for _ in range(num_groups)] + [{}]];



#     def GENERATE(self):
#         pass
#     def PRINT(self):
#         pass

#     def generateLayers(self,params,folder,verbose = True):

#         if verbose: print('WRITING LAYERS FOR DASHBOARD:')
#         params = {};
#         params['bgcolor'] = [1,1,1,1]
#         LAYERS = generateLayers(self.GRAPHS,self.NODES,
#                                 self.SIZES,self.ONDEMAND,WORLD,params,use_all_trips=True);
#         # tag = 'base'
#         # plotLayer(GRAPHS,layers[tag])
#         folder = 'figs/interact/layers/'

        
#         path = folder
#         if not os.path.exists(path): os.mkdir(path)

#         tags = ['drive_trips/','walk_trips/','gtfs_trips/','ondemand_trips/']
#         for tag in tags:
#             path = folder + tag
#             if not os.path.exists(path):
#                 os.mkdir(path)
#         path = folder + 'groups/';
#         if not os.path.exists(path): os.mkdir(path)
#         path = folder + 'runs/';
#         if not os.path.exists(path): os.mkdir(path)

#         plotLAYERS(self.GRAPHS,folder,LAYERS);

#     def sortDataForDashboard(self):
#         print('SORTING DATA FOR DASHBOARD...')

#         compute_UncongestedEdgeCosts(WORLD,GRAPHS)
#         compute_UncongestedTripCosts(WORLD,GRAPHS)
#         OUTPUT1 = sortingData(WORLD,GRAPHS,DELIVERY,factor='time',use_all_trips=True);
#         tripsSorted = OUTPUT1['trips_sorted'];
#         convergeSorted = OUTPUT1['converge_sorted'];

#         OUTPUT2 = sortingData(WORLD,GRAPHS,DELIVERY,factor='money',use_all_trips=True);
#         tripsSorted2 = OUTPUT2['trips_sorted'];
#         convergeSorted2 = OUTPUT2['converge_sorted'];


#         parent_dir = './figs/interact/layers/'
#         # out_parent_dir = './figs/interact/gifs/'

#         images0 = []; images1 = [];
#         base_names = ['group0_delivery','group1_delivery']
#         num_figs = [7,7];


#         tags = ['base','lines','source','target',
#                 'drive','ondemand','gtfs','walk',
#                 'group0','group1',
#                 'drive_trips',
#                 'walk_trips',
#                 'gtfs_trips',
#                 'ondemand_trips']; 


#         groups = []
#         groups0 = list(tripsSorted['ondemand']['groups'])
#         for group in groups0:
#             if len(tripsSorted['ondemand']['groups'][group]['costs'])>0: groups.append(group)



#     def LOAD_DATA(self):

#         self.FILENAMES = {};


# FILENAMES = {tag:[] for tag in tags}
# IMGS = {tag:[] for tag in tags}
# IMGS2 = {tag:[] for tag in tags}
# FIGS = {tag:[] for tag in tags}

# #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES 
# #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES 
# #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES 

# DATAINDS2 = {'imgs':{},'bars':{},'box':{}};
# data_counter = 0;

# ## CREATING FILE NAMES 

# for i,tag in enumerate(['base','lines','source','target','drive','walk','gtfs','ondemand']):
#     FILENAMES[tag] = [parent_dir + tag + '.png']
#     # DATAINDS2['imgs'][tag] = data_counter; data_counter = data_counter + 1;

# # for i,tag in enumerate(['group0','group1']):
# #     FILENAMES[tag] = [parent_dir + base_names[i] + str(j) + '.png' for j in range(num_figs[i])]
    

# # for i,tag in enumerate(['walk_trips','gtfs_trips','ondemand_trips']): #(['drive_trips','walk_trips','gtfs_trips']):
# #     current_folder = parent_dir + tag + '/';
# #     files = os.listdir(current_folder); #[:5];
# #     FILENAMES[tag] = [current_folder + file for file in files]



# ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES 
# ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES 
# ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES ###### ADDING IMAGES 

# # for i,tag in enumerate(FILENAMES):
# #     # if len(FILENAMES[tag])>0:
# #     for j,filename in enumerate(FILENAMES[tag]):
# #         # IMGS[tag].append(Image.open(filename)); #imageio.v2.imread(filename))
# #         IMGS[tag].append(px.imshow(sio.imread(filename)));

# ####################################################################################
# ####################################################################################
# ####################################################################################
# ####################################################################################



# DATA_IMGS = [];
# DATAINDS = {};
# STARTINDS = {tag:[] for tag in tags};
# all_start_inds = [];
# # for tag in tags:
# #     STARTINDS[tag] = len(DATA_IMGS)
# #     all_start_inds.append(len(DATA_IMGS));    
# #     new_data_inds = [];
# #     for i,IMG in enumerate(IMGS[tag]):
# #         new_data_inds.append(len(DATA_IMGS));
# #         DATA_IMGS.append(IMG);
# #     DATAINDS[tag] = new_data_inds.copy()


# # for figm in figms:
# # for imgm in DATA_IMGS:
# #     fig.add_trace(imgm.data[0], 1, 1)
# #     data_counter = data_counter + 1;
# # x, y = np.meshgrid(np.linspace(-pi/2, pi/2, 100), np.linspace(-pi/2, pi/2, 100))




# FILENAMES2 = {tag:[] for tag in tags}
# IMGS = {tag:[] for tag in tags}
# IMGS2 = {tag:[] for tag in tags}
# FIGS = {tag:[] for tag in tags}

# #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES 
# #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES 
# #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES #### LOADING FILES 


# #### ADDING IMAGES...#### ADDING IMAGES...#### ADDING IMAGES...#### ADDING IMAGES...
# #### ADDING IMAGES...#### ADDING IMAGES...#### ADDING IMAGES...#### ADDING IMAGES...
# #### ADDING IMAGES...#### ADDING IMAGES...#### ADDING IMAGES...#### ADDING IMAGES...
# for i,tag in enumerate(['base','lines','source','target','drive','walk','gtfs','ondemand']):
#     filename = parent_dir + tag + '.png'
#     FILENAMES2[tag] = filename
#     DATAINDS2['imgs'][tag] = data_counter; data_counter = data_counter + 1;
#     imgm = px.imshow(sio.imread(filename));
#     fig.add_trace(imgm.data[0], 1, 1)
    

# DATAINDS2['imgs']['groups'] = {};
# for group in DELIVERY['groups']:
#     DATAINDS2['imgs']['groups'][group] = {};
#     TRIPS = tripsSorted['ondemand']['groups'][group]['trips'];
#     for k,trip in enumerate(TRIPS):
#         node1 = int(trip[0]); node2 = int(trip[1]);
#         tag = group+'_'+str(node1)+'_'+str(node2);
#         try: 
#             filename = parent_dir + 'groups/'+'ondemand_'+tag+'.png';
#             imgm = px.imshow(sio.imread(filename));
#             FILENAMES2[tag] = filename
#             DATAINDS2['imgs']['groups'][group][tag] = data_counter; data_counter = data_counter + 1;        
#             fig.add_trace(imgm.data[0], 1, 1)
#             fig.data[-1].visible = False;
#         except:
#             pass;

# DATAINDS2['imgs']['runs'] = {};
# for g,group in enumerate(DELIVERY['groups']):
#     DATAINDS2['imgs']['runs'][group] = {};
#     RUNS = tripsSorted['ondemand']['groups'][group]['runs']
#     for k,runid in enumerate(RUNS):
#         try: 
#             filename = parent_dir + 'runs/' + group + '_run' + str(runid) + '.png';
#             imgm = px.imshow(sio.imread(filename));
#             DATAINDS2['imgs']['runs'][group][runid] = data_counter; data_counter = data_counter + 1;
#             fig.add_trace(imgm.data[0], 1, 1)
#             fig.data[-1].visible = False;
#         except:
#             pass;

# for mode in ['drive','walk','gtfs']:
#     DATAINDS2['imgs'][mode+'_trips'] = {};
#     TRIPS = tripsSorted[mode]['trips']
#     for k,trip in enumerate(TRIPS):
#         node1 = str(int(trip[0])); node2 = str(int(trip[1]));
#         tag = mode + '_' + node1 + '_' + node2;
#         try: 
#             filename = parent_dir + mode + '_trips/' + tag + '.png';
#             imgm = px.imshow(sio.imread(filename));
#             FILENAMES2[tag] = filename
#             DATAINDS2['imgs'][mode+'_trips'][tag] = data_counter; data_counter = data_counter + 1;
#             fig.add_trace(imgm.data[0],1,1)
#             fig.data[-1].visible = False
#         except:
#             pass 

# # for i in range(len(fig.data)): 
# #     if not(i in all_start_inds):
# #         fig.data[i].visible = False;


# colors = {'drive':[1,0,0,1],
#           'walk':[1,1,0,1],
#           'gtfs':[1,0.5,0.,1],
#           'ondemand':[0,0,1,1],
#           'groups':[[0,0,1,1],[0.5,0,1,1]],
#           'time':[1,0,0,1],'money':[0,0.8,0.,1.],'conven':[0,0,1,1],'switches':[0,0,0,1]}



# ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES 
# ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES 
# ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES 
# ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES ###### BOXES 

# modes = ['drive','walk','gtfs','ondemand']
# factors = ['time','money','conven','switches']
# use_factors = ['time','money']

# TRIPS_SORTED = {};
# TRIPS_SORTED['time'] = tripsSorted
# TRIPS_SORTED['money'] = tripsSorted2;

# for i,mode in enumerate(modes):
#     for factor in use_factors:
#         inds = INDS['box_'+mode+'_'+factor]
#         tag1 = mode + '_costs';
#         DATAINDS2['box'][tag1] = {};
#         data_counter = data_counter + 1;
#         costs = TRIPS_SORTED[factor][mode]['costs']
#         color = px.colors.label_rgb(list(255*np.array(colors[mode])))
#         edgecolor = 'rgb(0,0,0)'; opac = 0.7;        
#         fig.add_trace(go.Box(y=costs,fillcolor=color,opacity=opac,marker_color=edgecolor,line={'width':1}),inds[0],inds[1])
#                              # width=0.5,marker = {'color' :'rgb(0,0,0)','opacity':0.9}), i+5,1)
#         # fig.data[-1].visible = True;


# DATAINDS2['box']['groups'] = {};
# DATAINDS2['box']['runs'] = {};
# for i,group in enumerate(groups):
#     for factor in use_factors: 
#         inds = INDS['box_'+group+'_' + factor]        
#         # try: 
#         # counts = tripsSorted['ondemand']['groups'][group]['counts']
#         costs = TRIPS_SORTED[factor]['ondemand']['groups'][group]['costs']
#         trips = TRIPS_SORTED[factor]['ondemand']['groups'][group]['trips']
#         counts = TRIPS_SORTED[factor]['ondemand']['groups'][group]['group_counts']
#         color = px.colors.label_rgb(list(255*np.array(colors['groups'][i])))
#         edgecolor = 'rgb(0,0,0)'; opac = 0.7; 
#         name = group + '_'
#         fig.add_trace(go.Box(y=costs,fillcolor=color,opacity=opac,marker_color=edgecolor,name=name,line={'width':1}),inds[0],inds[1])
#         data_counter = data_counter + 1;

# for factor in use_factors:
#     for mode in modes:
#         inds = INDS['box_people_'+factor]
#         tag1 = mode + '_costs';
#         DATAINDS2['box'][tag1] = {};
#         data_counter = data_counter + 1;
#         costs = TRIPS_SORTED[factor][mode]['costs']
#         color = px.colors.label_rgb(list(255*np.array(colors[mode])))
#         edgecolor = 'rgb(0,0,0)'; opac = 0.7;        
#         fig.add_trace(go.Box(y=costs,fillcolor=color,opacity=opac,marker_color=edgecolor,line={'width':1}),inds[0],inds[1])
#         fig.data[-1].visible = True;
    
#                          # width=0.5,marker = {'color' :'rgb(0,0,0)','opacity':0.9}), i+5,1)
#     # fig.data[-1].visible = True;


# ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS 
# ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS 
# ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS 
# ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS ##### BARS 

# DATAINDS2['bars']['groups'] = {};
# DATAINDS2['bars']['runs'] = {};
# for i,group in enumerate(groups):
#     inds = INDS['bar_'+group+'_time']
#     inds2 = INDS['bar_'+group+'_money']
#     # try: 
#     # counts = tripsSorted['ondemand']['groups'][group]['counts']
#     costs = tripsSorted['ondemand']['groups'][group]['costs']
#     trips = tripsSorted['ondemand']['groups'][group]['trips']
#     counts = tripsSorted['ondemand']['groups'][group]['group_counts']
#     color = px.colors.label_rgb(list(255*np.array(colors['groups'][i])))
#     all_start_inds.append(len(fig.data))
#     fig.add_trace(go.Bar(x=counts,y=costs,width=0.5,base ='overlay',marker = {'color' :color,'opacity':0.5}),inds[0],inds[1])
#     DATAINDS2['bars']['groups'][group+'_full'] = data_counter; data_counter = data_counter + 1;        
#     DATAINDS2['bars']['groups'][group] = {};

#     ########################################
#     # counts = tripsSorted['ondemand']['groups'][group]['counts']
#     costs2 = tripsSorted2['ondemand']['groups'][group]['costs']
#     trips2 = tripsSorted2['ondemand']['groups'][group]['trips']
#     counts2 = tripsSorted2['ondemand']['groups'][group]['group_counts']
#     color = px.colors.label_rgb(list(255*np.array(colors['groups'][i])))
#     all_start_inds.append(len(fig.data))
#     # try: 
#     fig.add_trace(go.Bar(x=counts2,y=costs2,width=0.5,base ='overlay',marker = {'color' :color,'opacity':0.5}),inds2[0],inds2[1])
#     # DATAINDS2['bars']['groups'][group+'_full'] = data_counter;
#     data_counter = data_counter + 1;      
#     # except: pass
#     ########################################
    
#     for k,trip in enumerate(trips):
#         node1 = str(int(trip[0])); node2 = str(int(trip[1])); 
#         color = 'rgb(0,0,0)';
#         fig.add_trace(go.Bar(x=[counts[k]],y=[costs[k]],width=0.5,base ='overlay',marker = {'color' :color,'opacity':1.}), inds[0],inds[1])
#         fig.data[-1].visible = False; 
#         tag = group+'_'+str(node1)+'_'+str(node2);
#         DATAINDS2['bars']['groups'][group][tag] = data_counter; data_counter = data_counter + 1;        

#         node1 = str(int(trip[0])); node2 = str(int(trip[1])); 
#         color = 'rgb(0,0,0)';
#         fig.add_trace(go.Bar(x=[counts2[k]],y=[costs2[k]],width=0.5,base ='overlay',marker = {'color' :color,'opacity':1.}), inds2[0],inds2[1])
#         fig.data[-1].visible = False
#         tag = group+'_'+str(node1)+'_'+str(node2);
#         # DATAINDS2['bars']['groups'][group][tag] = data_counter;
#         data_counter = data_counter + 1;        

    

#     RUNS = tripsSorted['ondemand']['groups'][group]['runs']
#     DATAINDS2['bars']['runs'][group] = {};
#     DATAINDS[group+'_runs'] = []
#     all_start_inds.append(len(fig.data))
#     color = px.colors.label_rgb(list(255*np.array(colors['groups'][i])))
#     for k,run in enumerate(RUNS):
#         DATAINDS2['bars']['runs'][group][run] = data_counter;
#         RUN = RUNS[run];
#         DATAINDS[group + '_runs'].append(len(fig.data))
#         costs = RUN['costs']
#         counts = RUN['overall_counts'];
#         data_counter = data_counter + 1;
#         fig.add_trace(go.Bar(x=counts,y=costs,base='overlay',width=0.5,marker = {'color' :color,'opacity':0.9}),inds[0],inds[1])
#         fig.data[-1].visible = False
    
# ##### NEW


# use_factors = ['time','money']
# for i,mode in enumerate(['drive','walk','gtfs','ondemand']):
#     for factor in use_factors:
#         inds = INDS['bar_'+mode+'_'+factor]
#         DATAINDS2['bars'][mode+'_trips_' + factor] = {};
#         trips = TRIPS_SORTED[factor][mode]['trips']
#         costs = TRIPS_SORTED[factor][mode]['costs']
#         counts = TRIPS_SORTED[factor][mode]['counts'];
#         # counts = list(range(len(costs)));
#         # print(mode,' counts:',counts)
#         color = px.colors.label_rgb(list(255*np.array(colors[mode])))
#         opac = 0.5;
#         if mode == 'walk': opac = 1;
#         fig.add_trace(go.Bar(x=counts,y=costs,width=0.5,marker = {'color' :color,'opacity':opac}), inds[0],inds[1])
#         fig.data[-1].visible = True;
#         DATAINDS2['bars'][mode + '_full_'+'factor'] = data_counter
#         data_counter = data_counter + 1;
#         for k,trip in enumerate(trips):
#             node1 = str(int(trip[0])); node2 = str(int(trip[1]));
#             tag = mode + '_' + node1 + '_' + node2;
#             DATAINDS2['bars'][mode+'_trips_'+factor][tag] = data_counter
#             data_counter = data_counter + 1;
#             fig.add_trace(go.Bar(x=[counts[k]],y=[costs[k]],width=0.5,marker = {'color' :'rgb(0,0,0)','opacity':0.9}),inds[0],inds[1]);
#             fig.data[-1].visible = False;



# # use_factors = ['time','money']
# # mode = 'gtfs'
# # for factor in use_factors:
# #     inds = INDS['bar_people_'+factor]
# #     DATAINDS2['bars'][mode+'_trips_' + factor] = {};
# #     trips = TRIPS_SORTED[factor][mode]['trips']
# #     costs = TRIPS_SORTED[factor][mode]['costs']
# #     counts = TRIPS_SORTED[factor][mode]['counts'];
#     # color = px.colors.label_rgb(list(255*np.array(colors[mode])))
#     # opac = 0.5;
#     # fig.add_trace(go.Bar(x=counts,y=costs,width=0.5,marker = {'color' :color,'opacity':opac}), inds[0],inds[1])
#     # fig.data[-1].visible = True;
#     # data_counter = data_counter + 1;
#     # DATAINDS2['bars'][mode + '_full_'+'factor'] = data_counter
#     # for k,trip in enumerate(trips):
#     #     node1 = str(int(trip[0])); node2 = str(int(trip[1]));
#     #     tag = mode + '_' + node1 + '_' + node2;
#     #     DATAINDS2['bars'][mode+'_trips_'+factor][tag] = data_counter
#     #     data_counter = data_counter + 1;
#     #     fig.add_trace(go.Bar(x=[counts[k]],y=[costs[k]],width=0.5,marker = {'color' :'rgb(0,0,0)','opacity':0.9}),inds[0],inds[1]);
#     #     fig.data[-1].visible = False;



        
# fig.update_layout(barmode='overlay')

# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############################################################################################
# #############################################################################################



# ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS 
# ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS 
# ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS ### SLIDERS 

# print('building sliders...')

# SLIDERS = [];
# # for tag in ['group0','group1']: 
# #     steps = []
# #     tag2 = tag + '_runs'
# #     datainds = DATAINDS[tag]; # + DATAINDS[tag2]
# #     print('len1: ',len1)
# #     print('len2: ',len2)

# #     len1 = len(DATAINDS[tag]);
# #     len2 = len(DATAINDS[tag2]);
# #     for i in range(len(DATAINDS[tag])): #intlen(fig.data)/2):    
# #         datainds = DATAINDS[tag]+DATAINDS[tag2]
# #         step = dict(method="update",label='',args=[{"visible": [False] * (len1+len2)},{},datainds])
# #         step["args"][0]["visible"][i] = True  
# #         step["args"][0]["visible"][i+len2] = True  
# #         steps.append(step)
# #     loc = CTRL_LOCS[tag2]
# #     if len(loc)>=3: length = loc[2];
# #     else: length = 0.2;
# #     SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="bottom",active=0,currentvalue={},pad={"t": 0},steps=steps))


# for group in groups: #['group0','group1']: 
#     steps = []
#     datainds1 = [];
#     datainds2 = [];
#     for run in DATAINDS2['imgs']['runs'][group]:
#         datainds1.append(DATAINDS2['imgs']['runs'][group][run]);

#     for run in DATAINDS2['bars']['runs'][group]:        
#         datainds2.append(DATAINDS2['bars']['runs'][group][run]);
#     len1 = len(datainds1);
#     len2 = len(datainds2);
#     datainds = datainds1 + datainds2; 
#     for i in range(len2):
#         step = dict(method="update",label='',args=[{"visible": [False] * (len1+len2)},{},datainds])
#         step["args"][0]["visible"][i] = True  
#         step["args"][0]["visible"][i+len1] = True  
#         steps.append(step)

#     tag2 = group + '_runs';
#     loc = CTRL_LOCS[tag2]
#     if len(loc)>=3: length = loc[2];
#     else: length = 0.2;
#     SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="top",active=0,currentvalue={},pad={"t": 0},steps=steps))


# for tag in groups: #['group0','group1']: 
#     ### NEW
#     steps = []    
#     datainds1 = []; datainds2 = [];
#     TAGS = DATAINDS2['imgs']['groups'][tag];
#     for tag2 in TAGS:
#         try: datainds1.append(DATAINDS2['imgs']['groups'][tag][tag2])
#         except: pass
#         try: datainds2.append(DATAINDS2['bars']['groups'][tag][tag2]);
#         except: pass
#     datainds = datainds1 + datainds2
#     len1 = len(datainds1); len2 = len(datainds2);

#     if len2>0:
#         print('len1: ',len1)
#         print('len2: ',len2)
#         for i in range(len2):
#             step = dict(method="update",label='',args=[{"visible": [False] * (len1+len2)},{},datainds])
#             step["args"][0]["visible"][i] = True  
#             step["args"][0]["visible"][len1+i] = True  
#             steps.append(step)
#         loc = CTRL_LOCS[tag]
#         if len(loc)>=3: length = loc[2];
#         else: length = 0.2;
#         SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="top",active=0,currentvalue={},pad={"t": 0},steps=steps))


# for tag0 in ['walk','drive','gtfs']: #,'ondemand_trips']:
#     # NEW
    
#     datainds1 = []; datainds2 = [];
#     dataindsn = {};
#     tag = tag0 + '_trips'
#     TAGS = DATAINDS2['imgs'][tag];
#     lenn = []
#     for tag2 in TAGS:
#         try: datainds1.append(DATAINDS2['imgs'][tag0 + '_trips'][tag2])
#         except: pass
#     for factor in use_factors:
#         dataindsn[factor] = []
#         for tag2 in TAGS:
#             dataindsn[factor].append(DATAINDS2['bars'][tag0 + '_trips_' + factor][tag2]);
#         lenn.append(len(dataindsn[factor]));

#     datainds = datainds1
#     for factor in use_factors:
#         datainds = datainds + dataindsn[factor]
#     len1 = len(datainds1);  len2 = len(datainds2);

#     # if lenn[0]>0
#     # print('1: ',datainds1)
#     # print('time: ' ,dataindsn['time'])
#     # print('money: ',dataindsn['money'])
#     # print(datainds)

#     if len1 > 0: 
#         steps = []
#         for i in range(len1):
#             step = dict(method="update",label='',args=[{"visible": [False] * (len1+np.sum(lenn))},{},datainds])
#             step["args"][0]["visible"][i] = True  
#             for k,factor in enumerate(use_factors):
#                 step["args"][0]["visible"][len1 + int(np.sum(lenn[:k])) + i] = True  
#             steps.append(step)
            
#         loc = CTRL_LOCS[tag]
#         if len(loc)>=3: length = loc[2];
#         else: length = 0.2;
#         SLIDERS.append(dict(x=loc[0],y=loc[1],len=length,xanchor="left",yanchor="top",
#                             active=0,currentvalue={},pad={"t": 0},steps=steps))    


# print('building buttons...')
# BUTTONS = [];
# for tag in ['lines','source','target','drive','gtfs','walk','ondemand']: 
#     loc = CTRL_LOCS[tag];
#     # print(DATAINDS[tag])
#     BUTTONS.append(dict(
#             buttons=list([
#                 dict(args=[{"visible":True},{},[DATAINDS2['imgs'][tag]]],label=tag + " ON" ,method="update"),
#                 dict(args=[{"visible":False},{},[DATAINDS2['imgs'][tag]]],label=tag + " OFF",method="update")]),
#             # type = "buttons",
#             direction="down",
#             pad={"r": 0, "t": 0},
#             showactive=True,x=loc[0],xanchor="left",y=loc[1],yanchor="top"));



# print('adding controls...')



# fig.update_layout(updatemenus=BUTTONS)
# fig.update_layout(sliders=SLIDERS)
# width = np.sum(column_widths); height = 1.2*np.sum(row_heights); #+padb1*3)
# fig.update_layout(width=width, height=height, xaxis_visible=False, yaxis_visible=False)



# fig.update_layout(showlegend=False)

# # fig.update_layout(xaxis_visible=False, yaxis_visible=False)

# # fig.update_layout(autosize=True,
# #     height=600,
# #     width=500,
# #     margin=dict(l=0,r=0,t=20,b=0),
# #     grid = {'rows': 2, 'columns': 1, 'pattern': "independent"})

# # fig.update_layout(column_widths = [400,300],row_heights = [800,300])



# print('showing...')
# fig.show()

# print('done showing.')

# import plotly.io as pio
# pio.write_html(fig, file='case3.html', auto_open=True)




    ########## ====================== GRAPH BASIC =================== ###################
    ########## ====================== GRAPH BASIC =================== ###################
    ########## ====================== GRAPH BASIC =================== ###################

    def nodesFromTrips(self,trips): ### ADDED TO CLASS
        nodes = [];
        for i,trip in enumerate(trips):
            node0 = trip[0]; node1 = trip[1];
            if not(node0 in nodes):
                nodes.append(node0);
            if not(node1 in nodes):
                nodes.append(node1);
        return nodes

    def edgesFromPath(self,path): ### ADDED TO CLASS
        """ computes edges in a path from a list of nodes """
        edges = []; # initializing list
        for i,node in enumerate(path): # loops through nodes in path
            if i<len(path)-1: # if not the last node
                node1 = path[i]; node2 = path[i+1]; 
                edges.append((node1,node2)) # add an edge tag defined as (node1,node2) which is the standard networkx structure
        return edges

    def locsFromNodes(self,nodes,GRAPH): ### ADDED TO CLASS
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
    rad_miles = 100.;
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


    # print(DISTS)
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
        self.group_polygons = params['group_polygons'];

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
            params2['polygon'] = self.group_polygons[k];
            params2['total_num_groups'] = self.num_groups;
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

    def generateCongestionModels(self,NETWORK,counts = {'num_counts':2,'num_per_count':1},verbose=False):

        print('DEMAND CURVE - generate (takes 10 MIN...)')
        print('...runs VRP solver (num_pts*num_per_count) times...')
        print('### ~ approx. 1 run/15 seconds...')

        if hasattr(self,'groups'):

            mode = 'ondemand'
            all_possible_trips = list(NETWORK.segs);
            if verbose: print('starting on-demand curve computations for a total of',len(all_possible_trips),'trips...')
            # trips_by_group = {group:[] for _,group in enumerate(list(self.groups)[::-1])}
            trips_by_group = {group:[] for _,group in enumerate(self.groups)}
            for _,seg in enumerate(all_possible_trips):
                group = NETWORK.segs[seg].group;
                trips_by_group[group].append(seg)

            # for k, group in enumerate(self.groups):
            for k, group in enumerate(trips_by_group):
                possible_trips = trips_by_group[group]
                GROUP = self.groups[group]
                GROUP.generateCongestionModel(possible_trips,NETWORK,self,counts,verbose=verbose)

    # def saveCongestionModels(self,NETWORK,verbose=False):

    #     print('saving demand curves for VRP solver... ')
    #     if hasattr(self,'groups'):
    #         mode = 'ondemand'
    #         for k, group in self.groups: #enumerate(trips_by_group): # possible_trips = trips_by_group[group]
    #             GROUP = self.groups[group]
    #             GROUP.fit
    #             GROUP.saveCongestionModel(possible_trips,NETWORK,self,counts,verbose=verbose)



    # def saveCongestionModel(self,verbose=True): #,possible_trips,NETWORK,ONDEMAND,counts = {'num_counts':2,'num_per_count':1},verbose=True):
    #     self.fit = {}
    #     self.fit['counts'] = np.outer(counts,np.ones(num_per_count))
    #     self.fit['pts'] = pts.copy();

    #     shp = np.shape(pts);
    #     xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
    #     yy = np.reshape(pts,shp[0]*shp[1])
    #     order = 1;
    #     coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
    #     Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
    #     self.fit['poly'] = coeffs[::-1]
    #     self.fit['Yh'] = Yh.copy();                

    def plotCongestionModels(self,ax=None,colors={},save_file=None):  ## NOT FIXED YET....

        self.groupFitColors = [];
        for i,group in enumerate(self.groups):
            if len(colors)>0:
                self.groupFitColors.append(colors[group]);
            elif hasattr(self.groups[group],'polygon_color'):
                self.groupFitColors.append(self.groups[group].polygon_color); #[1-frac,0,frac,0.5*frac+0.5]);
            else:
                frac = i/len(self.groups);
                self.groupFitColors.append([1-frac,0,frac,0.5*frac+0.5]);

            GROUP = self.groups[group]
            counts = self.groups[group].fit['counts']
            pts = self.groups[group].fit['pts']
            shp = np.shape(pts);
            xx = np.reshape(counts,shp[0]*shp[1])
            coeffs = self.groups[group].fit['poly']
            Yh = self.groups[group].fit['Yh'];
            for j in range(np.shape(pts)[1]):
                plt.plot(counts,pts[:,j],'o',color=self.groupFitColors[i])
            plt.plot(xx,Yh,'--',color=self.groupFitColors[i])

        if not(save_file==None):
            plt.savefig(save_file,bbox_inches='tight',pad_inches = 0,transparent=True)            
            plt.show()



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



        



# # def createGroupsDF(self):
# def createGroupsDF(polygons,types0 = []):  ### ADDED TO CLASS
#     ngrps = len(polygons);
#     groups = ['group'+str(i) for i in range(len(polygons))];
#     depotlocs = [];
#     for polygon in polygons:
#         depotlocs.append((1./np.shape(polygon)[0])*np.sum(polygon,0));    
#     if len(types0)==0: types = [['direct','shuttle'] for i in range(ngrps)]
#     else: types = types0;
#     GROUPS = pd.DataFrame({'group':groups,
#     'depot_loc':depotlocs,'type':types,'polygon':polygons,
#     'num_possible_trips':np.zeros(ngrps),
#     'num_drivers':np.zeros(ngrps) 
#        }); #,index=list(range(ngrps))
#     # new_nodes = pd.DataFrame(node_tags,index=[index_node])
#     # NODES[mode] = pd.concat([NODES[mode],new_nodes]);
#     return GROUPS

# # def createDriversDF(self): 
# def createDriversDF(params,WORLD):  ### ADDED TO CLASS
#     num_drivers = params['num_drivers']
#     if not('start_time' in params): start_times = [WORLD['main']['start_time'] for _ in range(num_drivers)]
#     elif not(isinstance(params['start_time'],list)): start_times = [params['start_time'] for _ in range(num_drivers)]
#     if not('end_time' in params): end_times = [WORLD['main']['end_time'] for _ in range(num_drivers)]
#     elif not(isinstance(params['end_time'],list)): end_times = [params['end_time'] for _ in range(num_drivers)]
#     if not('am_capacity' in params): am_capacities = [8 for _ in range(num_drivers)];
#     elif not(isinstance(params['am_capacity'],list)): am_capacities = [params['am_capacity'] for _ in range(num_drivers)]        
#     if not('wc_capacity' in params): wc_capacities = [2 for _ in range(num_drivers)];
#     elif not(isinstance(params['wc_capacity'],list)): wc_capacities = [params['wc_capacity'] for _ in range(num_drivers)]
#     OUT = pd.DataFrame({'start_time':start_times, 'end_time':end_times,'am_capacity':am_capacities, 'wc_capacity':wc_capacities})
#     return OUT
        

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


class RUN:
    def __init__(self,params,GRAPHS,FEEDS):

        self.run_id = params['run_id'];
        self.start_time = params['start_time']
        self.end_time = params['end_time']
        self.am_capacity = params['am_capacity'];
        self.wc_capacity = params['wc_capacity'];
        self.group_ind = params['group_ind']
        self.group = params['group'];

        self.GRAPHS = GRAPHS;
        self.FEEDS = FEEDS;

        ######################################################
        self.booking_ids = [];
        self.date = '2023-07-31'
        self.loc = params['loc']
        self.depot = {'pt': {'lat': self.loc[1], 'lon': self.loc[0]}, 'node_id': 0}
        self.depot_node = ox.distance.nearest_nodes(self.GRAPHS['drive'], self.loc[0], self.loc[1]);

        self.expected_cost = [0.];
        self.current_expected_cost = 0.;
        self.actual_average_cost = [];


        self.costs = {'dist':[],'time':[],'money':[],'switches':[],'conven':[]};
        currents = {'current_dist':None,'current_time':None,'current_money':None,'current_conven':None,'current_switches':None}
        self.costs = {**self.costs,**currents}

        self.DF = None;
        self.total_passengers = None; 
        self.max_num_passengers = None;
        self.ave_num_passengers = None;            


        self.VMT = [];
        self.PMT = [];
        self.VMTbyPMT = [];
        self.current_VMT = None; 
        self.current_PMT = None;
        self.current_VMTbyPMT = None;
        self.VTT = [];
        self.PTT = [];
        self.VTTbyPTT = [];
        self.current_VTT = None; 
        self.current_PTT = None;
        self.current_VTTbyPTT = None;        
        

class GROUP: 
    def __init__(self,grpsDF,driversDF,params):

        self.GRAPHS = params['GRAPHS'];
        self.FEEDS = params['FEEDS'];

        self.group_ind = params['group_ind']
        self.group = 'group'+str(self.group_ind);
        self.polygon = params['polygon']

        self.total_num_groups = params['total_num_groups'];
        self.polygon_color = [0,0,1,0.2 + 0.4*(self.group_ind/self.total_num_groups)];

        self.fit = {'poly':params['default_poly']}#np.array([406.35315058,  18.04891652])};

        self.booking_ids = [];
        self.time_matrix = np.zeros([1,1]);
        self.expected_cost = [0.];
        self.current_expected_cost = 0.;
        self.actual_average_cost = [];
        self.driver_runs = [];
        self.runs = {};
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

        self.PDF = [];
        self.MDF = [];
        self.current_PDF = [];
        self.current_MDF = [];

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
                run_data = {'run_id': i,
                    'start_time': driver_start_time,'end_time': driver_end_time,
                    'am_capacity': am_capacity,'wc_capacity': wc_capacity}
                self.driver_runs.append(run_data)
                run_data2 = {'loc':self.loc,'group':self.group,'group_ind':self.group_ind}
                run_data2 = {**run_data2,**run_data}                
                self.runs[i] = RUN(run_data2,self.GRAPHS,self.FEEDS);

        # self.booking_ids = [];
        # self.date = '2023-07-31'
        # self.loc = params['loc']
        # self.depot = {'pt': {'lat': self.loc[1], 'lon': self.loc[0]}, 'node_id': 0}
        # self.depot_node = ox.distance.nearest_nodes(self.GRAPHS['drive'], self.loc[0], self.loc[1]);


        else: 
            for i in range(num_drivers):
                driver_start_time = WORLD['main']['start_time'];
                driver_end_time = WORLD['main']['end_time'];
                am_capacity = 8
                wc_capacity = 2
                run_data = {'run_id': i,
                    'start_time': driver_start_time,'end_time': driver_end_time,
                    'am_capacity': am_capacity,'wc_capacity': wc_capacity}
                self.driver_runs.append(run_data)
                run_data2 = {'loc':self.loc,'group':self.group,'group_ind':self.group_ind}
                run_data2 = {**run_data2,**run_data}
                self.runs[i] = RUN(run_data2,self.GRAPHS,self.FEEDS);

        grpsDF['num_drivers'].iloc[self.group_ind] = num_drivers;
        grpsDF['num_possible_trips'].iloc[self.group_ind] = 0;

    # def addFit(self,poly):
    #     self.fit = {'poly':poly}

    # def saveCongestionModel(self,verbose=True): #,possible_trips,NETWORK,ONDEMAND,counts = {'num_counts':2,'num_per_count':1},verbose=True):
    #     self.fit = {}
    #     self.fit['counts'] = np.outer(counts,np.ones(num_per_count))
    #     self.fit['pts'] = pts.copy();

    #     shp = np.shape(pts);
    #     xx = np.reshape(np.outer(counts,np.ones(shp[1])),shp[0]*shp[1])
    #     yy = np.reshape(pts,shp[0]*shp[1])
    #     order = 1;
    #     coeffs = np.polyfit(xx, yy, order); #, rcond=None, full=False, w=None, cov=False)
    #     Yh = np.array([np.polyval(coeffs,xi) for _,xi in enumerate(xx)])
    #     self.fit['poly'] = coeffs[::-1]
    #     self.fit['Yh'] = Yh.copy();


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
            num_trips = len(possible_trips);
            if num_trips>1: counts = np.linspace(0.1*num_trips,0.9*(num_trips-1),num_pts)
            else: counts = np.linspace(1,len(possible_trips)-1,num_pts)


            counts = [int(count) for _,count in enumerate(counts)];
                    
            if verbose: print('starting on-demand curve computation for group NUMBER ',self.group_ind) #'with',len(possible_trips),'...')
                    
            GRAPH = self.GRAPHS['ondemand'];
                    
            # ONDEMAND = self;
            print('...for a total number of trips of',len(possible_trips))
            print('counts to compute: ',counts)

            pts = np.zeros([len(counts),num_per_count]);

            ptsx = {};
            for i,count in enumerate(counts):
                count2 = count;
                counts_added = 0;
                print('...computing averages for',count,'active ondemand trips in',self.group,'...'); #' group',self.group_ind,'...')
                for j in range(num_per_count):
                    count_completed = False;
                    while not(count_completed): # or counts_added<10:
                        try:
                            current_average = np.mean(pts[i,:j])
                            # try: 
                            trips_to_plan = sample(possible_trips,count2)
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
                            # print(payload)
                            manifests = optimizer.offline_solver(payload)


                            PDF = self.payloadDF(payload,self.GRAPHS,include_drive_nodes = True);
                            MDF = self.manifestDF(manifests,PDF)
                            self.PDF = PDF;
                            self.MDF = MDF;
                            self.current_PDF = PDF;
                            self.current_MDF = MDF;
                            average_time,times_to_average = self.assignCostsFromManifest(trips_to_plan,NETWORK.segs,nodes,MDF,NETWORK,0)


                            pts[i,j] = average_time
                            count_completed = True;
                        except:
                            print('COUNT FAILED...')
                            count2 = count2 + 1;
                            counts_added = counts_added + 1;
                            if counts_added > 10:
                                break
                            print('trying again for count of ',count2)
                            count_completed = False;







                    # except:
                    #     pts[i,j] = current_average
                    #     print('balked on running VRP...')


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


    def routeFromStops(self,nodes,GRAPH): ### ADDED TO CLASS        
    # def routeFromStops(nodes,GRAPH): ### ADDED TO CLASS
        PATH_NODES = [];
        for i,node in enumerate(nodes):
            if not(node==None):
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
    # def routesFromManifest(MDF,GRAPHS): ### ADDED TO CLASSA
        MDF = self.current_MDF;
        PLANS = {};
        if len(MDF)>0:
            GRAPHS = self.GRAPHS;    
            max_id_num = int(np.max(list(MDF['run_id'])))+1
            for i in range(max_id_num):
                ZZ = MDF[MDF['run_id']==i]
                ZZ = list(ZZ.sort_values(['scheduled_time'])['drive_node'])
                out = self.routeFromStops(ZZ,GRAPHS['drive']);
                PLANS[i] = out.copy();
        return PLANS

    def singleRoutesFromManifest(self,version='by_nodes'): #,MDF,GRAPHS): ### ADDED TO CLASS
    # def singleRoutesFromManifest(MDF,GRAPHS): ### ADDED TO CLASS
        MDF = self.current_MDF;
        GRAPHS = self.GRAPHS;
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
                out = self.routeFromStops(zz,GRAPHS['drive']);
                node1 = int(zz[0]);
                node2 = int(zz[-1]);
                if version == 'by_nodes':
                    PLANS[i][(node1,node2)] = {}
                    PLANS[i][(node1,node2)]['booking_id'] = idd;
                    PLANS[i][(node1,node2)]['nodes'] = out['nodes']
                    PLANS[i][(node1,node2)]['edges'] = out['edges'];
                elif version == 'by_booking_ids':
                    PLANS[i][idd] = {}
                    PLANS[i][idd]['orig_dest'] = (node1,node2); 
                    PLANS[i][idd]['nodes'] = out['nodes']
                    PLANS[i][idd]['edges'] = out['edges'];

        return PLANS        

    def assignCostsFromManifest(self,active_segs,all_segs,nodes,MDF,NETWORK,ave_cost,track=True): #active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS

    # def assignCostsFromManifest(active_trips,nodes,MDF,WORLD,ave_cost,track=True):  ### ADDED TO CLASS
        mode = 'ondemand';

        times_to_average = [0.];

        actual_costs = {};
        runs_info = {};
        run_ids = {};

        # print(MDF['run_id'])
        # print(list(MDF['run_id'].unique()))
        # adsfasdfasd
        
        SEG_DATA = {};
        if len(MDF)>0:

            ############################################################################################
            ############################################################################################
            ############################################################################################            
            # PLANS = self.routesFromManifest(MDF,self.GRAPHS)
            PLANS = self.singleRoutesFromManifest(version='by_booking_ids')
            # print(PLANS2)
            # asdfasdfasdf

            # for runid in PLANS:
            #     PLAN = PLANS[runid];
            #     for trip in PLAN:
            #         TRIP = PLAN[trip];
            #         node1 = trip[0]; node2 = trip[1];
            #         # tag = 'ondemand'+'_'+group+'_'+'run'+str(runid)+'_'+str(node1)+'_'+str(node2);
            #         # ONDEMAND_TRIPS[tag] = TRIP['edges'];
            #         # ondemand_trip_tags.append(tag);
            #         # tags.append(tag)
            #         tag = 'ondemand'+'_'+group+'_'+str(node1)+'_'+str(node2);
            #         ONDEMAND_TRIPS[tag] = TRIP['edges'];
            #         ondemand_trip_tags.append(tag);
            ############################################################################################
            ############################################################################################
            ############################################################################################

            runids = list(MDF['run_id'].unique())
            for runid in runids:

                MDF2 = MDF[MDF['run_id'] == runid];
                MDF2 = MDF2.sort_values(by="scheduled_time")  
                RUN = self.runs[runid];
                RUN.DF = MDF2;



                run_start_time = np.min(MDF2['scheduled_time']);
                run_end_time = np.max(MDF2['scheduled_time']);
                total_run_time = run_end_time - run_start_time
                VTT = total_run_time
                VMT = total_run_time;

                prev_event_time = run_start_time;
                bookingids = MDF2['booking_id'].unique();

                no_passenger_time = 0;
                num_passengers = [];
                PTT = 0; PMT = 0;

                for bookingid in bookingids:

                    mask1 = (MDF2['booking_id'] == bookingid) & (MDF2['action'] == 'pickup')
                    mask2 = (MDF2['booking_id'] == bookingid) & (MDF2['action'] == 'dropoff')
                    ind1 = np.where(mask1)[0][0]
                    ind2 = np.where(mask2)[0][0]

                    # PASSEN = MDF2[MDF2['booking_id'] == bookingid]
                    PICKUP = MDF2.iloc[ind1]; #PASSEN[PASSEN['action'] == 'pickup'];
                    DROPOFF = MDF2.iloc[ind2]; #PASSEN[PASSEN['action'] == 'dropoff'];

                    time1 = PICKUP['scheduled_time'];
                    time2 = DROPOFF['scheduled_time'];
                    MDF3 = MDF2.iloc[ind1:ind2+1];
                    current_num_passengers = np.sum(MDF3['action']=='pickup');
                    num_passengers.append(current_num_passengers)

                    SEG_DATA[bookingid] = {};
                    SEG_DATA[bookingid]['current_num_passengers'] = current_num_passengers ######
                    SEG_DATA[bookingid]['pickup_time_scheduled'] = time1;
                    SEG_DATA[bookingid]['dropoff_time_scheduled'] = time2;
                    # path_nodes = PLANS[runid][bookingid]['nodes']
                    # SEG_DATA[bookingid]['path'] = path_nodes;


                    PTT = PTT + (time2 - time1);
                    PMT = PMT + (time2 - time1);




                RUN.total_passengers = len(bookingids)
                RUN.max_num_passengers = np.max(num_passengers);
                RUN.ave_num_passengers = np.sum(num_passengers)/len(num_passengers);


                RUN.VTT.append(VMT)
                RUN.PTT.append(PMT)
                RUN.VTTbyPTT.append(VTT/PTT)
                RUN.VMT.append(VMT)
                RUN.PMT.append(PMT)
                RUN.VMTbyPMT.append(VMT/PMT)

                RUN.current_VTT = VTT 
                RUN.current_PTT = PTT
                RUN.current_VTTbyPTT = VTT/PTT
                RUN.current_VMT = VMT
                RUN.current_PMT = PMT
                RUN.current_VMTbyPMT = VMT/PMT;


        ######################################################        ######################################################        ######################################################
        ######################################################        ######################################################        ######################################################        

        ######################################################       ######################################################       ######################################################
        ######################################################       ######################################################       ######################################################

        # print(NUM_PASSENGERS)

        SEGS = NETWORK.segs;
        for i,seg in enumerate(active_segs):
            SEG = SEGS[seg];
            node1 = seg[0]
            node2 = seg[1]
            booking_id = SEG.booking_id
            main_node_id1 = nodes[seg[0]]['main']
            main_node_id2 = nodes[seg[1]]['main']
            curr_node_id1 = nodes[seg[0]]['curr']
            curr_node_id2 = nodes[seg[1]]['curr']

            # print(curr_node_id1)
            # print(curr_node_id2)
            # try: 
            # mask1 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
            # mask2 = (MDF['booking_id'] == booking_id) & (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');
            # mask1 = (MDF['node_id'] == curr_node_id1) & (MDF['action'] == 'pickup');
            # mask2 = (MDF['node_id'] == curr_node_id2) & (MDF['action'] == 'dropoff');
            mask1 = (MDF['booking_id'] == booking_id) & (MDF['action'] == 'pickup');
            mask2 = (MDF['booking_id'] == booking_id) & (MDF['action'] == 'dropoff');

            # print(MDF[mask1]['scheduled_time'].iloc[0])
            # print(MDF[mask2]['scheduled_time'].iloc[0])

            time1 = list(MDF[mask1]['scheduled_time'])
            time2 = list(MDF[mask2]['scheduled_time'])

            if len(time1)>0 and len(time2)>0:
                time1 = time1[0]
                time2 = time2[0];
                time_diff = np.abs(time2-time1)
                run_id = int(list(MDF[mask1]['run_id'])[0]);
            else:
                time1 = None;
                time2 = None;
                time_diff = None;
                run_id = None;


            if booking_id in SEG_DATA:
                current_num_passengers = SEG_DATA[booking_id]['current_num_passengers']
                # = SEG_DATA[booking_id]['current_num_passengers']
                # = SEG_DATA[booking_id]['current_num_passengers']
                # print('booking id found...',booking_id)
            else:
                current_num_passengers = 1;


            SEG.num_passengers = current_num_passengers; 

            SEG.pickup_time_scheduled = time1;
            SEG.dropoff_time_scheduled = time2; 
            SEG.pickup_time = time1;
            SEG.dropoff_time = time2; 
            SEG.run_id = run_id;


            if run_id in PLANS:
                PLAN = PLANS[run_id];
                if booking_id in PLAN:
                    path_nodes = PLANS[run_id][booking_id]['nodes']
                    dist = SEG.pathCost(path_nodes,self.GRAPHS['drive'],weight='dist');
                    SEG.current_path = path_nodes;
                    # print(dist)
                else:
                    dist = None;
                    SEG.current_path = None;
            else:
                dist = None;
                SEG.current_path = None;
            #####
            # GRAPH = self.GRAPHS['drive']
            # weight = 'dist';
            # cost = 0;
            # path = path_nodes;
            # # print(path)
            # for i in range(len(path)-1):
            #     node1 = path[i]
            #     node2 = path[i+1];
            #     # print((node1,node2) in GRAPH.edges)
            #     test_edge = list(GRAPH.edges)[0];
            #     if len(test_edge)==3: tag = (node1,node2,0);
            #     else: tag = (node1,node2);
            #     if tag in GRAPH.edges:
            #         EDGE = GRAPH.edges[tag]
            #         cost = cost + EDGE[weight]
            # dist = cost
            SEG.costs['dist'].append(dist)
            SEG.costs['current_dist'] = dist;
            #####

            actual_costs[seg] = time_diff;
            run_ids[seg] = run_id;

            runs_info[seg] = {}
            runs_info[seg]['runid'] = run_id;

            if not(time_diff==None):
                if time_diff < 720000:
                    times_to_average.append(time_diff);

            try: 
                MDF2 = MDF[MDF['run_id'] == run_id]
                # print(np.where(MDF2['node_id'] == curr_node_id1));#[0][0];
                # print(np.where(MDF2['node_id'] == curr_node_id2)); #[0][0];
                # print(np.where(MDF2['bookingode_id'] == curr_node_id1));#[0][0];
                # print(np.where(MDF2['node_id'] == curr_node_id2)); #[0][0];
                # ind1 = np.where(MDF2['node_id'] == curr_node_id1)[0][0];
                # ind2 = np.where(MDF2['node_id'] == curr_node_id2)[0][0];
                ind1 = np.where(MDF2['booking_id'] == booking_id)[0][0];
                ind2 = np.where(MDF2['booking_id'] == booking_id)[0][0];

                MDF2 = MDF2.iloc[ind1:ind2+1]
                runs_info[seg]['num_passengers'] = MDF2['booking_id'].nunique();
            except: 
                runs_info[seg]['num_passengers'] = 1;
                # actual_costs[seg] = 1000000000.;
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


            # time_cost = est_ave_time
            time_cost = actual_costs[seg]; #est_ave_time
            money_cost = NETWORK.monetary_cost;


            # NETWORK.segs[seg].costs['current_time'] = time_cost;
            if seg in active_segs:
                NETWORK.segs[seg].costs['time'].append(time_cost);#est_ave_time);
                NETWORK.segs[seg].costs['current_time'] = time_cost;#est_ave_time);
                NETWORK.segs[seg].costs['money'].append(money_cost);#est_ave_time);
                NETWORK.segs[seg].costs['current_money'] = money_cost;#est_ave_time);

                try:
                    NETWORK.segs[seg]['run_id'] = run_ids[seg] 
                    NETWORK.segs[seg]['num_passengers'] = runs_info[seg]['num_passengers'];
                except:
                    pass
            else:
                if len(NETWORK.segs[seg].costs['time'])==0: prev_time = time_cost;
                else: prev_time = NETWORK.segs[seg].costs['time'][-1]
                NETWORK.segs[seg].costs['time'].append(prev_time);
                NETWORK.segs[seg].costs['current_time'] = prev_time;
                NETWORK.segs[seg].costs['time'].append(money_cost);
                NETWORK.segs[seg].costs['current_time'] = money_cost;


            factors = ['conven','switches']; #,'dist','time']
            for j,factor in enumerate(factors):#WORLD[mode]['trips'][trip]['costs']):
                # if not(factor == 'time'): 
                # WORLD[mode]['trips']['costs'][factor] = 0. 
                NETWORK.segs[seg].costs[factor].append(0.);
                NETWORK.segs[seg].costs['current_'+factor] = 0. 
        return average_time,times_to_average



class TRIP:
    def __init__(self,trip_id,modes,nodes,CONVERTER,delivery_grps=[],deliveries=[]): #,node_types,NODES,deliveries = []):
    # def makeTrip(modes,nodes,NODES,delivery_grps=[],deliveries=[]): #,node_types,NODES,deliveries = []):
        trip = {};
        legs = [];

        self.trip_id = trip_id;

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
                        seg_id = mode + '_seg' + str(int(len(NETWORK.segs)))
                        NETWORK.segs[seg] = SEG(seg_id,seg,mode,PERSON,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND,params={'group':group})
                        if not(self.trip_id in NETWORK.segs[seg].trip_ids):
                            NETWORK.segs[seg].trip_ids.append(self.trip_id)
                        if not(PERSON.person_id in NETWORK.segs[seg].people):
                            NETWORK.segs[seg].people.append(PERSON.person_id)
                    else:
                        if not(self.trip_id in NETWORK.segs[seg].trip_ids):
                            NETWORK.segs[seg].trip_ids.append(self.trip_id)
                        if not(PERSON.person_id in NETWORK.segs[seg].people):
                            NETWORK.segs[seg].people.append(PERSON.person_id)


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
                    # weightsb = {};
                    # costsb = {};
                    for l,factor in enumerate(PERSON.weights[mode]):
                        weight = PERSON.weights[mode][factor]
                        cost = NETWORK.segs[seg].costs['current_'+factor]
                        if factor == 'switches':
                            try: 
                                weight = applyProgressiveWeight(cost,weight);
                            except:
                                print('switching weight FAILED...')
                                weight = weight[0];             
                        weights.append(weight); #PERSON.weights[mode][factor])
                        costs.append(cost); #NETWORK.segs[seg].costs['current_'+factor])
                        # weightsb[factor] = PERSON.weights[mode][factor];
                        # costsb[factor] = PERSON.costs[mode][factor];
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
            if True: #isinstance(weights,list):
                values_filtered = [];
                weights_filtered = [];
                for i,value in enumerate(values):
                    weight = weights[i]
                    if isinstance(value, (int, float, complex)) and isinstance(weight, (int, float, complex)):
                        values_filtered.append(value)
                        weights_filtered.append(weight)
                values_filtered = np.array(values_filtered)
                weights_filtered = np.array(weights_filtered)
                cost = weights_filtered@values_filtered;

            elif False: #isinstance(weights,dict):
                values_filtered = [];
                weights_filtered = [];
                for i,factor in enumerate(values):
                    value = values[factor];
                    weight = weights[factor];
                    if factor == 'switches':
                        try: 
                            weight = applyProgressiveWeight(value,weight);
                        except:
                            print('switching weight FAILED...')
                            weight = weight[0];             
                    if isinstance(value, (int, float, complex)) and isinstance(weight, (int, float, complex)):
                        values_filtered.append(value)
                        weights_filtered.append(weight)
                values_filtered = np.array(values_filtered)
                weights_filtered = np.array(weights_filtered)
                cost = weights_filtered@values_filtered;                
        return cost

class SEG:
    def __init__(self,seg_id,seg,mode,PERSON,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND,params={}): #WORLD,params={}):
    # def querySeg(seg,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group='group0'):
                    # cost,path,_ = querySeg(seg_details,mode,PERSON,NODES,GRAPHS,DELIVERY,WORLD,group=group)
        factors = ['dist','time','money','conven','switches']
        self.mode = mode
        self.seg_id = seg_id;
        self.source = seg[0]
        self.target = seg[1]
        # self.start_time = 0;
        # self.end_time = 14400;
        self.factors = factors;
        self.active = True;
        self.CONVERTER = CONVERTER

        self.trip_ids = [];
        self.people = [];

        self.uncongested = {};
        self.uncongested['costs'] = {'time':None,'dist':None,'money':None,'conven':None,'switches':None};

        # self.pickup_time_window_start = PERSON.pickup_time_window_start;
        # self.pickup_time_window_end = PERSON.pickup_time_window_end;
        # self.dropoff_time_window_start = PERSON.dropoff_time_window_start
        # self.dropoff_time_window_end = PERSON.dropoff_time_window_end


        # try: 
        #     ##### NEEDS TO BE UPDATED 

        source_drive = CONVERTER.convertNode(self.source,mode,'drive');#,to_type='feed')                
        target_drive = CONVERTER.convertNode(self.target,mode,'drive');#,to_type='feed')                


        # if True: 
        self.pickup_time_window_start = PERSON.pickup_time_window_start;
        self.pickup_time_window_end = PERSON.pickup_time_window_end;
        self.dropoff_time_window_start = PERSON.dropoff_time_window_start;
        self.dropoff_time_window_end = PERSON.dropoff_time_window_end;



        # except:
        #     print('balked on adding deliv seg...')
        #     self.pickup_time_window_start = PERSON.pickup_time_window_start;
        #     self.pickup_time_window_end = PERSON.pickup_time_window_end;
        #     self.dropoff_time_window_start = PERSON.dropoff_time_window_start
        #     self.dropoff_time_window_end = PERSON.dropoff_time_window_end


        self.pickup_time_scheduled = None;
        self.dropoff_time_scheduled = None;
        self.num_passengers = 1;


        self.booking_id = PERSON.booking_id;
        self.pickup_node_id = PERSON.pickup_node_id;
        self.dropoff_node_id = PERSON.dropoff_node_id;
        self.am = PERSON.am;
        self.wc = PERSON.wc;


        # start1 = WORLD['main']['start_time']; #seg_details[2]
        # end1 = WORLD['main']['end_time']; #seg_details[4]
        distance = 0;
        travel_time = 0;
        
        current_costs = {'current_dist':distance,'current_time':travel_time,
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
        else: self.group = None;


        NETWORK = NETWORKS[mode];

        booking_id = int(len(NETWORK.booking_ids));
        self.booking_id = booking_id;
        self.run_id = None;
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

    def pathCost(self,path,GRAPH,weight='c'):
        cost = 0;
        # print(path)
        for i in range(len(path)-1):
            node1 = path[i]
            node2 = path[i+1];
            # print((node1,node2) in GRAPH.edges)
            test_edge = list(GRAPH.edges)[0];
            if len(test_edge)==3: tag = (node1,node2,0);
            else: tag = (node1,node2);
            if tag in GRAPH.edges:
                EDGE = GRAPH.edges[tag]
                cost = cost + EDGE[weight]
        return cost

    def planSeg(self,mode,GRAPHS,FEEDS,NETWORKS,CONVERTER,ONDEMAND,track=False):
        NETWORK = NETWORKS[mode];
        if mode == 'drive' or mode == 'walk':
            self.planDijkstraSeg(mode,NETWORK,GRAPHS,track=track);
        if mode == 'gtfs':
            self.planGTFSSeg(mode,NETWORK,GRAPHS,FEEDS,self.CONVERTER,ONDEMAND,track=track);
        if mode == 'ondemand':
            pass
            #self.planDelivSeg(mode,NETWORKS,GRAPHS,FEEDS,CONVERTER,ONDEMAND,track=track);



    def planDijkstraSeg(self,mode,NETWORK,GRAPHS,track=False,mass=0):

        # NETWORK = NETWORKS[mode]
        GRAPH = GRAPHS[mode];
        try:
            temp = nx.multi_source_dijkstra(GRAPH, [self.source], target=self.target, weight='c'); #Dumbfunctions
            # path = nx.shortest_path(GRAPH, source=self.source, target=self.target, weight='c'); #, method='dijkstra')[source]
            travel_time = temp[0];
            path = temp[1];
            distance = self.pathCost(path,GRAPH,weight='dist');
        except: 
            print('no path found for dijkstra seg ',(self.source,self.target),'in mode',mode,'...')
            travel_time = 10000000000;
            distance = 100000000000;
            path = [];

        # print(path)

        money_cost = NETWORK.monetary_cost
        conven_cost = 0;
        switch_cost = 0;


        self.costs['current_dist'] = distance
        self.costs['current_time'] = travel_time
        self.costs['current_money'] = money_cost;
        self.costs['current_conven'] = conven_cost;
        self.costs['current_switches'] = switch_cost;
        self.current_path = path;
        self.mass = 0;
                    
        if track==True:
            self.costs['dist'].append(distance)
            self.costs['time'].append(travel_time)
            self.costs['money'].append(1)
            self.costs['conven'].append(conven_cost)
            self.costs['switches'].append(switch_cost)
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

        # print((self.source,self.target))
        # print(self.source in PREV_NODES)
        # print(self.target in PREV_NODES)

        try:
            time = REACHED[self.source][-1][self.target]
            # print(REACHED[source])
        except:
            time = 1000000000000.;
            distance = 10000000000.;

        try:
            stop_list,trip_list = create_chains(self.source,self.target,PREV_NODES,PREV_TRIPS);
            #### Computing Number of switches...
            init_stop = stop_list[0];
            num_segments = 1;
            for i,stop in enumerate(stop_list):
                if stop == init_stop: num_segments = len(stop_list)-i-1;
            _,stopList,edgeList = list_inbetween_stops(FEED,stop_list,trip_list); #,GRAPHS);
            path,_ = gtfs_to_transit_nodesNedges(stopList,self.CONVERTER)
            distance = self.pathCost(path,GRAPH,weight='dist');
            switches = num_segments; #- 1;
            # print('gtfs path found.')
        except: 
            # print('no path found for gtfs seg ',(self.source,self.target),'...')
            num_segments = 1;
            switches = 0; #num_segments; # - 1;
            path = [];
            distance = 100000000000.;
        # # print('num segs gtfs: ',num_segments)                
        # print(path)

        money_cost = NETWORK.monetary_cost
        conven_cost = 0;
        switch_cost = switches;


        self.costs['current_dist'] = distance;
        self.costs['current_time'] = time
        self.costs['current_money'] = money_cost;
        self.costs['current_conven'] = conven_cost; 
        self.costs['current_switches'] = switch_cost; 
        self.current_path = path;
        self.mass = mass;
        
        if track==True:
            self.costs['dist'].append(distance)
            self.costs['time'].append(time)
            self.costs['money'].append(money_cost)
            self.costs['conven'].append(conven_cost)
            self.costs['switches'].append(switch_cost)
            self.path.append(path);

        num_missing_edges = 0;
        if mass > 0:
            #print(path)
            for j,node in enumerate(path):
                if j < len(path)-1:
                    edge = (path[j],path[j+1],0)
                    if edge in NETWORK.edge_masses:
                        edge_mass = NETWORK.edge_masses[edge][-1] + mass;
                        # edge_cost = 1.*edge_mass + 1.;
                        NETWORK.edge_masses[edge][-1] = edge_mass;
                        # WORLD[mode]['edge_costs'][edge][-1] = edge_cost;
                        # WORLD[mode]['current_edge_costs'][edge] = edge_cost;    
                        NETWORK.current_edge_masses[edge] = edge_mass;
                    else:
                        num_missing_edges = num_missing_edges + 1
                        # if np.mod(num_missing_edges,10)==0:
                        #     print(num_missing_edges,'th missing edge...')
        if verbose:
            if num_missing_edges > 1:
                print('# edges missing in gtfs segment...',num_missing_edges)



    # def addDelivSeg(seg_details,mode,GRAPHS,DELIVERIES,WORLD,group='group0',mass=1,track=False,PERSON={}):
    # def planDelivSeg(self,mode,NETWORKS,GRAPHS,FEEDS,CONVERTER,ONDEMAND,group='group0',mass=1,track=False,PERSON={}): ###### ADDED TO CLASS 


    #     NETWORK = NETWORKS[mode]

    #     start1 = self.start_time; #seg_details[2]
    #     end1 = self.end_time; #seg_details[4]
        

    #     trip = (source,target);#,start1,start2,end1,end2);
    #     GRAPH = GRAPHS[mode];
    #     # if mode == 'transit' and mass > 0:
    #     # #     print(mode)
    #     # try:
    #     #     temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c'); #Dumbfunctions
    #     #     distance = temp[0];
    #     #     # if mode == 'walk':
    #     #     #     distance = distance/1000;
    #     #     path = temp[1]; 
    #     # except: 
    #     #     #print('no path found for bus trip ',trip,'...')
    #     #     distance = 1000000000000;
    #     #     path = [];
    #     distance = 0;

    #     if not(seg in NETWORK.segs.keys()):

    #         # print('adding delivery segment...')

    #         NETWORK.segs[seg] = {};
    #         NETWORK.segs[seg].costs = {'time':[],'money':[],'conven':[],'switches':[]}
    #         #WORLD[mode]['trips'][trip]['costs'] = {'current_time':[],'current_money':[],'current_conven':[],'current_switches':[]}
    #         NETWORK.segs[seg].path = [];
    #         NETWORK.segs[seg].delivery_history = [];
    #         NETWORK.segs[seg].mass = 0;
    #         NETWORK.segs[seg].delivery = None; #delivery;

    #         ##### NEW
    #         NETWORK.segs[seg].typ = None;  #shuttle or direct

    #         if len(PERSON)>0:
    #             if hasattr(PERSON,'group_options'): NETWORK.segs[seg].group_options = PERSON.group_options;

    #         NETWORK.segs[seg].costs['current_time'] = distance;
    #         NETWORK.segs[seg].costs['current_money'] = 1;
    #         NETWORK.segs[seg].costs['current_conven'] = 1; 
    #         NETWORK.segs[seg].costs['current_switches'] = 1; 
    #         NETWORK.segs[seg].current_path = None; #path;        

    #         # WORLD[mode]['trips'][trip][''] asdfasdf

            

    #         NETWORK.segs[seg].am = int(1);
    #         NETWORK.segs[seg].wc = int(0);


    #         NETWORK.segs[seg].group = group;

    #         booking_id = int(len(NETWORK.booking_ids));
    #         NETWORK.segs[seg].booking_id= booking_id;
    #         NETWORK.booking_ids.append(booking_id);


    #         for _,group in enumerate(ONDEMAND.groups):
    #             GROUP = ONDEMAND.groups[group];
    #             time_matrix = GROUP.time_matrix;
    #             shp = np.shape(time_matrix)
    #             time_matrix = np.block([[time_matrix,np.zeros([shp[0],2])],[np.zeros([2,shp[1]]),np.zeros([2,2]) ]]);
    #             GROUP.time_matrix = time_matrix;

    #             NETWORK.segs[seg].pickup_node_id = shp[0];
    #             NETWORK.segs[seg].dropoff_node_id = shp[0]+1;


    #         try: 
    #             ##### NEEDS TO BE UPDATED 
    #             dist = nx.shortest_path_length(GRAPHS['drive'], source=trip[0], target=trip[1], weight='c');
    #             maxtriptime = dist*4;
    #             pickup_start = np.random.uniform(low=start1, high=end1-maxtriptime, size=1)[0];
    #             pickup_end = pickup_start + 60*20;
    #             dropoff_start = pickup_start;
    #             dropoff_end = pickup_end + maxtriptime;

    #             NETWORK.segs[seg].pickup_time_window_start = int(pickup_start);
    #             NETWORK.segs[seg].pickup_time_window_end = int(pickup_end);
    #             NETWORK.segs[seg].dropoff_time_window_start = int(dropoff_start);
    #             NETWORK.segs[seg].dropoff_time_window_end = int(dropoff_end);

    #         except:
    #             print('balked on adding deliv seg...')
    #             NETWORK.segs[seg].pickup_time_window_start = int(start1);
    #             NETWORK.segs[seg].pickup_time_window_end = int(end1);
    #             NETWORK.segs[seg].dropoff_time_window_start = int(start1);
    #             NETWORK.segs[seg].dropoff_time_window_end = int(end1);










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

        self.initial_costs_computed = False;

        self.start_time = params['time_window'][0];
        self.end_time = params['time_window'][1];


        self.monetary_cost = 0;
        if 'monetary_cost' in params: self.monetary_cost = params['monetary_cost'];





        dists = self.createEdgeDists();
        nx.set_edge_attributes(self.GRAPH,dists,'dist');

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
                maxspeed = 1.5; # in mph
                maxspeed = maxspeed * 0.447 # assumed in meters? 1 mph = 0.447 m/s
                length = EDGE['length'] ; 
                t0 = length/(maxspeed); # * 0.447)
                EDGE['cost_fx'] = [t0];
                POLYS[edge] = [t0];

        return POLYS

    ######################################
    def createEdgeDists(self):
        DISTS = {};        
        GRAPH = self.GRAPH; #S[mode];
        for i,edge in enumerate(GRAPH.edges):
            EDGE = GRAPH.edges[edge];
            if 'length' in EDGE: DISTS[edge] = EDGE['length']; 
            elif 'geometry' in EDGE: DISTS[edge] = EDGE['geometry'].length*111139 #degrees to meters
            else: DISTS[edge] = 0;
        return DISTS

    # def pathCost(path,weight='c'):
    #     GRAPH = self.GRAPH
    #     cost = 0;
    #     for i in range(len(path)-1):
    #         node1 = path[i]
    #         node2 = path[i+1];
    #         EDGE = GRAPH.edges[(node1,node2)]
    #         cost = cost + EDGE[weight]
    #     return cost



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

        ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE 
        ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE 
        ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE 
        ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE 
        ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE ###### DRIVE    

        if self.mode == 'drive':
        # def world_of_drive(WORLD,PEOPLE,GRAPHS,verbose=False,clear_active=True): ##### ADDED TO CLASS 
            #graph,costs,sources, targets):
            if verbose: print('starting driving computations...')
            GRAPH = self.GRAPH;


            if kk == 0 or not(self.initial_costs_computed): #(not('edge_masses' in WORLD[mode].keys()) or (kk==0)): #?????/
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

                self.initial_costs_computed = True;

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


        ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK 
        ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK 
        ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK 
        ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK 
        ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK ######## WALK 

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
            if not(hasattr(self,'edge_masses')) or (kk==0) or not(self.initial_costs_computed) :
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
                self.initial_costs_computed = True;


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

        ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
        ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
        ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
        ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 
        ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND ###### ONDEMAND 

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
                    if True: 
                        segs_to_plan = GROUPS_OF_SEGS[group]
                        GROUP = ONDEMAND.groups[group]

                        ########### -- XX START HERE ---- ####################
                        ########### -- XX START HERE ---- ####################
                        ########### -- XX START HERE ---- ####################
                        print('COMPUTING ondemand trips for ',group)
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

                            
                            manifests = optimizer.offline_solver(payload)

                            # manifest = optimizer(payload) ####
                            # manifest = fakeManifest(payload)

                            PDF = GROUP.payloadDF(payload,GRAPHS,include_drive_nodes = True);
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
                    #     GROUP.expected_cost.append(1000)
                    #     GROUP.current_expected_cost = 1000
                    #     GROUP.actual_average_cost.append(1000)



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


        ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS 
        ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS 
        ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS 
        ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS 
        ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS 
        ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS ######## GTFS 



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
                    # self.planGTFSSeg(mode,NETWORK,GRAPHS,FEEDS,self.CONVERTER,ONDEMAND,track=track);

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
    # def compute_UncongestedEdgeCosts(WORLD,GRAPHS):  #### AeDDED TO CLASS 
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

# def convertNode(self,node,from_mode,to_mode,from_type = 'graph',to_type = 'graph',verbose=False):
#         transit_node2 = CONVERTER.convertNode(transit_node2,'gtfs','gtfs',to_type='feed')
    
            try:
                if self.mode == 'gtfs': 
                    path = SEG.current_path 
                    time = SEG.costs['current_time']
                    distance = SEG.costs['current_dist'];

                else: 
                    temp = nx.multi_source_dijkstra(GRAPH, [source], target=target, weight='c0'); #Dumbfunctions
                    time = temp[0];                
                    path = temp[1]; 
                    distance = SEG.pathCost(path,GRAPH,weight='dist')
            except: 
                #print('no path found for bus trip ',trip,'...')
                time = None;
                distance = None;
                path = [];


            SEG.uncongested = {};
            SEG.uncongested['path'] = path;
            SEG.uncongested['costs'] = {}; 
            SEG.uncongested['costs']['dist'] = distance;
            SEG.uncongested['costs']['time'] = time;
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

    def __init__(self,person,CONVERTER,GRAPHS,FEEDS,NETWORKS,ONDEMAND,PRE,params,current_trip_num): 

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



        PREp = PRE[person]; 

        # 'take_car': 1.0,
        # 'take_transit': 1.0,
        # 'take_ondemand': 1.0,
        # 'take_walk': 1.0,
        # 'leave_time_start': 67984.0,
        # 'leave_time_end': 69784.0,
        # 'arrival_time_start': 73736.0,
        # 'arrival_time_end': 75536.0,
        # 'orig_loc': [-85.29339164, 35.00417406],
        # 'dest_loc': [-85.31486454, 35.04002926],
        # 'home_node': 202639241,
        # 'work_node': 202641745,
        # 'pop': 1.0,


        # 'seg_types': [('ondemand',),
        #               ('walk', 'gtfs', 'walk'),
        #               ('walk', 'gtfs', 'ondemand'),
        #               ('ondemand', 'gtfs', 'walk'),
        #               ('ondemand', 'gtfs', 'ondemand')],


        # 'median_income': 40380.0,

        # 'drive_weight_time': 1,
        # 'drive_weight_money': 0.009014363546310055,
        # 'drive_weight_conven': 0,
        # 'drive_weight_switches': 0,

        # 'walk_weight_time': 1,
        # 'walk_weight_money': 0,
        # 'walk_weight_conven': 0,
        # 'walk_weight_switches': 0,

        # 'ondemand_weight_time': 1,
        # 'ondemand_weight_money': 0.07211490837048044,
        # 'ondemand_weight_conven': 0,
        # 'ondemand_weight_switches': [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 1.0],

        # 'transit_weight_time': 1,
        # 'transit_weight_money': 0.027043090638930165,
        # 'transit_weight_conven': 0,
        # 'transit_weight_switches': [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 1.0]}



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

        self.person_id = person;

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


        self.trip_ids = [];


                        # 'pickup_time_window_start': 18000,
                        # 'pickup_time_window_end': 19800,
                        # 'dropoff_time_window_start': 18900,
                        # 'dropoff_time_window_end': 22200,


        start1 = params['start_time']; #PREp['leave_time_start'];
        # print(pickup_time_window_start)
        end1 = params['end_time'] #PREp['leave_time_end'];

        if False: #True:
            self.pickup_time_window_start = 18000;
            self.pickup_time_window_end = 19800;
            self.dropoff_time_window_start = 18900;
            self.dropoff_time_window_end = 22200;

        if True:
            # start1 = pickup_time_window_start
            # end1 = dropoff_time_window_end;
            try:
                dist = nx.shortest_path_length(GRAPHS['drive'], source=source_drive, target=target_drive, weight='c');
            except:
                dist = 3600.;

            maxtriptime = dist*1.2;
            pickup_start = np.random.uniform(low=start1, high=end1-maxtriptime, size=1)[0];
            pickup_end = pickup_start + 60*20;
            dropoff_start = pickup_start;
            dropoff_end = pickup_end + maxtriptime;

            self.pickup_time_window_start = int(pickup_start);
            self.pickup_time_window_end = int(pickup_end);
            self.dropoff_time_window_start = int(dropoff_start);
            self.dropoff_time_window_end = int(dropoff_end);            


        elif False:
            self.pickup_time_window_start = params['start_time']
            self.pickup_time_window_end = params['end_time']
            self.dropoff_time_window_start = params['start_time']
            self.dropoff_time_window_end = params['end_time']

        elif False:
            self.pickup_time_window_start = PREp['leave_time_start'];
            self.pickup_time_window_end = PREp['leave_time_end'];
            self.dropoff_time_window_start = PREp['arrival_time_start'];
            self.dropoff_time_window_end = PREp['arrival_time_end'];


        self.time_windows = {};


        current_booking_id = len(ONDEMAND.booking_ids)
        self.booking_id = current_booking_id;
        self.pickup_node_id = 1;  # None;
        self.dropoff_node_id = 2; # None;




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
                self.costs[mode][factor] = 0.;




        # 'transit_weight_switches': [0.0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 1.0]}



                tagg = mode;
                if mode == 'gtfs': tagg = 'transit';
                self.weights[mode][factor] = PRE[person][tagg + '_weight_' + factor]
                #PRE[person]['weight'][mode][factor]


                # try:
                #     self.weights[mode][factor] = PRE[person]['weight'][mode][factor]
                # except:
                #     if factor == 'time': self.weights[mode][factor] = 1.
                #     else: self.weights[mode][factor] = 1.
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
                trip_id = 'trip'+str(int(current_trip_num))
                self.trips[segs] = TRIP(trip_id,segs,nodes_temp,CONVERTER,deliveries_temp2);#,deliveries_temp);
                current_trip_num = current_trip_num + 1;
                self.trip_ids.append(trip_id)
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


def create_chains(stop1,stop2,PREV_NODES,PREV_TRIPS,max_trans = 4): ### ADDED TO CLASS
    STOP_LIST = [stop2];

    TRIP_LIST = [];
    for i in range(max_trans):
        
        # try: 
        # print(PREV_NODES[stop][-(i+1)][stop])
        stop = STOP_LIST[-(i+1)]
        # print(stop in PREV_NODES)
        stop2 = PREV_NODES[stop1][-(i+1)][stop]     
        trip = PREV_TRIPS[stop1][-(i+1)][stop]
        # except:
        #     stop2 = None;
        #     trip = None;

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


def gtfs_to_transit_nodesNedges(stopList,CONVERT): #### ADDED TO CLASS

    nodeList = [];
    edgeList = [None];
    prev_node = None;
    for i,stop in enumerate(stopList):
        # try: 
        # print(CONV)
        new_node = CONVERT.convertNode(stop,'gtfs','gtfs',from_type='feed')

    # def convertNode(self,node,from_mode,to_mode,from_type = 'graph',to_type = 'graph',verbose=False):


        nodeList.append(new_node)
        edgeList.append((prev_node,new_node,0))
        # except:
        #     pass
    edgeList = edgeList[1:]
    return nodeList,edgeList

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
    

def plotODs(GRAPHS,SIZES,NODES,scale=1.,figsize=(15,15),ax=None,with_regions=False,group_polygons = [],colors = [],save_file=None):
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
        fig,ax = ox.plot_graph(GRAPHS['drive'], #ax=ax,
                            bgcolor=bgcolor,  
                            node_color=node_colors,
                            node_size = node_sizes,
                            edge_color=edge_colors,
                            edge_linewidth=edge_widths,
                            figsize=figsize,
                            show=False); #file_format='svg')
        if with_regions:
            for i,shape in enumerate(group_polygons):
                plotCvxHull(ax,shape,params = {'color':colors[i][:3],'alpha':colors[i][3]});
        
        if not(save_file==None):
            plt.savefig(save_file,bbox_inches='tight',pad_inches = 0,transparent=True)            
            # plt.show()



        # else:
        #     fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=bgcolor,  
        #                         node_color=node_colors,
        #                         node_size = node_sizes,
        #                         edge_color=edge_colors,
        #                         edge_linewidth=edge_widths,
        #                         figsize=figsize,
        #                         show=False); #file_format='svg')



def plotShapesOnGraph(GRAPHS,shapes,figsize=(10,10),ax=None):
    if not(ax==None):
        print('drawing shapes')
        plt.sca(ax)
        ox.plot_graph(GRAPHS['drive'], #use_active_ax=True,
                        
                        bgcolor=[0.8,0.8,0.8,0.8],
                        node_color = [1,1,1,1],node_size = 1,
                        edge_color = [1,1,1,1],
                        edge_linewidth = 2,
                        figsize = figsize,
                        show = False,close=False);
    # print(ax)
    else:
        fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=[0.8,0.8,0.8,0.8],
                             node_color = [1,1,1,1],node_size = 1,
                             edge_color = [1,1,1,1],
                             edge_linewidth = 0.5,
                             figsize = figsize,
                             show = False,close=False);
    



    for shape in shapes:
        plotCvxHull(ax,shape,params = {});
    plt.show()
        # group_polygons = [np.random.rand(10,2) for i in range(5)];




