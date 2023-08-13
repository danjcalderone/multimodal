








# ------------ CODE DUMP ------------ CODE DUMP ------------ CODE DUMP 
# ------------ CODE DUMP ------------ CODE DUMP ------------ CODE DUMP 
# ------------ CODE DUMP ------------ CODE DUMP ------------ CODE DUMP 
# ------------ CODE DUMP ------------ CODE DUMP ------------ CODE DUMP 
# ------------ CODE DUMP ------------ CODE DUMP ------------ CODE DUMP 



cutoff = VARS['lam'][-1];
for k, node in enumerate(all_pickup_nodes):
    if node in VARS['pickups'][ind]:
        ii = np.where([node==n for _,n in enumerate(all_pickup_nodes)])[0][0];
        if Mlocs[ii]>cutoff:
            print('check')
        
        else:
            print('error: ', Mlocs[ii]-cutoff);

print(VARS['lam'][ind])
Mlocs1 = [];

nbins = 20;
bin_width = (np.max(Mlocs)-np.min(Mlocs))/nbins;
bins = np.min(Mlocs) + bin_width*np.array(list(range(nbins+1)))
for i,node in enumerate(NODES):
    Mlocs1.append(MLOCS[node])
ax = plt.hist(Mlocs,bins,color='blue')
ax = plt.hist(Mlocs1,bins, color = 'orange',alpha=1.)
plt.xlim([0,1000])

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
ox.config(use_cache=True, log_console=True)

# get graphs of different infrastructure types, then combine
place = 'Berkeley, California, USA'
G1 = ox.graph_from_place(place, custom_filter='["highway"~"residential"]')
G2 = ox.graph_from_place(place, custom_filter='["railway"~"rail"]')
G = nx.compose(G1, G2)

# get building footprints
fp = ox.footprints_from_place(place)

# plot highway edges in yellow, railway edges in red
ec = ['y' if 'highway' in d else 'r' for _, _, _, d in G.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G, bgcolor='k', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)

# add building footprints in 50% opacity white
fp.plot(ax=ax, color='w', alpha=0.5)
plt.show()

# transit_start_nodes = []; #sample(list(bus_graph.nodes()), num_sources)
# transit_end_nodes = []; #sample(list(bus_graph.nodes()), num_targets)    
# walk_transit_init_nodes = [];
# walk_transit_final_nodes = [];
# for i,_ in enumerate(source_nodes):
#     loc = ORIG_LOC[i];
#     node = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0],loc[1]);    
#     transit_start_nodes.append(node);
#     x = GRAPHS['transit'].nodes[node]['x']
#     y = GRAPHS['transit'].nodes[node]['y']
#     node = ox.distance.nearest_nodes(GRAPHS['walk'],x,y);
#     walk_transit_init_nodes.append(node);    

# for i,_ in enumerate(target_nodes):
#     loc = DEST_LOC[i];
#     node = ox.distance.nearest_nodes(GRAPHS['transit'], loc[0],loc[1]);
#     transit_end_nodes.append(node);
#     x = GRAPHS['transit'].nodes[node]['x']
#     y = GRAPHS['transit'].nodes[node]['y']
#     node = ox.distance.nearest_nodes(GRAPHS['walk'],x,y);
#     walk_transit_final_nodes.append(node);    
    
# TRANSIT_START_LOC = []; 
# for i,node in enumerate(transit_start_nodes):
#     x = GRAPHS['transit'].nodes[node]['x']
#     y = GRAPHS['transit'].nodes[node]['y']
#     TRANSIT_START_LOC.append((x,y));
#     #DEST_LOC.append((center_point[0],center_point[1]))

# TRANSIT_END_LOC = [];
# for i,node in enumerate(transit_end_nodes):
#     x = GRAPHS['transit'].nodes[node]['x']
#     y = GRAPHS['transit'].nodes[node]['y']
#     TRANSIT_END_LOC.append((x,y));
#     #DEST_LOC.append((center_point[0],center_point[1]))
# WALK_TRANSIT_INIT_LOC = [];
# for i,node in enumerate(walk_transit_init_nodes):
#     x = GRAPHS['walk'].nodes[node]['x']
#     y = GRAPHS['walk'].nodes[node]['y']
#     WALK_TRANSIT_INIT_LOC.append((x,y));

# WALK_TRANSIT_FINAL_LOC = [];
# for i,node in enumerate(walk_transit_final_nodes):
#     x = GRAPHS['walk'].nodes[node]['x']
#     y = GRAPHS['walk'].nodes[node]['y']
#     WALK_TRANSIT_FINAL_LOC.append((x,y));
    
################################################################
################################################################    
        
# DELIVERY_TRANSIT_LOC = [];
# for i,node in enumerate(delivery_transit_nodes):
#     x = GRAPHS['transit'].nodes[node]['x']
#     y = GRAPHS['transit'].nodes[node]['y']
#     DELIVERY_TRANSIT_LOC.append((x,y));    
    
    
# DELIVERY_LOC = [];
# for i,node in enumerate(delivery_nodes):
#     x = sample_graph.nodes[node]['x']
#     y = sample_graph.nodes[node]['y']
#     DELIVERY_LOC.append((x,y));
    
    
# delivery2_nodes = sample(list(sample_graph.nodes()), num_deliveries2)    
# DELIVERY2_LOC = [];
# for i,node in enumerate(delivery_nodes):
#     x = sample_graph.nodes[node]['x']
#     y = sample_graph.nodes[node]['y']
#     DELIVERY2_LOC.append((x,y));    


# #G2 = max(nx.strongly_connected_components_subgraphs(GRAPHS['transit']), key=len)
# center_node = ox.distance.nearest_nodes(GRAPHS['transit'], center_point[0],center_point[1]);
# important_nodes1 = transit_start_nodes + transit_end_nodes + delivery_transit_nodes;
# important_nodes2 = transit_start_nodes + transit_end_nodes + delivery_transit_nodes;
# graph = GRAPHS['transit']
# print(len(GRAPHS['transit'].nodes))
# print('original node number: ', len(important_nodes2))
# for i,node in enumerate(important_nodes1):
#     try: 
#         path = nx.shortest_path(graph, source=center_node, target=node, weight=None);
#         for j,nn in enumerate(path):
#             if not(nn in important_nodes2):
#                 important_nodes2.append(nn)
#     except:
#         print('no path found...')
# print('final node number: ', len(important_nodes2)) 
# GRAPHS['transit'] = GRAPHS['transit'].subgraph(important_nodes2)
# print(GRAPHS['transit'])



# start_time = time.time();
# ORIGS = {}; DESTS = {}; ODS = {};
# for k,typ in enumerate(od_types): #range(len(orig_types)):
#     graph = GRAPHS[typ];
#     ORIGS[typ] = [];
#     DESTS[typ] = [];
    
#     for i in range(len(ORIG_LOC)):
#         ORIGS[typ].append(ox.distance.nearest_nodes(graph, ORIG_LOC[i][0], ORIG_LOC[i][1]));
#         #print(ORIG[-1])
#     for i in range(len(DEST_LOC)):
#         DESTS[typ].append(ox.distance.nearest_nodes(graph, DEST_LOC[i][0], DEST_LOC[i][1]));
#         #print(DEST[-1])
#     ODS[typ] = {};        
#     for i in range(len(ORIG_LOC)):
#         for j in range(len(DEST_LOC)):
#             ODS[typ][(i,j)] = [ORIGS[typ][i],DESTS[typ][j]]
# end_time = time.time()
# print('time to compute origins and destinations: ',end_time-start_time)

# print(NODES['ondemand'])
# find_node = list(GRAPHS['transit'].nodes())[0]
# start_time = time.time();
# xx = GRAPHS['transit'][find_node]
# end_time = time.time();
# print('time to find in graph: ', end_time-start_time)

# find_node = list(GRAPHS['transit'].nodes())[0]
# start_time = time.time();
# xx = NODES['transit']['drive'][find_node]
# end_time = time.time();
# print('time to index in df: ', end_time-start_time)

# find_node = list(GRAPHS['transit'].nodes())[0]
# start_time = time.time();
# loc = np.where(NODESDF == find_node)
# end_time = time.time();
# print('time to find in df: ', end_time-start_time)
# zmodes = ['drive','transit','drive'];
# znodes = [[delivery1_node_lists['source'][0]],
#           [list(GRAPHS['transit'].nodes())[0]],
#           [list(GRAPHS['transit'].nodes())[-1]],
#           [delivery1_node_lists['source'][-1]]];
# ztypes = ['drive','transit','transit','drive']

# ztrip = makeTrip(zmodes,znodes,ztypes,NODES);#deliveries = []):

# zmodes2 = ['drive','transit','drive'];
# znodes2 = [{'nodes':[delivery1_node_lists['source'][0]],'type':'drive'},
#            {'nodes':[list(GRAPHS['transit'].nodes())[0]],'type':'transit'},
#            {'nodes':[list(GRAPHS['transit'].nodes())[-1]],'type':'transit'},
#            {'nodes':[delivery1_node_lists['source'][-1]],'type':'drive'}];
# ztypes2 = ['drive','transit','transit','drive']

# ztrip2 = makeTrip2(zmodes2,znodes2,NODES);#deliveries = []):
# print(delivery1_node_lists['source'][0])
# print(NODES['drive']['transit'][NODES['drive'].index[0]])
# print(ztrip)
# print(ztrip2['structure'])
# calcTrip(ztrip,PEOPLE['person0'],NODES,GRAPHS,WORLD)
# print(NODES['drive'].index.drop_duplicated())

# import peartree as pt

# # Automatically identify the busiest day and
# # read that in as a Partidge feed
# # feed = pt.get_representative_feed(path)

# # Set a target time period to
# # use to summarize impedance
# start = 7*60*60  # 7:00 AM
# end = 10*60*60  # 10:00 AM

# # Converts feed subset into a directed
# # network multigraph
# G = pt.load_feed_as_graph(feed, start, end)






##### ----- GRAPHING ----- GRAPHING -------- GRAPHING ------ ##############
##### ----- GRAPHING ----- GRAPHING -------- GRAPHING ------ ##############
##### ----- GRAPHING ----- GRAPHING -------- GRAPHING ------ ##############
##### ----- GRAPHING ----- GRAPHING -------- GRAPHING ------ ##############


##### -------- DRIVE --------------

#### EDGES 
bgcolor = [0.8,0.8,0.8,1]
node_color = [];
NODE_COLOR = [];
node_size = [];
node_edgecolor = [];
node_zorder = [];
edge_color = [];
edge_width = [];
drive_trip_edges = [];
for i,trip in enumerate(DRIVE['trips']):
    path = DRIVE['trips'][trip]['path'][-1]
    for j,node in enumerate(path):
        if j < len(path)-1:
            edge = (path[j],path[j+1],0);
        drive_trip_edges.append(edge)
        
graph = GRAPHS['drive']        

for k,node in enumerate(graph.nodes):
    if False:
        print('asdf')
#     elif node == sink:
#         node_color.append([0,0,0]); #[0,0,1])
#         node_size.append(100)
#     elif node in all_pickup_nodes: 
#         ii = np.where([node == n for n in all_pickup_nodes])[0][0];
#         # NODE_COLOR[i][node] = [0,0,1]
#         node_color.append(cmap(Mlocs[ii]/maxMloc))
#         node_size.append(200)
    elif node in node_lists['source']['drive']:
        node_color.append([0,0,1])
        node_size.append(100)        
        node_edgecolor.append('k')   
        node_zorder.append(100);
    elif node in node_lists['target']['drive']:
        node_color.append([1,0,0])
        node_size.append(100)
        node_edgecolor.append('k')
        node_zorder.append(100);        

    elif node in node_lists['transit_drive1']['drive']:
        node_color.append([0,0,1,0.5])
        node_size.append(50)
        node_edgecolor.append('w')
        node_zorder.append(10);        
    elif node in node_lists['transit_drive2']['drive']:
        node_color.append([1,0,0,0.5])
        node_size.append(50)
        node_edgecolor.append('w')
        node_zorder.append(10);        
        
#     elif node in target_nodes:
#         node_color.append([0,0,1])
#         node_size.append(200)        
#     elif node in delivery_nodes:
#         node_color.append([1,0,1])
#         node_size.append(200)        
#     elif node in delivery2_nodes:
#         node_color.append([1,0,0])
#         node_size.append(200)        
        
    else: 
        node_color.append([1,1,1]); #[0,0,1])
        node_size.append(10)
        node_edgecolor.append('w')
        node_zorder.append(1);        
                
for k,edge in enumerate(graph.edges):    
    edge_found = False;
    if edge_found == False:
        if edge in drive_trip_edges:
            edge_color.append([0,0,0]); 
            edge_width.append(4)             
        else: 
            edge_color.append([1,1,1]);
            edge_width.append(2)             
        
                    
fig, ax = ox.plot_graph(GRAPHS['drive'],bgcolor=bgcolor,
                        node_color=node_color,node_edgecolor=node_edgecolor,
                        node_size=node_size,node_zorder=1,#node_zorder,
                        edge_color=edge_color, edge_linewidth=edge_width)

##### -------- ONDEMAND --------------

#### EDGES         

bgcolor = [0.8,0.8,0.8,1]
node_color = [];
NODE_COLOR = [];
node_size = [];
edge_color = [];
node_edgecolor = [];
edge_width = [];
ondemand_trip_edges = [];
for i,trip in enumerate(ONDEMAND['trips']):
    path = ONDEMAND['trips'][trip]['tsp']['route']; #path'][-1]
    for j,node in enumerate(path):
        if j < len(path)-1:
            edge = (path[j],path[j+1],0);
        ondemand_trip_edges.append(edge)       
        
graph = GRAPHS['ondemand']
for k,node in enumerate(graph.nodes):

    if False: 
        print('asdf')
#     if node == sink:        
#         node_color.append([0,0,0]); #[0,0,1])
#         node_size.append(400)
#     elif node in all_pickup_nodes: 
#         ii = np.where([node == n for n in all_pickup_nodes])[0][0];
#         # NODE_COLOR[i][node] = [0,0,1]
#         node_color.append(cmap(Mlocs[ii]/maxMloc))
#         node_size.append(200)

    elif node in node_lists['source']['ondemand']:
        node_color.append([0,0,1])
        node_size.append(100)
        node_edgecolor.append('k')
    elif node in node_lists['target']['ondemand']:
        node_color.append([1,0,0])
        node_size.append(100)
        node_edgecolor.append('k')        
    elif node in node_lists['transit_ondemand1']['ondemand']:
        node_color.append([1,0,1])
        node_size.append(50)
        node_edgecolor.append('w')        
    elif node in node_lists['transit_ondemand2']['ondemand']:
        node_color.append([1,0,1])
        node_size.append(50)        
        node_edgecolor.append('w')
        
    elif node in delivery1_node_lists['source']:
        node_color.append([0,0,1])
        node_size.append(100)        
        node_edgecolor.append('k')        
    elif node in delivery2_node_lists['source']:
        node_color.append([0,0,1])
        node_size.append(100)        
        node_edgecolor.append('k')        

    elif node in delivery1_node_lists['transit']:
        node_color.append([1,0.5,0])
        node_size.append(100)        
        node_edgecolor.append('w')        
    elif node in delivery2_node_lists['transit']:
        node_color.append([1,0.5,0])
        node_size.append(100)        
        node_edgecolor.append('w')        
        
    else: 
        node_color.append([1,1,1]); #[0,0,1])
        node_size.append(2)
        node_edgecolor.append('w')        
        
                
for k,edge in enumerate(graph.edges):    
    edge_found = False;
    if edge_found == False:
        if False: #edge in drive_trip_edges:
            edge_color.append([0,0,0]); 
            edge_width.append(4)             
        elif edge in ondemand_trip_edges:
            edge_color.append([0,0,0]); 
            edge_width.append(4)                         
        else: 
            edge_color.append([1,1,1]);
            edge_width.append(2)             
        
                    


##### --------- TRANSIT --------------


bgcolor = [0.8,0.8,0.8]
node_color = [];
NODE_COLOR = [];
node_size = [];
edge_color = [];
edge_width = [];

# walk_transit_nodes = [];
# for i,person in enumerate(PEOPLE):
#     walk_transit_nodes.append(PERSON['sources']['walk_final']);
#     walk_transit_nodes.append(PERSON['targets']['walk_initial']);

transit_trip_edges = [];        
for i,trip in enumerate(TRANSIT['trips']):
    path = TRANSIT['trips'][trip]['path'][-1]
    if (len(path)>1) and (len(path) < 50):
        #print(TRANSIT['trips'][trip]['path'][0])
        for j,node in enumerate(path):
            if j < len(path)-1:
                edge = (path[j],path[j+1],0);
                transit_trip_edges.append(edge)                
                edge = (path[j+1],path[j],0);
                transit_trip_edges.append(edge)           
                
for k,node in enumerate(GRAPHS['transit'].nodes):
    if False: #node in delivery_transit_nodes:
        node_color.append([0,1,0])
        node_size.append(50)
        
    elif node in node_lists['transit_walk1']['transit']:
        node_color.append([1,0.75,0])
        node_size.append(50)        
    elif node in node_lists['transit_walk2']['transit']:
        node_color.append([1,0.5,0])
        node_size.append(50)        
    elif node in node_lists['transit_ondemand1']['transit']:
        node_color.append([1,0,0.75])
        node_size.append(50)        
    elif node in node_lists['transit_ondemand2']['transit']:
        node_color.append([1,0,0.5])
        node_size.append(50)        
        
        
    else: 
        node_color.append([1,1,1]); #[0,0,1])
        node_size.append(2)                
            

for k,edge in enumerate(GRAPHS['transit'].edges):    
    edge_found = False;
    if edge_found == False:
        if edge in transit_trip_edges:
            edge_color.append([0,0,0]); 
            edge_width.append(4)             
        else: 
            edge_color.append([1,1,1]);
            edge_width.append(2)         
            
            
            

fig, ax = ox.plot_graph(GRAPHS['transit'],bgcolor=bgcolor,
                        node_color=node_color,node_edgecolor='w',node_size=node_size,node_zorder=1,
                        edge_color=edge_color, edge_linewidth=edge_width)

fileName = 'current'
printFigs = True;
plt.axis('off')
if printFigs:
    plt.savefig(fileName+'.pdf',bbox_inches='tight',pad_inches = 0,transparent=True)

        
                    

##### --------- WALK -----------------


draw_source_nodes = [];
draw_target_nodes = [];
draw_transit_nodes = [];

bgcolor = [0.8,0.8,0.8]
node_color = [];
NODE_COLOR = [];
node_size = [];
edge_color = [];
edge_width = [];

walk_initial_edges = [];
walk_final_edges = [];
walk_trip_edges = [];

for i,trip in enumerate(WALK['trips']):
    path = WALK['trips'][trip]['path'][-1]
    for j,node in enumerate(path):
        if j < len(path)-1:
            edge = (path[j],path[j+1],0);
        walk_trip_edges.append(edge)


graph = GRAPHS['walk']
# print(graph.nodes)

# for i,person in enumerate(PEOPLE):
#     draw_source_nodes.append(PERSON['sources']['walk_initial']);
#     draw_transit_nodes.append(PERSON['sources']['walk_final']);
#     draw_transit_nodes.append(PERSON['targets']['walk_initial']);
#     draw_target_nodes.append(PERSON['targets']['walk_final']);
    
# for p,person in enumerate(PEOPLE):
#     PERSON = PEOPLE[person]
#     node1 = PERSON['sources']['walk_initial'];
#     node2 = PERSON['targets']['walk_initial'];
#     trip = (node1,node2);
#     path = WALK['trips'][trip]['path'][-1]
#     for j,node in enumerate(path):
#         if j < len(path)-1:
#             edge = (path[j],path[j+1],0);
#             walk_initial_edges.append(edge);                           
#             edge = (path[j+1],path[j],0);
#             walk_initial_edges.append(edge);
#     node1 = PERSON['sources']['walk_final'];
#     node2 = PERSON['targets']['walk_final'];
#     trip = (node1,node2);
#     path = WALK['trips'][trip]['path'][-1]
#     for j,node in enumerate(path):
#         if j < len(path)-1:
#             edge = (path[j],path[j+1],0);
#             walk_final_edges.append(edge)                            
#             edge = (path[j+1],path[j],0);
#             walk_final_edges.append(edge)
            
   
for k,node in enumerate(graph.nodes):
    if False: #node in draw_source_nodes:
        node_color.append([0,0,0])
        node_size.append(200)
    elif node in node_lists['source']['walk']:
        node_color.append([0,0,1])
        node_size.append(50)        
    elif node in node_lists['target']['walk']:
        node_color.append([1,0,0])
        node_size.append(50)        
    elif node in node_lists['transit_walk1']['walk']:
        node_color.append([1,.8,0])
        node_size.append(50)        
    elif node in node_lists['transit_walk2']['walk']:
        node_color.append([1,0.5,0])
        node_size.append(50)        
        
#     elif node in walk_transit_final_nodes:

#         node_color.append([1,0,0])
#         node_size.append(200)        
    else: 
        node_color.append([1,1,1]); #[0,0,1])
        node_size.append(1)    
    

for k,edge in enumerate(GRAPHS['walk'].edges):
    edge_found = False;
    if edge_found == False:
        if False: #edge in walk_initial_edges:
            edge_color.append([1,0,0]); 
            edge_width.append(4)
        elif edge in walk_final_edges:
            edge_color.append([0,0,1]); 
            edge_width.append(4)
        elif edge in walk_trip_edges:
            edge_color.append([0,0,0]); 
            edge_width.append(4)
        else: 
            edge_color.append([1,1,1]);
            edge_width.append(1)

fig, ax = ox.plot_graph(GRAPHS['walk'],bgcolor=bgcolor,
                        node_color=node_color,node_edgecolor='w',node_size=node_size,node_zorder=1,
                        edge_color=edge_color, edge_linewidth=edge_width)