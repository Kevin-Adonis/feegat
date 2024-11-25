
import os
import numpy as np
import networkx as nx
import math
from top import singleton
import random


class Topology(object):
    def __init__(self, config, data_dir='./data/'):
        singleton.data_dir = data_dir
        singleton.topology_name = config.topology_file
        singleton.topology_file = data_dir + config.topology_file

        self.res_link_loads_file = data_dir + config.topology_file + '_link_loads_file'
        self.res_solutions_file = data_dir + config.topology_file + '_solutions_file'

        singleton.shortest_paths_file = singleton.topology_file +'_shortest_paths'
        singleton.Kshortest_paths_file = singleton.topology_file +'_Kshortest_paths'
        singleton.DG = nx.DiGraph()

        self.load_topology()
        self.calculate_paths()

    def load_topology(self):
        print('[*] Loading topology...', singleton.topology_file)
        f = open(singleton.topology_file, 'r')
        header = f.readline()
        singleton.num_nodes = int(header[header.find(':')+2:header.find('\t')])
        singleton.num_links = int(header[header.find(':', 10)+2:])
        f.readline()
        singleton.idx2edge = {}
        singleton.edge2idx = {}
        singleton.link_capacities = np.empty((singleton.num_links))
        singleton.link_weights = np.empty((singleton.num_links))

        singleton.link_capacities_map = {}
        singleton.link_weights_map = {}

        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link

            sd = (int(s),int(d))

            singleton.idx2edge[int(i)] = (int(s),int(d))
            singleton.edge2idx[(int(s),int(d))] = int(i)
            singleton.link_capacities[int(i)] = float(c)
            singleton.link_weights[int(i)] = int(w)
            singleton.DG.add_edges_from([(int(s),int(d),{'weight':int(w),'cap':float(c)})])

            singleton.link_capacities_map[sd] = float(c)
            singleton.link_weights_map[sd] = int(w)
        
        assert len(singleton.DG.nodes()) == singleton.num_nodes and len(singleton.DG.edges()) == singleton.num_links

        f.close()


    
    def calculate_paths(self):
        singleton.idx2sd = []
        singleton.sd2idx = {}
        # Shortest paths
        singleton.shortest_paths = []


        singleton.sdi2edge = {}
        singleton.edge2sdi = {}

        singleton.sd2edge = {}
        singleton.edge2sd = {}


        if 0 and os.path.exists(singleton.Kshortest_paths_file) and os.path.exists(singleton.shortest_paths_file):
            print('[*] Loading shortest paths...', singleton.shortest_paths_file)
            f = open(singleton.shortest_paths_file, 'r')
            f1 = open(singleton.Kshortest_paths_file, 'r')
            singleton.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>')+1:])
                singleton.idx2sd.append((s,d))
                singleton.sd2idx[(s,d)] = singleton.num_pairs
                singleton.num_pairs += 1
                singleton.shortest_paths.append([])
                paths = line[line.find(':')+1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    singleton.shortest_paths[-1].append(node_path)
                    paths = paths[idx+3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(singleton.shortest_paths_file, 'w+')
            f1 = open(singleton.Kshortest_paths_file, 'w+')
            singleton.num_pairs = 0

            for s in range(singleton.num_nodes):
                for d in range(singleton.num_nodes):
                    if s != d:
                        singleton.idx2sd.append((s,d))
                        singleton.sd2idx[(s,d)] = singleton.num_pairs
                        singleton.num_pairs += 1
                        singleton.shortest_paths.append(list(nx.all_shortest_paths(singleton.DG, s, d, weight='weight')))
                        line = str(s)+'->'+str(d)+': '+str(singleton.shortest_paths[-1])
                        f.writelines(line+'\n')

                        #计算最短的K条路径
                        paths = list(nx.shortest_simple_paths(singleton.DG, s, d, weight='weight')) 

                        self.path_pad(paths,s,d)
                        
                        k,i = singleton.K,0

                        singleton.sd2edge[(s,d)] = []
                        for path in paths[:k]:
                            edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
                            singleton.sdi2edge[(s,d,i)] = edges
                            singleton.sd2edge[(s,d)] += edges
                            i += 1

                    
                        line1 = str(s)+'->'+str(d)+': '+str(paths[:k])
                        f1.writelines(line1+'\n')
        
            for key in singleton.sdi2edge:
                for edge in singleton.sdi2edge[key]:
                    singleton.edge2sdi.setdefault(edge,[])
                    singleton.edge2sdi[edge].append(key)

            for key in singleton.sd2edge:
                for edge in singleton.sd2edge[key]:
                    singleton.edge2sd.setdefault(edge,[])
                    singleton.edge2sd[edge].append(key)
                
  
        assert singleton.num_pairs == singleton.num_nodes*(singleton.num_nodes-1)
        f.close()
        f1.close()
        
        
        print('pairs: %d, nodes: %d, links: %d\n'\
                %(singleton.num_pairs, singleton.num_nodes, singleton.num_links))

        self.init_GAT()
    
    # 根据K补齐路径
    def path_pad(self,paths,s,d):
        
        k = singleton.sd2idx[(s,d)]
        singleton.sdi2path.setdefault(k,[])

        for i in range(min(len(paths),singleton.K)):
            singleton.sdi2path[k].append(i)

        i = 0
        while len(paths)<singleton.K:
            singleton.sdi2path[k].append(i)
            paths.append(paths[i])
            i = i + 1

    def init_GAT(self):

        singleton.rand = [list(range(singleton.K)) for i in range(singleton.num_pairs)]
        for line in singleton.rand:
            random.shuffle(line)

        #print('singleton.rand',singleton.rand)

        singleton.edge_adj = {}
        singleton.adjee = np.zeros((singleton.num_links,singleton.num_links),int)
        for edge in singleton.DG.edges():
            a,b = edge
            singleton.edge_adj.setdefault(edge,[])
            adj_edges = list(singleton.DG.in_edges(a)) + list(singleton.DG.out_edges(b))
            while (b,a) in adj_edges:
                adj_edges.remove((b,a))
            singleton.edge_adj[edge] = adj_edges
            for e in adj_edges:
                singleton.adjee[singleton.edge2idx[edge]][singleton.edge2idx[e]] = 1

        num_sd = singleton.num_nodes*(singleton.num_nodes-1)
        num_ed = singleton.num_links

        singleton.adjfe = np.zeros((num_sd,num_ed),int)
        singleton.adjef = np.zeros((num_ed,num_sd),int)

        for sd in singleton.sd2edge:
            for ed in singleton.sd2edge[sd]:
                singleton.adjfe[singleton.sd2idx[sd]][singleton.edge2idx[ed]] = 1

        for ed in singleton.edge2sd:
            for sd in singleton.edge2sd[ed]:
                singleton.adjef[singleton.edge2idx[ed]][singleton.sd2idx[sd]] = 1
        


        
        

class Traffic(object):
    def __init__(self, config, num_nodes, data_dir='./data/', is_training=False):
        if is_training:
            singleton.traffic_file = data_dir + config.topology_file + config.traffic_file
        else:
            singleton.traffic_file = data_dir + config.topology_file + config.test_traffic_file
        singleton.num_nodes = num_nodes
        self.load_traffic(config)
        self.make_inputs()

    def load_traffic(self, config):
        assert os.path.exists(singleton.traffic_file)
        print('[*] Loading traffic matrices...', singleton.traffic_file)
        f = open(singleton.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == singleton.num_nodes*singleton.num_nodes
            matrix = np.zeros((singleton.num_nodes, singleton.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v/singleton.num_nodes)
                j = v%singleton.num_nodes
                if i != j:
                    matrix[i][j] = float(volumes[v])
            #print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        singleton.traffic_matrices = np.array(traffic_matrices)

        tms_shape = singleton.traffic_matrices.shape
        singleton.tm_cnt = tms_shape[0]

        #消除0
        singleton.traffic_matrices = np.clip(singleton.traffic_matrices, 10, np.inf)

        print('Traffic matrices dims: [%d, %d, %d]\n'%(tms_shape[0], tms_shape[1], tms_shape[2]))

    def make_inputs(self):

        for idx in range(0,len(singleton.traffic_matrices)):
            tmp = []
            for sd in singleton.idx2sd:
                s,d = sd
                tmp.append(singleton.traffic_matrices[idx][s][d])
            singleton.tm2demands.append(tmp)
            singleton.tm2demands_log.append(np.log(tmp)/np.log(100000000))

class Environment(object):
    def __init__(self, config, is_training=False):
        singleton.data_dir = './data/'
        singleton.topology = Topology(config, singleton.data_dir)
        singleton.traffic = Traffic(config, singleton.num_nodes, singleton.data_dir, is_training=is_training)
        singleton.traffic_matrices = singleton.traffic_matrices*100*8/300/1000    #kbps
     
        singleton.shortest_paths_node = singleton.shortest_paths         
        singleton.shortest_paths_link = self.convert_to_edge_path(singleton.shortest_paths_node)  # paths consist of links

  
    def convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len-1):
                    e = singleton.edge2idx[(node_paths[i][j][n], node_paths[i][j][n+1])]
                    assert e>=0 and e<singleton.num_links
                    edge_paths[i][j].append(e)
                #print(i, j, edge_paths[i][j])

        return edge_paths
    
    
