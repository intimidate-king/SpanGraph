import random
import dgl
import torch as th
import numpy as np
import pickle

class GraphLoader:
    def __init__(self):
        pass


    def load_node_edge_map(self, args):
        db_name = args.db_name if args.db_name.endswith('/') else args.db_name + '/'
        id_node_map_path = './data/' + db_name  + '_id_node_map.pickle'
        with open(id_node_map_path, 'rb') as f:  id_node_map = pickle.load(f)

        return id_node_map


    def load_graph_merge(self, args):
        print('\n************loading the specified graph and feature data************')

        edgelist_path = './data/' + args.db_name + '/graph.edgelist'
        with open(edgelist_path, 'r') as f:
            edges = [tuple(line.strip().split(',')) for line in f]
        edges = [tuple((int(e[0]), int(e[1]))) for e in edges]
        src_nodes = [e[0] for e in edges]
        dst_nodes = [e[1] for e in edges]

        g = dgl.DGLGraph()
        g.add_nodes(len(set(src_nodes).union(set(dst_nodes))))
        g.add_edges(src_nodes, dst_nodes)
        print('Loaded graph: ', args.db_name, g,'\n')

        nf = np.load('./data/'  + args.db_name + '/node_feat.npy')
        nf = th.from_numpy(nf).float()
        print('node feature shape:', nf.shape)

        ef = np.load('./data/'  + args.db_name + '/edge_feat.npy')
        ef = th.from_numpy(ef).float()
        print('edge feature shape:', ef.shape)

        e_label = np.load('./data/'  + args.db_name + '/edge_label.npy').tolist()
        e_label = th.tensor(e_label)
        print('edge labels shape:', e_label.shape)

        print('***************************loading completed***************************\n')
        return g, nf, ef, e_label, edges

    def load_graph(self, args):
        print('\n************loading the specified graph and feature data************')
        edgelist_path = './data/' + args.db_name + '/graph.edgelist'
        with open(edgelist_path, 'r') as f:
            edges = [tuple(line.strip().split(',')) for line in f]

        edges = [tuple((int(e[0]), int(e[1]))) for e in edges]
        src_nodes = [e[0] for e in edges]
        dst_nodes = [e[1] for e in edges]

        g = dgl.DGLGraph()
        g.add_nodes(len(set(src_nodes).union(set(dst_nodes))))
        g.add_edges(src_nodes, dst_nodes)
        print('Loaded graph: ', args.db_name, g,'\n')

        nf = np.load('./data/'  + args.db_name + '/node_feat.npy')
        nf = th.from_numpy(nf).float()
        print('node feature shape:', nf.shape)

        ef = np.load('./data/'  + args.db_name + '/edge_feat.npy')
        ef = th.from_numpy(ef).float()
        print('edge feature shape:', ef.shape)

        e_label = np.load('./data/'  + args.db_name + '/edge_label.npy').tolist()
        e_label = th.tensor(e_label)
        print('edge labels shape:', e_label.shape)

        train_mask, test_mask, val_mask = self._split_dataset(e_label, (args.r_train, args.r_test, args.r_val))

        print('***************************loading completed***************************\n')
        return g, nf, ef, e_label, train_mask, test_mask, val_mask

    def _split_dataset(self, labels, ratio_tuple):   
        shuffle_list = [i for i in range(labels.shape[0])]
        random.shuffle(shuffle_list)
        train_ct = int(len(shuffle_list) * ratio_tuple[0])
        test_ct =  int(len(shuffle_list) * ratio_tuple[1])
        val_ct =   int(len(shuffle_list) * ratio_tuple[2])
        print('# of train edge:', train_ct, '   # of test edge:', test_ct, ' # of val edge:', val_ct)

        train_mask = np.zeros(labels.shape[0])
        test_mask = np.zeros(labels.shape[0])
        val_mask = np.zeros(labels.shape[0])
        for idx in range(0, train_ct):
            train_mask[shuffle_list[idx]] = 1
        for idx in range(train_ct, train_ct + test_ct):
            test_mask[shuffle_list[idx]] = 1
        for idx in range(len(shuffle_list) - val_ct, len(shuffle_list)):
            val_mask[shuffle_list[idx]] = 1

        train_mask = th.BoolTensor(train_mask)
        test_mask = th.BoolTensor(test_mask)
        val_mask = th.BoolTensor(val_mask)

        return train_mask, test_mask, val_mask

