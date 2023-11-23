import dgl
import torch as th
import torch.nn.functional as F
import numpy as np
from graph.graph import GraphLoader
from gnn.spangraph import SpanGraph

def performance(pred, labels, acc=None):
    tp = 0; fp = 0; tn = 0; fn = 0
    for i in range(len(labels)):
        tp += 1 if pred[i] == 1 and labels[i] == 1 else 0
        fp += 1 if pred[i] == 1 and labels[i] == 0 else 0
        tn += 1 if pred[i] == 0 and labels[i] == 0 else 0
        fn += 1 if pred[i] == 0 and labels[i] == 1 else 0
    print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)

    acc = float((tp + tn) / (tp + tn + fp + fn)) if acc == None else acc
    precision = 0
    if tp + fp != 0:
        precision = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    tnr = float(tn / (tn + fp))
    tpr = float(tp / (tp + fn))
    f1 = 0
    if precision + recall != 0:
        f1 = 2 * float(precision *  recall / (precision + recall))
    return precision, recall, tnr, tpr, f1

def eval_model_merge(args):
    gloader = GraphLoader()
    ### load the training graph
    test_g, test_g_nf, test_g_ef, test_g_e_label, test_edges= gloader.load_graph_merge(args)
    test_g_id_node_map = gloader.load_node_edge_map(args)
    test_g.ndata['nf'], test_g.edata['ef'], test_g.edata['e_label'] = test_g_nf, test_g_ef, test_g_e_label

    args.db_name = 'rank_trainticket'
    full_g, full_g_nf, full_g_ef, full_g_e_label, full_edges = gloader.load_graph_merge(args)
    full_g_id_node_map = gloader.load_node_edge_map(args)
    full_g.ndata['nf'], full_g.edata['ef'], full_g.edata['e_label'] = full_g_nf, full_g_ef, full_g_e_label

    test_g_node_id_map = {v: k for k, v in test_g_id_node_map.items()}

    new_node_feats = [];new_edge_src_nids = []; new_edge_dst_nids = [];
    node_id_map = {}
    cur_nid = full_g.number_of_nodes();
    num_of_node_before_adding = full_g.number_of_nodes(); num_of_edge_before_adding = full_g.number_of_edges()
    for node in test_g_id_node_map:
        if node in full_g_id_node_map:
            node_id_map[node] = full_g_id_node_map[node]
        else:
            node_id_map[node] = cur_nid
            new_node_feats.append(test_g.ndata['nf'][test_g_id_node_map[node]].clone().detach())
            cur_nid += 1
    src_nodes = [e[0] for e in test_edges]
    dst_nodes = [e[1] for e in test_edges]

    new_edge_src_nids = [node_id_map[test_g_node_id_map[src]] for src in src_nodes]
    new_edge_dst_nids = [node_id_map[test_g_node_id_map[dst]] for dst in dst_nodes]

    if len(new_node_feats) > 0:
        full_g.add_nodes(len(new_node_feats))
        print(full_g.number_of_nodes())
        full_g.ndata['nf'][num_of_node_before_adding: full_g.number_of_nodes()] = th.stack(new_node_feats)

    if len(new_edge_src_nids) > 0:
        full_g.add_edges(new_edge_src_nids, new_edge_dst_nids)
        full_g.edata['ef'][num_of_edge_before_adding : full_g.number_of_edges()] = test_g_ef
        full_g.edata['e_label'][num_of_edge_before_adding : full_g.number_of_edges()] = test_g_e_label


    nf = full_g.ndata.pop('nf')
    ef = full_g.edata.pop('ef')
    e_label = full_g.edata['e_label']
    test_mask = np.zeros(e_label.shape[0])
    test_mask[num_of_edge_before_adding:] = 1
    test_mask = th.BoolTensor(test_mask)

    n_classes = 2
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    best_model = WTAGNN(full_g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes,
                        args.n_layers,  F.relu, args.dropout)
    best_model.load_state_dict(th.load('./output/best.model.' + args.model_name))
    acc, predictions, labels = evaluate(best_model, full_g, nf, ef, e_label, test_mask)
    precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)
    print("accuracy {:.4f}".format(acc))
    print("precision {:.4f}".format(precision))
    print("recall {:.4f}".format(recall))
    print("tnr {:.4f}".format(tnr))
    print("tpr {:.4f}".format(tpr))
    print("f1 {:.4f}".format(f1))
    print('acc/pre/rec: ', str("{:.2f}".format(acc* 100) ) + '%/' + str("{:.2f}".format(precision* 100) ) + '%/' +
          str("{:.2f}".format(recall* 100) ) + '%')

def evaluate(model, g, nf, ef, labels, mask):
    model.eval()
    with th.no_grad(): 
        n_logits, e_logits = model(nf, ef)
        e_logits = e_logits[mask]
        labels = labels[mask]
        _, indices = th.max(e_logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), indices, labels



    
