import time
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


from gnn.eval import performance, evaluate
from gnn.spangraph import SpanGraph
from graph.graph import GraphLoader

def start_train(args):
    gloader = GraphLoader()
    g, nf, ef, e_label, train_mask, test_mask, val_mask = gloader.load_graph(args)  

    # g = g.to('cuda:0')
    n_classes = 2
    n_edges = g.number_of_edges()
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    print('\n************initilize model************')
    model = SpanGraph(g, input_node_feat_size, input_edge_feat_size,
                   args.n_hidden, n_classes, args.n_layers,  F.relu, args.dropout)
    print(model)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        print('using gpu:', args.gpu)
        th.cuda.empty_cache()
        th.cuda.set_device(args.gpu)
        nf, ef, e_label = nf.cuda(), ef.cuda(), e_label.cuda()
        train_mask, val_mask, test_mask = train_mask.cuda(), val_mask.cuda(), test_mask.cuda()
        model.cuda()

    loss_fcn = th.nn.CrossEntropyLoss()
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('\n************start training************')
    dur, max_acc = [], -1
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        n_logits, e_logits = model(nf, ef)
        loss = loss_fcn(e_logits[train_mask], e_label[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc, predictions, labels = evaluate(model, g, nf, ef, e_label, val_mask)
        precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)

        if f1 > max_acc:
            max_acc = f1
            th.save(model.state_dict(), './output/best.model.' + args.model_name)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             f1, n_edges / np.mean(dur) / 1000))

    best_model = SpanGraph(g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes,
                args.n_layers,  F.relu, args.dropout)
    if cuda:
        best_model.cuda()
    best_model.load_state_dict(th.load( './output/best.model.' + args.model_name))
    acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)
    precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)

    print("accuracy {:.4f}".format(acc))
    print("precision {:.4f}".format(precision))
    print("recall {:.4f}".format(recall))
    print("tnr {:.4f}".format(tnr))
    print("tpr {:.4f}".format(tpr))
    print("f1 {:.4f}".format(f1))
    print('acc/pre/rec: ', str("{:.2f}".format(acc* 100) ) + '%/' + str("{:.2f}".format(precision* 100) ) + '%/' +
          str("{:.2f}".format(recall* 100) ) + '%')

def start_train_cv(args):
    gloader = GraphLoader()
    g, nf, ef, e_label, _, _, _ = gloader.load_graph(args)  
    g = g.to('cuda:0')
    n_classes = 2
    n_edges = g.number_of_edges()
    input_node_feat_size, input_edge_feat_size = nf.shape[1], ef.shape[1]

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        print('using gpu:', args.gpu)
        th.cuda.empty_cache()
        th.cuda.set_device(args.gpu)
        nf, ef = nf.cuda(), ef.cuda()

    print('\n************start training for {:d} folds************'.format(args.fold))
    kf = StratifiedKFold(n_splits=args.fold, shuffle=True)
    kf.get_n_splits()
    print(kf)
    fold = 0
    total_precision = total_acc = total_recall = 0
    for train_index, test_index in kf.split(e_label, e_label):
        fold += 1
        print('\nfold #: ', str(fold))
        train_mask = np.zeros(e_label.shape[0])
        train_mask[train_index] = 1
        train_mask = th.BoolTensor(train_mask)

        test_mask= np.zeros(e_label.shape[0])
        test_mask[test_index] = 1
        test_mask = th.BoolTensor(test_mask)

        model = SpanGraph(g, input_node_feat_size, input_edge_feat_size, args.n_hidden,
                       n_classes, args.n_layers,  F.relu, args.dropout)
        print(model)

        if cuda:
            e_label = e_label.cuda()
            train_mask = train_mask.cuda()
            test_mask = test_mask.cuda()
            model.cuda()

        loss_fcn = th.nn.CrossEntropyLoss()

        optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        dur = []; max_acc = -1
        for epoch in range(args.n_epochs):
            model.train()
            if epoch >= 3: t0 = time.time()
            # forward
            n_logits, e_logits = model(nf, ef)

            loss = loss_fcn(e_logits[train_mask], e_label[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc, predictions, labels = evaluate(model, g, nf, ef, e_label, test_mask)
            precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)

            if f1 > max_acc:
                max_acc = f1
                th.save(model.state_dict(), './output/best.model.' + args.model_name + '.fold.' + str(fold))

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                f1, n_edges / np.mean(dur) / 1000))

        best_model = SpanGraph(g, input_node_feat_size, input_edge_feat_size, args.n_hidden, n_classes,
                            args.n_layers,  F.relu, args.dropout)
        if cuda:
            best_model.cuda()
        best_model.load_state_dict(th.load('./output/best.model.' + args.model_name + '.fold.' + str(fold)))

        acc, predictions, labels = evaluate(best_model, g, nf, ef, e_label, test_mask)

        precision, recall, tnr, tpr, f1 = performance(predictions.tolist(), labels.tolist(), acc)
        print("accuracy {:.4f}".format(acc))
        print("precision {:.4f}".format(precision))
        print("recall {:.4f}".format(recall))
        print("tnr {:.4f}".format(tnr))
        print("tpr {:.4f}".format(tpr))
        print("f1 {:.4f}".format(f1))


        total_precision += precision
        total_acc += acc
        total_recall += recall

    print('\n************training done! Averaged model performance************')
    print('acc/pre/rec/f1: ', str("{:.2f}".format(total_acc/ args.fold* 100) ) + '%/' 
                         + str("{:.2f}".format(total_precision / args.fold* 100) ) + '%/' +
                           str("{:.2f}".format(total_recall/args.fold* 100) ) + '%' +
                            str("{:.2f}".format(2 * 100 * float((total_precision / args.fold) *  (total_recall/args.fold) / (total_precision / args.fold + total_recall/args.fold))) ) + '%')