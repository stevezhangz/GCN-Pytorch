import os
import numpy as np


def label2int(label):
    SET=set(label)
    mapping = {w: i for i, w in enumerate(SET)}
    intlabel=np.array(list(map(mapping.get,label)))
    return intlabel,mapping

def label2onehot(label):
    # get all element
    SET=set(label)
    identity=np.identity(len(SET))
    mapping = {w: identity[i] for i, w in enumerate(SET)}
    onehot_label=list(map(mapping.get,label))
    return onehot_label,mapping

def build_adj(num_nodes,edges,use_identity=False,normalize=False):
    adj=np.zeros(shape=(num_nodes,num_nodes))
    for n1,n2 in edges:
        adj[n1][n2]=1
    if use_identity:
        adj=adj+np.identity(num_nodes)
    if normalize:
        adj=adj/np.max(adj)
    degree=np.sum(adj,axis=0)
    degree_matrix=np.identity(degree.shape[0])
    for i in range(degree.shape[0]):
        degree_matrix[i][i]=degree[i]
    return adj,degree

def load_cora(datapath,use_identity=True,normalize=True,one_hot_label=False, adj_process=False):
    edges=np.genfromtxt(os.path.join(datapath,'cora.cites'),dtype=np.dtype(int))
    feature_label=np.genfromtxt(os.path.join(datapath,'cora.content'),dtype=np.dtype(str))
    node_index=np.array(feature_label[:,0]).astype(np.int)
    index,index_mapping=label2int(node_index)
    edges=[[index_mapping[n1],index_mapping[n2]] for n1,n2 in edges]
    feature=np.array(feature_label[:,1:-1]).astype(np.float32)
    label=feature_label[:,-1]
    if one_hot_label:
        label,_=label2onehot(label)
    else:
        label,_=label2int(label)
    max_index=np.max(edges)
    adj,degree=build_adj(max_index+1,edges,use_identity=use_identity,normalize=normalize)
    if adj_process==True:
        adj=np.matmul(np.matmul(degree**0.5,adj),degree**0.5)
    return feature,label,adj,degree,_

