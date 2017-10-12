'''
Reference implementation of node2vec.

Original node2vec functions and implementation
 Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016

Modifications for:
Connectome embeddings: A deep learning framework for
mapping higher-order relations between brain structure and function
 Author: Gideon Rosenthal
'''

import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from sklearn.preprocessing import Normalizer
import pickle

def create_embedding(dir_name, input_edge_list, output_embedding, current_dti, current_name,
                     permutation_no=500, lesion_node = 0, dimensions=30, walk_length=20,
                     num_walks=800, window_size=3, iter=1, workers=10, p=0.1, q=1.6, sg=0,
                     weighted=True, directed=False):
    '''

    Args:
        Connectome embedding related attributes
        dir_name: directory name
        input_edge_list:  name of input edge list
        output_embedding: name of output embedding
        current_dti: matrix of current dti to embbed
        current_name: name of the analysis
        permutation_no: how many permutations are needed
        lesion_node: if a lesion node is needed

        node2vec related attributes
        dimensions: dimensions of embeddings
        walk_length: Length of walk per source
        num_walks:Number of walks per source
        window_size : Context size for optimization
        iter : Number of epochs in SGD
        workers : Number of parallel workers
        p: Return hyperparameter
        q: Inout hyperparameter
        sg: skipgram = 1, cbow=0
        weighted:Boolean specifying (un)weighted
        directed:Graph is (un)directed


    Returns:
        word2Vecmodelsorted: word2vec embeddings

    '''
    zero = 1.11324633283e-16
    #creating edge list in the format which is digestible by node2vec
    if lesion_node > 0:

        with open(input_edge_list, 'w') as edge_list:
            for r in range(current_dti.shape[0]):
                for c in range(current_dti.shape[0]):
                    if current_dti[r, c] != 0 and r != lesion_node and c != lesion_node:
                        edge_list.write('%s %s %s \n' % (r, c, current_dti[r, c]))
                    if r == lesion_node or c == lesion_node:
                        edge_list.write('%s %s %s \n' % (r, c, zero))

    else:
        with open(input_edge_list, 'w') as edge_list:
            for r in range(current_dti.shape[0]):
                for c in range(current_dti.shape[0]):
                    if current_dti[r, c] != 0:
                        edge_list.write('%s %s %s \n' % (r, c, current_dti[r, c]))

    # we multiply the num_walks by  permutation_no to save time in calling the functions.
    walks_agg = node2vec_agg_walks(input_edge_list, walk_length=walk_length, num_walks=num_walks * permutation_no,
                                   workers=workers, p=p, q=q, weighted=weighted, directed=directed)
    with open(dir_name + current_name + '_walks_lesion_' + str(lesion_node), 'w') as f:
        pickle.dump(walks_agg, f)
    word2Vecmodelsorted = node2veclearn_agg(walks_agg, output_embedding, num_walks=num_walks,
                                            permutation_no=permutation_no, number_of_nodes=current_dti.shape[0],
                                            dimensions=dimensions, window_size=window_size, iter=iter, workers=workers)

    return word2Vecmodelsorted

def read_graph(input_edge_list, weighted, directed):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input_edge_list, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_edge_list, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, dimensions, window_size, workers, iter, output_embedding, sg=0):
    '''
    Learn embeddings
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=sg,
                     workers=workers, iter=iter)
    model.save(output_embedding + '.embeddings')
    # model.save_word2vec_format(output_embedding + 'word2vecformat.embeddings')

    return model


def normalize_embeddings(word2Vecmodel):
    normalizer = Normalizer(copy=False)

    word2Vecmodelsorted = np.zeros([word2Vecmodel.syn0.shape[0], word2Vecmodel.syn0.shape[1]])
    for i in range(word2Vecmodel.syn0.shape[0]):
        word2Vecmodelsorted[i] = normalizer.fit_transform(word2Vecmodel[str(i)])
    return word2Vecmodelsorted


def node2veclearn(input_edge_list, output_embedding, dimensions=128, walk_length=10, num_walks=10, window_size=10,
                  iter=1, workers=8, p=1, q=1, weighted=True, directed=True, sg=0):
    """Pipeline for representational learning for all nodes in a graph.

    Keyword arguments:
    input_edge_list -- Input graph path
    output_embedding -- Embeddings path
    dimensions -- Number of dimensions (default=128)
    walk-length -- Length of walk per source (default=10)
    num-walks -- Number of walks per source (default=10)
    window-size -- Context size for optimization (default=10)
    iter -- Number of epochs in SGD (default=1)
    workers -- Number of parallel workers (default=8)
    p -- Return hyperparameter (default=1)
    q -- Inout hyperparameter (default=1)
    weighted -- Boolean specifying (un)weighted (default=True)
    directed -- Graph is (un)directed(default=True)

    example -

    working_dir = '/home/lab_users/Downloads/NKI_Rockland/hagmann/'
    input_edge_list = working_dir + 'hagmann_dti_no_ENT_only_positive.txt'
    output_embedding = working_dir + 'hagmann_dti.embeddings'

    node2veclearn(input_edge_list, output_embedding, dimensions = 30, walk_length = 50,  num_walks=400, window_size=3, iter=1, workers=40, p=0.2, q=2.0, weighted=True, directed=True)

    """

    nx_G = read_graph(input_edge_list, weighted, directed)
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    model = learn_embeddings(walks, dimensions, window_size, workers, iter, output_embedding, sg)
    return model


def node2vec_agg_walks(input_edge_list, walk_length=10, num_walks=10, workers=8, p=1, q=1, weighted=True,
                       directed=True):
    """Pipeline for representational learning for all nodes in a graph.

    Keyword arguments:
    input_edge_list -- Input graph path
    walk-length -- Length of walk per source (default=10)
    num-walks -- Number of walks per source (default=10)
    workers -- Number of parallel workers (default=8)
    p -- Return hyperparameter (default=1)
    q -- Inout hyperparameter (default=1)
    weighted -- Boolean specifying (un)weighted (default=True)
    directed -- Graph is (un)directed(default=True)

    example -

    working_dir = '/home/lab_users/Downloads/NKI_Rockland/hagmann/'
    input_edge_list = working_dir + 'hagmann_dti_no_ENT_only_positive.txt'
    output_embedding = working_dir + 'hagmann_dti.embeddings'

    node2veclearn(input_edge_list, output_embedding, dimensions = 30, walk_length = 50,  num_walks=400, window_size=3, iter=1, workers=40, p=0.2, q=2.0, weighted=True, directed=True)

    """

    nx_G = read_graph(input_edge_list, weighted, directed)
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks_parallel(num_walks, walk_length, workers)

    return walks


def node2veclearn_agg(walks, output_embedding, num_walks=10, permutation_no=10, number_of_nodes=83, dimensions=128,
                      window_size=10, iter=1, workers=8, sg=0):
    """Pipeline for representational learning for all nodes in a graph.

    Keyword arguments:
    input_edge_list -- Input graph path
    output_embedding -- Embeddings path
    dimensions -- Number of dimensions (default=128)
    num-walks -- Number of walks per source (default=10)
    permutation_no -- number of permutation (default = 10)
    window-size -- Context size for optimization (default=10)
    iter -- Number of epochs in SGD (default=1)
    workers -- Number of parallel workers (default=8)
    sg -- skipgram = 1, cbow=1
    p -- Return hyperparameter (default=1)
    q -- Inout hyperparameter (default=1)
    weighted -- Boolean specifying (un)weighted (default=True)
    directed -- Graph is (un)directed(default=True)

    example -

    working_dir = '/home/lab_users/Downloads/NKI_Rockland/hagmann/'
    input_edge_list = working_dir + 'hagmann_dti_no_ENT_only_positive.txt'
    output_embedding = working_dir + 'hagmann_dti.embeddings'

    node2veclearn(input_edge_list, output_embedding, dimensions = 30, walk_length = 50,  num_walks=400, window_size=3, iter=1, workers=40, p=0.2, q=2.0, weighted=True, directed=True)

    """

    word2vec_permutations = np.zeros([permutation_no, number_of_nodes, dimensions])
    count = 0
    for permute in range(0, permutation_no * num_walks * number_of_nodes, num_walks * number_of_nodes):
        model = learn_embeddings(walks[permute:permute + num_walks * number_of_nodes], dimensions, window_size, workers,
                                 iter, output_embedding, sg)
        word2Vecmodelsorted = normalize_embeddings(model)

        word2vec_permutations[count, ...] = word2Vecmodelsorted
        count += 1
    return word2vec_permutations


def node2veclearn_update(input_edge_list, org_embedding, new_embedding, dimensions=128, walk_length=10, num_walks=10,
                         window_size=10, iter=1, workers=8, p=1, q=1, weighted=True, directed=True):
    """Pipeline for updating an embedding

    Keyword arguments:
    org_embedding-- original embedging
    new_embedding -- new Embeddings path
    dimensions -- Number of dimensions (default=128)
    walk-length -- Length of walk per source (default=10)
    num-walks -- Number of walks per source (default=10)
    window-size -- Context size for optimization (default=10)
    iter -- Number of epochs in SGD (default=1)
    workers -- Number of parallel workers (default=8)
    p -- Return hyperparameter (default=1)
    q -- Inout hyperparameter (default=1)
    weighted -- Boolean specifying (un)weighted (default=True)
    directed -- Graph is (un)directed(default=True)

    example -

    working_dir = '/home/lab_users/Downloads/NKI_Rockland/hagmann/'
    input_edge_list = working_dir + 'hagmann_dti_no_ENT_only_positive.txt'
    org_embedding = working_dir + 'hagmann_dti.embeddings'
    new_embedding = working_dir + 'hagmann_dti_updated'

    node2veclearn_update(org_embedding, new_embedding,  walk_length = 50,  num_walks=400, p=0.2, q=2.0, weighted=True, directed=True)

    """

    nx_G = read_graph(input_edge_list, weighted, directed)
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)

    model = Word2Vec.load(org_embedding)
    model.train(walks)
    model.save(new_embedding + '.embeddings')
    # model.save_word2vec_format(new_embedding + 'word2vecformat.embeddings')

    return model