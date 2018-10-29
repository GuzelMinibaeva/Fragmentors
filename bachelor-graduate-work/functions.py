import pandas as pd
import networkx as nx
import numpy as np
import math as mt
from itertools import product
from collections import defaultdict
from scipy.sparse.linalg import expm_multiply
from timeit import default_timer as timer
from operator import eq
import pickle


def is_psd(matrix, tol=1e-8):
    """
        Checks if given matrix is positive semi-definite.
    """

    matrix = np.array(matrix)
    eigen_values = np.linalg.eig(matrix)

    return np.all(eigen_values > -tol), eigen_values


def adjacency_matrix(graph1, graph2):
    """
        Builds an adjacency matrix for tensor product of given graphs.
    """

    product = nx.tensor_product(graph1, graph2)

    for n, m in list(product.nodes(data=True)):
        if not eq(*m['element']):  # '*' is for unpacking, 'eq' is for checking equivalence of tuple elements
            product.remove_node(n)
    for i, j, datadict in list(product.edges(data=True)):
        if not (eq(*datadict.get('s_bond', (True, True))) and eq(*datadict.get('p_bond', (True, True)))):
            product.remove_edge(i, j)

    matrix = nx.adjacency_matrix(product)

    return matrix


def structural_kernel(matrix, lam):
    """
        Calculates the structural kernel for two graphs using a given matrix of graph's tensor product
        with lam as a decaying factor.
    """

    matrix = matrix.tocsc()
    e = np.ones(matrix.shape[1])
    kernel = np.dot(np.transpose(e), expm_multiply(lam * matrix, e))

    return kernel


def conditional_kernel(reaction1, reaction2, gam, dictionary):
    """
        Calculates the conditional kernel between two reactions (graph meta of CGR).
    """

    list_con1 = [float(dictionary[reaction1]['temperature_1/K']), float(dictionary[reaction1]['amount.1'])]
    list_con1.extend(dictionary[reaction1]['additive.1'])
    list_con1 = np.array(list_con1)

    list_con2 = [float(dictionary[reaction1]['temperature_1/K']), float(dictionary[reaction1]['amount.1'])]
    list_con2.extend(dictionary[reaction2]['additive.1'])
    list_con2 = np.array(list_con2)

    kernel = mt.exp(- gam * (np.dot((list_con1 - list_con2), np.transpose(list_con1 - list_con2))))

    return kernel


def structural_kernel_matrices(graph_list, list_lam, dumping=False):
    """
        Returns a dictionary where lambda (decaying factor) is a key, value - matrix of structural kernels
        corresponding to lam.
    """

    dict_of_matrices_exp = {}
    start = timer()
    time_init = 0
    time_struct = 0
    tmp_struct = defaultdict(lambda: defaultdict(dict))
    cnt = 0

    for (num1, graph1), (num2, graph2) in product(enumerate(graph_list), repeat=2):
        start1 = timer()
        matrix = adjacency_matrix(graph1, graph2)
        end1 = timer()
        time_init = time_init + (end1 - start1)

        for lam in list_lam:
            start2 = timer()
            tmp_struct[lam][num1][num2] = structural_kernel(matrix, lam)
            end2 = timer()
            time_struct = time_struct + (end2 - start2)

        if num2 == len(graph_list) - 1:
            cnt += 1
            print('{} / {} done'.format(cnt, len(graph_list)))

    for lam in list_lam:
        matrix_struct = pd.DataFrame(tmp_struct[lam])
        dict_of_matrices_exp[lam] = matrix_struct

        if dumping:

            file_name = './dumps/structure_kernel_dict(lam={}); {} reactions'.format(lam, len(graph_list))
            obj = matrix_struct
            output = open(file_name, 'wb')
            pickle.dump(obj, output)
            output.close()
    end = timer()
    time_total = end - start
    print('\nTime init = {} \nTime of struct kernel calc = {} \nTotal time = {}'.format(time_init, time_struct, time_total))

    return dict_of_matrices_exp


def dict_of_kernel_struct_con_matrices(list_lam, list_gam, graph_list, conditions_list, dumping=False):
    """
        Returns a 3D-dictionary where keys (lambda and gamma) give a matrix of kernels with corresponding parameter's values.
    """

    dict_of_matrices_struct_con = defaultdict(dict)
    start = timer()
    tmp_struct_con = defaultdict(lambda: defaultdict(dict))

    for lam in list_lam:
        input = open('./dumps/structure_kernel_dict(lam={}); {} reactions'.format(lam, len(graph_list)), 'rb')
        matrix_kernel_structural = pickle.load(input)
        input.close()

        for gam in list_gam:
            for num1 in range(len(conditions_list)):
                for num2 in range(len(conditions_list)):
                    tmp_struct_con[gam][num1][num2] = matrix_kernel_structural.at[num1, num2] \
                                                      * conditional_kernel(num1, num2, gam, conditions_list)
            matrix_struct_con = pd.DataFrame(tmp_struct_con[gam])
            dict_of_matrices_struct_con[lam][gam] = matrix_struct_con

    if dumping:

        file_name = './dumps/structure_conditions_kernel_dict ({} reactions)'.format(len(graph_list))
        obj = dict_of_matrices_struct_con
        output = open(file_name, 'wb')
        pickle.dump(obj, output)
        output.close()
    end = timer()
    time_total = end - start
    print('\nTotal time of structural-conditional kernel = {}'.format(time_total))

    return dict_of_matrices_struct_con
