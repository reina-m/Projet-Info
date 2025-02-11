import random
from modules.open_digraph import *

def random_int_list(n, bound):
    return [random.randrange(0, bound) for _ in range(n)]

def random_int_matrix(n, bound, null_diag=True):
    matrix = [[random.randrange(0, bound) for _ in range(n)] for _ in range(n)]
    if null_diag:
        for i in range(n):
            matrix[i][i] = 0
    return matrix

def random_symetric_int_matrix(n, bound, null_diag = True):
    res = random_int_matrix(n, bound, null_diag)
    for i in range(n):
        for j in range(i + 1, n):
            res[j][i] = res[i][j]      
    return res

def random_oriented_int_matrix(n ,bound, null_diag=True):
    res = random_int_matrix(n,bound,null_diag)
    for i in range(n):
        for j in range(i + 1,n):
            if random.randrange(0, 2) == 1: 
                res[i][j] = 0
            else:
                res[j][i] = 0
    return res

def random_triangular_int_matrix(n ,bound, null_diag = True):
    res = random_int_matrix(n,1,null_diag)
    for i in range(n):
        for j in range(i + 1, n):
            x = random.randrange(0, bound)
            res[i][j] = x
    return res 




class Graph:
    def graph_from_adjacency_matrix(self, matrix):
        graph = self.empty()
        n=len(matrix)
        for i in range(n):
            graph.add_node(i)
        for x in range(n):
            for y in range(n):
                for _ in range(matrix[x][y]):
                    graph.add_edge(x,y)
        return graph

    def random_graph(self, n, bound, inputs=0, outputs= 0, form = "free"):
        if form == "free":
            matrix = random_int_matrix(n, bound)
        elif form == "DAG":
            matrix = random_triangular_int_matrix(n, bound)
        elif form == "oriented":
            matrix = random_oriented_int_matrix(n, bound)
        elif form == "loop-free":
            matrix = random_int_matrix(n, bound, null_diag=True)
        elif form == "undirected":
            matrix = random_symetric_int_matrix(n, bound, null_diag=False)
        elif form == "loop-free undirected":
            matrix = random_symetric_int_matrix(n, bound, null_diag=True)
        else:
            raise ValueError("Graphe inconnue")
        
        graph = self.graph_from_adjacency_matrix(matrix)
        nodes = list(graph.nodes)
        if inputs > len(nodes) or outputs > len(nodes):
            raise ValueError("depasse le nombre de noeuds")
        
        graph.inputs = random.sample(nodes, inputs)
        graph.outputs = random.sample(nodes, outputs)
        return graph