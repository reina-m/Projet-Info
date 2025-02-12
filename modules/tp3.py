import random
from open_digraph import *

def random_int_list(n, bound):
    return [random.randrange(0, bound) for _ in range(n)]

def random_int_matrix(n, bound, null_diag=True, number_generator=(lambda : random.random())):
    res = []
    for i in range(n):
        line = []
        for j in range(n):
            line.append(int(bound*number_generator()))
        res.append(line)
        if null_diag:
            res[i][i] = 0
    return res

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
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            res[i][j] = random.randrange(0, bound+1)
    return res 




class Graph:
    def graph_from_adjacency_matrix(self, matrix):
        graph = Graph()
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

class Graph:
    def node_to_index(self):
        return {node: idx for idx, node in enumerate(sorted(self.nodes))}


class Graph:
    def adjacency_matrix(self):
        n = len(self.nodes)
        index_map = self.node_to_index()
        matrix = [[0] * n for _ in range(n)]
        for src, dest in self.edges:
            i = index_map[src]
            j = index_map[dest]
            matrix[i][j] += 1
        return matrix



