import random
from ..matrix import (
    random_int_list,
    random_int_matrix,
    random_symetric_int_matrix,
    random_oriented_int_matrix,
    random_triangular_int_matrix,
)
from ..node import node

class OpenDigraphFactoryMixin:
    """mixin containing factory methods for graph creation"""

    @classmethod
    def empty(cls):
        """
        creates empty graph with no nodes
        
        returns:
            open_digraph: new empty graph
        """
        return cls([], [], [])

    @classmethod
    def from_dot_file(cls, p):
        """
        creates graph from DOT file
        
        args:
            path (str): path to DOT file
            
        returns:
            open_digraph: parsed graph
            
        raises:
            ValueError: if file format is invalid
            
        supports:
            - node labels and attributes
            - edge multiplicities 
            - input/output node markers
        """
        with open(p, "r") as f:
            data = f.read()
        s = data.find("{")
        e = data.rfind("}")
        if s == -1 or e == -1 or s > e:
            raise ValueError("No matching brakets found in DOT file")

        core = data[s+1:e].strip()
        g = cls.empty()
        name_to_id = {}

        parts = [ln.strip() for ln in core.split(";") if ln.strip()]

        for line in parts:
            if "->" in line:
                bracket_part = ""
                if "[" in line and "]" in line:
                    bstart = line.index("[")
                    bend = line.rindex("]")
                    bracket_part = line[bstart+1:bend].strip()
                    line = line[:bstart].strip()

                nodes_list = [x.strip() for x in line.split("->") if x.strip()]
                edge_attrs = {}
                if bracket_part:
                    for a in bracket_part.split(","):
                        kv = a.strip().split("=")
                        if len(kv) == 2:
                            k = kv[0].strip()
                            v = kv[1].strip().strip('"')
                            edge_attrs[k] = v

                for i in range(len(nodes_list) - 1):
                    src_name, tgt_name = nodes_list[i], nodes_list[i+1]
                    if src_name not in name_to_id:
                        name_to_id[src_name] = g.add_node()
                    if tgt_name not in name_to_id:
                        name_to_id[tgt_name] = g.add_node()

                    mult = int(edge_attrs.get('mult', 1))
                    g.add_edges([(name_to_id[src_name], name_to_id[tgt_name])] * mult)


            else:
                bracket_part = ""
                if "[" in line and "]" in line:
                    bstart = line.index("[")
                    bend = line.rindex("]")
                    bracket_part = line[bstart+1[bend].strip()]
                    line = line[:bstart].strip()

                node_name = line.strip()
                node_attrs = {}
                if bracket_part:
                    for a in bracket_part.split(","):
                        kv = a.strip().split("=")
                        if len(kv) == 2:
                            k = kv[0].strip()
                            v = kv[1].strip().strip('"')
                            node_attrs[k] = v

                if node_name in name_to_id:
                    n_id = name_to_id[node_name]
                else:
                    n_id = g.add_node()
                    name_to_id[node_name] = n_id

                if "label" in node_attrs:
                    g.nodes[n_id].label = node_attrs["label"]
                if "input" in node_attrs and node_attrs["input"].lower() in ["true","1"]:
                    g.add_input_id(n_id)
                if "output" in node_attrs and node_attrs["output"].lower() in ["true","1"]:
                    g.add_output_id(n_id)

        return g
    
    def adjacency_matrix(self):
        """
        converts graph to adjacency matrix format
        
        args:
            none
            
        returns:
            list[list[int]]: matrix where entry [i][j] is number of edges i->j
            
        notes:
            - ignores input/output nodes
            - uses node_to_index() for consistent ordering
        """
        n = len(self.nodes)
        m = self.node_to_index()
        a = [[0] * n for _ in range(n)]

        for u, v in self.nodes.items():
            i = m[u]  
            for w, x in v.children.items():
                j = m[w]  
                a[i][j] = x 

        return a

    @classmethod
    def graph_from_adjacency_matrix(cls, matrix):
        """
        converts adjacency matrix to graph
        
        args:
            matrix (list[list[int]]): adjacency matrix with edge counts
            
        returns:
            open_digraph: equivalent graph
            
        raises:
            ValueError: if matrix is not square
        """
        n = len(matrix)
        if not all(len(row) == n for row in matrix):
            raise ValueError("Matrix must be square")
            
        g = cls.empty()
        nodes = [g.add_node() for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    g.add_edge(nodes[i], nodes[j], matrix[i][j])
                    
        return g

    @classmethod
    def random_graph(cls, n, bound=2, inputs=0, outputs=0, form="free"):
        """
        generates random graph with given parameters
        
        args:
            n (int): number of nodes
            bound (int): maximum edge multiplicity
            inputs (int): number of input nodes
            outputs (int): number of output nodes
            form (str): graph type constraint:
                "free" - no constraints
                "DAG" - directed acyclic
                "oriented" - max one edge between nodes
                "loop-free" - no self-loops
                "undirected" - symmetric edges
                "loop-free undirected" - symmetric without loops
                
        returns:
            open_digraph: random graph meeting constraints
        """
        if inputs + outputs > n:
            raise ValueError("Too many inputs/outputs for node count")
            
        matrix_types = {
            "free": lambda: random_int_matrix(n, bound),
            "DAG": lambda: random_triangular_int_matrix(n, bound),
            "oriented": lambda: random_oriented_int_matrix(n, bound),
            "loop-free": lambda: random_int_matrix(n, bound, null_diag=True),
            "undirected": lambda: random_symetric_int_matrix(n, bound),
            "loop-free undirected": lambda: random_symetric_int_matrix(n, bound, null_diag=True)
        }
        
        if form not in matrix_types:
            raise ValueError(f"Unknown graph form: {form}")
            
        g = cls.graph_from_adjacency_matrix(matrix_types[form]())
        
        nodes = g.get_nodes_id()
        random.shuffle(nodes)
        for i in range(inputs):
            g.add_input_node(nodes[i])
        for i in range(outputs):
            g.add_output_node(nodes[inputs + i])
            
        return g