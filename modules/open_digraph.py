import os
import copy
import webbrowser
import tempfile
import random


class node:
    def __init__(self, identity, label, parents, children):
        '''
        identity : int; its unique id in the graph
        label : string
        parents : int->int dict; maps a parent node's id to its multiplicity
        children : int->int; maps a child node's id to its multiplicity
        '''
        self.id = identity
        self.label = label
        self.parents = parents
        self.children = children

    def __str__(self):
        return f"Node(id={self.id}, label='{self.label}', parents={self.parents}, children={self.children})"
    
    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        return node(self.get_id(), self.get_label(), self.get_parents().copy(), self.get_children().copy())
    
    # getters
    def get_id(self):
        return self.id 
    def get_label(self):
        return self.label
    def get_parents(self):
        return self.parents
    def get_children(self):
        return self.children
    
    # setters
    def set_id(self, id):
        self.id = id
    def set_label(self, label):
        self.label = label
    def set_parents(self, parents):
        self.parents = parents
    def set_children(self, children):
        self.children = children

    def add_child_id(self, id, multiplicity=1):
        if id in self.children.keys():
            self.children[id] += multiplicity
        else:
            self.children[id] = multiplicity

    def add_parent_id(self, id, multiplicity=1):
        if id in self.parents.keys():
            self.parents[id] += multiplicity
        else:
            self.parents[id] = multiplicity

    def remove_parent_once(self, parent_id):
        '''
        parent_id: int
        removes one multiplicity from the parent node
        if multiplicity = 0, the parent is removed
        '''
        if parent_id in self.parents and self.parents[parent_id] > 0:
            self.parents[parent_id] -= 1
            if self.parents[parent_id] == 0:
                self.parents.pop(parent_id)


    def remove_child_once(self, child_id):
        '''
        child_id: int
        removes one multiplicity from the child node
        if multiplicity is 0, the child is removed
        '''
        if child_id in self.children and self.children[child_id] > 0:
            self.children[child_id] -= 1
            if self.children[child_id] == 0:
                self.children.pop(child_id)

    def remove_parent_id(self, parent_id):
        '''
        parent_id: int
        completely removes the parent node, regardless of multiplicity 
        '''
        if parent_id in self.parents:
            self.parents.pop(parent_id)

    def remove_child_id(self, child_id):
        '''
        child_id: int
        completely removes the child node, regardless of multiplicity 
        '''
        if child_id in self.children:
            self.children.pop(child_id)


    def indegree(self) -> int:
       return sum(self.get_parents().values())


    def outdegree(self) -> int:
        return sum(self.get_children().values())


    def degree(self) -> int:
        return self.indegree() + self.outdegree()

class open_digraph:  # for open directed graph
    def __init__(self, inputs, outputs, nodes):
        '''
        inputs : int list; the ids of the input nodes
        outputs : int list; the ids of the output nodes
        nodes : node iter;
        '''
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = {node.id: node for node in nodes}  # self.nodes: <int, node> dict

    def __str__(self):
        nodes_str = "\n  ".join(str(node) for node in self.nodes.values())
        return (f"OpenDigraph(\n  inputs={self.inputs}, \n  outputs={self.outputs}, \n  nodes=\n  {nodes_str}\n)")
    
    def __repr__(self):
        return self.__str__()
    
    # returns a copy of the graph
    def copy(self):
        return open_digraph(self.get_input_ids().copy(), self.get_output_ids().copy(), copy.deepcopy(self.get_nodes()))
    
    # getters 

    # returns list of input ids
    def get_input_ids(self):
        return self.inputs
    # returns list of output ids
    def get_output_ids(self):
        return self.outputs
    # returns <int, node> dict
    def id_node_map(self): 
        return self.nodes 
    # returns node list
    def get_nodes(self):
        return list(self.nodes.values())
    # returns node_ids list
    def get_nodes_id(self):
        return list(self.nodes.keys())
    # returns the node with given id
    def get_node_by_id(self, id): 
        return self.nodes[id]
    # returns list of nodes with given ids
    def get_nodes_by_ids(self, ids):
        return [self.nodes[id] for id in ids if id in self.nodes]
    
    # setters
    def set_inputs(self, inputs):
        self.inputs = inputs
    def set_outputs(self, outputs):
        self.outputs = outputs
    def add_input_id(self, id):
        self.inputs.append(id)
    def add_output_id(self, id):
        self.outputs.append(id)

    def new_id(self):
        '''
        returns an unused node id in the graph
        '''
        return max(self.nodes.keys(), default=0) + 1
    
    def add_edge(self, src, tgt, m=1):
        '''
        adds edge from src node to tgt node with corresponding multiplicity
        raises ValueError if src / tgt not found
        '''
        if src in self.nodes.keys() and tgt in self.nodes.keys(): 
            self.get_node_by_id(src).add_child_id(tgt, m)
            self.get_node_by_id(tgt).add_parent_id(src, m)
        else: 
            raise ValueError("Source or target node ID not found in the graph.")
    
    def add_edges(self, edges, mult=[]):
        '''
        edges: list of tuples (src, tgt)
        mult : list of multiplicities (must be empty or have the same length as edges)
        adds an edge from src to tgt for each pair in edges with the corresponding multiplicity.
        '''
        if mult and len(mult) != len(edges):
            raise ValueError("add_edges: mult list must be empty or have the same length as edges")

        for i, (src, tgt) in enumerate(edges):
            m = mult[i] if mult else 1  # use the provided multiplicity or default to 1
            m = mult[i] if mult else 1  # use the provided multiplicity or default to 1
            self.add_edge(src, tgt, m)

    
    def add_node(self, label='', parents={}, children={}):
            """
            adds a node with a label, parents, and children to the graph
            returns the new node id
            raises ValueError if parents or children are not in the graph
            """
            p_ids = list(parents.keys()) # extract parent ids
            c_ids = list(children.keys()) # extract child ids
            r = p_ids + c_ids  # combined parent and child ids
            
            if r and not all(n in self.nodes for n in r):
                raise ValueError("one or more parents/children do not exist in the graph")
            
            n_ID = self.new_id()  # generate a new id
            new_node = node(n_ID, label, {}, {})  # create new node
            self.nodes[n_ID] = new_node  # add to graph
            
            # create edge lists
            p = [(p, n_ID) for p in p_ids]  
            c = [(n_ID, c) for c in c_ids]
            edges = p + c
            mult = list(parents.values()) + list(children.values())
            
            self.add_edges(edges, mult)  # add edges to graph
            return n_ID

    # removes edge from src to tgt
    def remove_edge(self, src, tgt):
        '''
        src: int; id of src
        tgt: int; id of tgt node
        removes edge from src to tgt
        doesn't do anything if src, tgt don't exist
        '''
        if (src in self.get_nodes_id() and tgt in self.get_nodes_id()):
            s = self.get_node_by_id(src)
            t = self.get_node_by_id(tgt)
            s.remove_child_once(tgt)
            t.remove_parent_once(src)

    def remove_parallel_edges(self, src, tgt):
        '''
        src: int; id of src node
        tgt: int; id of tgt node
        removes all edges from src to tgt
        doesn't do anything if src, tgt don't exist 
        '''
        if (src in self.get_nodes_id() and tgt in self.get_nodes_id()):
            s = self.get_node_by_id(src)
            t = self.get_node_by_id(tgt)
            s.remove_child_id(tgt)
            t.remove_parent_id(src)

    def remove_node_by_id(self, node_id):
        '''
        node_id: int
        removes node from the graph along with all its edges
        '''
        n = self.get_node_by_id(node_id)
        edges = [(p, node_id) for p in n.get_parents()] + [(node_id, c) for c in n.get_children()]
        
        self.remove_several_parallel_edges(edges)  # remove all edges linked to node
        self.nodes.pop(node_id)  # remove node itself


    def remove_edges(self, edges):
        '''
        edges: src * tgt list; where src: int and tgt: int are node ids
        removes one set of edges for each src, tgt pair
        '''
        for src, tgt in edges:
            self.remove_edge(src, tgt)

    def remove_several_parallel_edges(self, edges):
        '''
        edges: src * tgt list; where src: int and tgt: int are node ids
        removes all of the edges for each src, tgt pair
        '''
        for src, tgt in edges:
            self.remove_parallel_edges(src, tgt)

    def remove_nodes_by_id(self, node_ids):
        '''
        node_ids: int list 
        removes all nodes whose ids are in the list 
        '''
        for node_id in node_ids:
            self.remove_node_by_id(node_id)

    def add_input_node(self, id): 
        '''
        id : int
        creates a new input node that points to the node id given
        raises ValueError if id does not exist in graph, or if id is already an input
        '''
        if id in self.get_nodes_id() and id not in self.get_input_ids():
            # leaves label empty
            self.add_input_id(self.add_node(children={id: 1}))
        else: 
            raise ValueError("add_input_node : Invalid given id")

    def is_well_formed(self):
        '''
        checks if the graph is well-formed by verifying the following properties:
        - each input and output node ID is in the graph
        - each input node has exactly one child with multiplicity 1 and no parents
        - each output node has exactly one parent with multiplicity 1 and no children
        - each key in nodes points to a node with the corresponding ID
        - if node j has node i as a child with multiplicity m, then node i must have node j as a parent with multiplicity m, and vice versa.

        returns: True if the graph is well-formed, False if not.
        '''
        nodes_id = set(self.get_nodes_id())

        # validate input/output nodes exist & their structure
        # validate input/output nodes exist & their structure
        for node_id in self.inputs + self.outputs:
            if node_id not in nodes_id:
                return False  # input/output node must exist in the graph
            
            node = self.get_node_by_id(node_id)
            parents, children = node.get_parents(), node.get_children()

            # check inputs
            if node_id in self.inputs:
                 # then he must have exactly one child (multiplicity = 1) and no parents
                if len(children) != 1 or list(children.values())[0] != 1 or len(parents) != 0:
                    return False

            # check outputs
            if node_id in self.outputs: 
                # then he must have exactly one parent (multiplicity = 1) and no children
                if len(parents) != 1 or list(parents.values())[0] != 1 or len(children) != 0:
                    return False

        # validate node consistency and relationships
        # validate node consistency and relationships
        for node_id, node in self.nodes.items():
            if node.get_id() != node_id:
                return False

            for child_id, multiplicity in node.get_children().items():
                if child_id not in self.nodes or self.nodes[child_id].get_parents().get(node_id, -1) != multiplicity:
                    return False  # a child must reference parent correctly

            for parent_id, multiplicity in node.get_parents().items():
                if parent_id not in self.nodes or self.nodes[parent_id].get_children().get(node_id, -1) != multiplicity:
                    return False  # a parent must reference child correctly

        return True

    def assert_is_well_formed(self):
        '''
        asserts the graph is well-formed
        raises AssertionError if the graph is not well-formed
        '''
        if not self.is_well_formed():
            raise AssertionError("asserrt_is_well_formed : graph is not well-formed.")

    def is_cyclic(self):
       """
       Vérifie si le graphe contient un cycle.
       Returns: True si le graphe est cyclique, False sinon.
       """
       copy = self.copy()
       in_degree = {node_id: node.indegree() for node_id, node in copy.nodes.items()}
       q = [node_id for node_id, deg in in_degree.items() if deg == 0]


       while q:
           node_id = q.pop(0)  #
           for child_id in copy.nodes[node_id].get_children():
               in_degree[child_id] -= 1
               if in_degree[child_id] == 0:
                   q.append(child_id)
           copy.remove_node_by_id(node_id)


       return len(copy.nodes) > 0
    
    
    def shift_indices(self, n: int):
        """
        Décale tous les indices des nœuds du graphe de n unités.

        Paramètre:
        - n : entier, le décalage à appliquer aux indices des nœuds.
              Peut être négatif.

        Vérifie que le décalage n’entraîne pas de collisions d'indices.
        Si tel est le cas, une exception est levée.
        """
        new_ids = {node_id: node_id + n for node_id in self.nodes}
        if len(set(new_ids.values())) < len(new_ids):
            raise ValueError("Le décalage entraîne des collisions d'indices.")
        new_nodes = {}
        for node_id, node in self.nodes.items():
            new_id = new_ids[node_id]
            new_parents = {new_ids[parent_id]: mult for parent_id, mult in node.get_parents().items()}
            new_children = {new_ids[child_id]: mult for child_id, mult in node.get_children().items()}
            
            node.set_id(new_id)
            node.set_parents(new_parents)
            node.set_children(new_children)
            new_nodes[new_id] = node

        self.nodes = new_nodes
        self.inputs = [new_ids[input_id] for input_id in self.inputs]
        self.outputs = [new_ids[output_id] for output_id in self.outputs]


    def min_id(self):
        """
        Retourne:
        - L'ID du nœud avec l'ID le plus bas, None si le graphe est vide.
        """
        return min(self.nodes.keys()) if self.nodes else None


    def max_id(self):
        """
        Retourne:
        - L'ID du nœud avec l'ID le plus haut, None si le graphe est vide.
        """
        return max(self.nodes.keys()) if self.nodes else None
    
    def add_output_node(self, id): 
        '''
        id : int
        creates a new output node that points to the node id given
        raises ValueError if id does not exist in graph, or if id is already an output
        '''
        if id in self.get_nodes_id() and id not in self.get_output_ids():
            # leaves label empty
            self.add_output_id(self.add_node(parents={id: 1}))
        else: 
            raise ValueError("add_output_node : Invalid given id")  

    def node_to_index(self):
        '''
        Return un dictionnaire associant chaque ID de nœud à un entier unique.
        '''
        return {node: idx for idx, node in enumerate(sorted(self.nodes.keys()))}

    def iparallel(self, g):
        '''
        @param : g an open digraph
        changes graph to its composition with g 
        '''
        new_graph = g.copy()
        shift = self.max_id() - new_graph.min_id() + 1 if self.nodes else 0
        new_graph.shift_indices(shift)
        for id, node in new_graph.id_node_map().items():
            self.nodes[id] = node
        for input in new_graph.get_input_ids():
            self.add_input_id(input)
        for output in new_graph.get_output_ids():
            self.add_output_id(output)
        return shift # optionnel, facilite la fonction icompose

    @classmethod
    def parallel(cls, g1, g2):
        '''
        @param : g1, g2 two open digraphs
        return open digraph : the parallel composition of g1 and g2
        '''
        new_graph = g1.copy()
        new_graph.iparallel(g2)
        return new_graph

    def icompose(self, g):
        """
        @param : f an open_digraph, the graph that will be composed with self in sequence
        the current graph will contain f followed by self
        """
        # check #outs == #ins
        if len(g.get_output_ids()) != len(self.get_input_ids()):
            raise ValueError("Mismatch in out/in counts")

        # parallel-merge g into self, get shift
        s = self.iparallel(g)

        # shifted IDs of g's outs/ins
        go = [x + s for x in g.get_output_ids()]
        gi = [x + s for x in g.get_input_ids()]

        # wire g's outs -> self's current ins
        for si, fo in zip(self.get_input_ids(), go):
            self.add_edge(fo, si)

        # final inputs become g's (shifted)
        self.set_inputs(gi)
        return self



    @classmethod
    def compose(cls, g1, g2):
        """
        @param : g1 g2 open_digraphs
        returns : open digraph, g1 composed in sequence with g2
        """
        new_graph = g1.copy()
        new_graph.icompose(g2)
        return new_graph

    @classmethod
    def identity(cls, n):
        '''
        @param : n (int)
        returns : open_digraph, the n-identity graph 
        '''
        g = cls.empty()
        for i in range(n):
            input_id = g.add_node()
            output_id = g.add_node()
            g.add_input_id(input_id)
            g.add_output_id(output_id)
            g.add_edge(input_id, output_id)
        return g
    
    def connected_components(self):
        """
        connected components in the graph.
        returns:
            - int: number of connected components.
            - dict: Mapping of node id to its component index.
        """
        v = set()  # visited nodes
        c = {}  # node-to-component map
        idx = 0  # component index

        def dfs(n, i):
            s = [n]  # stack for DFS
            while s:
                x = s.pop()
                if x not in v:
                    v.add(x)
                    c[x] = i
                    s.extend(self.nodes[x].parents.keys() | self.nodes[x].children.keys())

        for n in self.nodes:
            if n not in v:
                dfs(n, idx)
                idx += 1

        return idx, c


    def connected_components_list(self):
        """
        returns a list of subgraphs, each corresponding to a connected component.
        """
        n, m = self.connected_components()
        sg = [[] for _ in range(n)] #subgraph
        
        for x, i in m.items():
            sg[i].append(self.nodes[x]) # add node to its respective component list
        
        return [open_digraph(
            [inp for inp in self.get_input_ids() if m[inp] == i],
            [out for out in self.get_output_ids() if m[out] == i],
            sg[i]
        ) for i in range(n)]
    
    def adjacency_matrix(self):
        '''
        retourne la matrice d'adjacence du graphe (en ignorant inputs et outputs).
        '''
        n = len(self.nodes)
        m = self.node_to_index()
        a = [[0] * n for _ in range(n)]

        for u, v in self.nodes.items():
            i = m[u]  # row
            for w, x in v.children.items():
                j = m[w]  # column
                a[i][j] = x 

        return a

    def topological_sort(self):
        if self.is_cyclic():
            raise ValueError("Le graphe est cyclique")

        graph_copy = self.copy()
        l = [] 

        while graph_copy.nodes:
            co_feuilles = [node_id for node_id, node in graph_copy.nodes.items() if not node.parents]
            if not co_feuilles:  
                raise ValueError("Le graphe devrait être acyclique mais nope.")
            l.append(co_feuilles)
            for node_id in co_feuilles:
                graph_copy.remove_node_by_id(node_id)

        return l

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def save_as_dot_file(self, path, verbose=False):

        assert path.endswith(".dot"), "path must end with .dot"
        s = "digraph G {\n\n"

        # Write out all nodes
        # Write out all nodes
        #    - If it's an input node, add input="true"
        #    - If it's an output node, add output="true"
        #    - If it has a label, use label="..."
        #    - If verbose, also show \nid=ID
        for node_id, node in self.nodes.items():
            # Gather attributes in a small dict
            attr_dict = {}
            if node.label:
                # e.g. label="A"
                # if verbose we add the ID on a second line
                if verbose:
                    attr_dict["label"] = f'{node.label}\\nid={node_id}'
                else:
                    attr_dict["label"] = node.label
            else:
                # no label
                if verbose:
                    attr_dict["label"] = f'{node_id}'
                    attr_dict["label"] = f'{node_id}'
                else:
                    attr_dict["label"] = ""

            # mark as input or output
            if node_id in self.inputs:
                attr_dict["input"] = "true"
            if node_id in self.outputs:
                attr_dict["output"] = "true"
            
            if node_id in self.inputs:
                attr_dict["shape"] = "diamond"
            elif node_id in self.outputs:
                attr_dict["shape"] = "box"
            else:
                attr_dict["shape"] = "circle"
            
            if node_id in self.inputs:
                attr_dict["shape"] = "diamond"
            elif node_id in self.outputs:
                attr_dict["shape"] = "box"
            else:
                attr_dict["shape"] = "circle"

            # build final bracket string: e.g. [label="A", input="true"]
            if attr_dict:
                # turn dict into a list of key="value"
                attributes_str = ", ".join(f'{k}="{v}"' for k,v in attr_dict.items())
                s += f'    v{node_id} [{attributes_str}];\n'
            else:
                # if truly no attributes, just v{node_id}
                # if truly no attributes, just v{node_id}
                s += f'    v{node_id};\n'

        s += "\n"

        #  write edges
        for node_id, node in self.nodes.items():
            for child_id, multiplicity in node.get_children().items():
                if multiplicity <= 1:
                    # single edge
                    s += f'    v{node_id} -> v{child_id};\n'
                else:
                    # multiple edges
                    for _ in range(multiplicity):
                        s += f'    v{node_id} -> v{child_id};\n'

        s += "\n}\n"

        with open(path, "w") as f:
            f.write(s)

    def display(self, verbose=False, filename_prefix="my_graph"):
        """
        Saves the graph to a fixed .dot file, converts it to a PNG using Graphviz,
        then opens the PNG in the default viewer (on macOS, Preview).
        """
        dot_path = f"{filename_prefix}.dot"
        png_path = f"{filename_prefix}.png"
        self.save_as_dot_file(dot_path, verbose=verbose)
        # convert the .dot file to a PNG
        os.system(f"dot -Tpng '{dot_path}' -o '{png_path}'")
        # open the resulting PNG in the default viewer
        abs_png = os.path.abspath(png_path)
        webbrowser.open(f"file://{abs_png}")
        os.remove(dot_path)
    
    #############################################
    ##                 Direction               ##
    #############################################
    def dijkstra(self, src, tgt, direction=None):
        """
        Implémentation de l'algorithme de Dijkstra pour un graphe orienté.

        """
        Q = [src]
        dist = {src: 0}
        prev = {}

        while Q:
            u = None
            min_dist = None
            for node in Q:
                if min_dist is None or dist[node] < min_dist:
                    min_dist = dist[node]
                    u = node


            if tgt is not None and u == tgt:
                return dist, prev
            Q.remove(u)
            if direction == -1:
                neighbours = self.get_node_by_id(u).get_parents().keys()
            elif direction == 1:
                neighbours = self.get_node_by_id(u).get_children().keys()
            else: # bidirection
                neighbours = list(self.get_node_by_id(u).get_parents().keys()) + list(self.get_node_by_id(u).get_children().keys())

            for v in neighbours:
                w = 1
                new_dist = dist[u] + w

                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    if v not in Q:
                        Q.append(v)

        return dist, prev

    def shortest_path(self, src, tgt):
        """
        Reconstitue le plus court chemin entre src et tgt.
        """
        return self.dijkstra(src, tgt, direction=0)[0][tgt]


    def ancestors_in_common(self, u, v):
        """
        Retourne un dictionnaire des ancêtres communs de u et v avec leurs distances respectives.
        """
        dist_u, _ = self.dijkstra(u, direction=-1)  
        dist_v, _ = self.dijkstra(v, direction=-1) 
        communs = {}
        for node in dist_u:
            if node in dist_v:
                communs[node] = (dist_u[node], dist_v[node])

        return communs
    
    #############################################
    ##                 Direction               ##
    #############################################
    def dijkstra(self, src, tgt, direction=None):
        """
        Implémentation de l'algorithme de Dijkstra pour un graphe orienté.

        """
        Q = [src]
        dist = {src: 0}
        prev = {}

        while Q:
            u = None
            min_dist = None
            for node in Q:
                if min_dist is None or dist[node] < min_dist:
                    min_dist = dist[node]
                    u = node


            if tgt is not None and u == tgt:
                return dist, prev
            Q.remove(u)
            if direction == -1:
                neighbours = self.get_node_by_id(u).get_parents().keys()
            elif direction == 1:
                neighbours = self.get_node_by_id(u).get_children().keys()
            else: # bidirection
                neighbours = list(self.get_node_by_id(u).get_parents().keys()) + list(self.get_node_by_id(u).get_children().keys())

            for v in neighbours:
                w = 1
                new_dist = dist[u] + w

                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    if v not in Q:
                        Q.append(v)

        return dist, prev

    def shortest_path(self, src, tgt):
        """
        Reconstitue le plus court chemin entre src et tgt.
        """
        return self.dijkstra(src, tgt, direction=0)[0][tgt]


    def ancestors_in_common(self, u, v):
        """
        Retourne un dictionnaire des ancêtres communs de u et v avec leurs distances respectives.
        """
        dist_u, _ = self.dijkstra(u, direction=-1)  
        dist_v, _ = self.dijkstra(v, direction=-1) 
        communs = {}
        for node in dist_u:
            if node in dist_v:
                communs[node] = (dist_u[node], dist_v[node])

        return communs

    @classmethod
    def empty(cls):
        return cls([], [], [])
    
    @classmethod
    def from_dot_file(cls, p):
        # open & read
        with open(p, "r") as f:
            data = f.read()
        # locate top-level braces
        s = data.find("{")
        e = data.rfind("}")
        if s == -1 or e == -1 or s > e:
            raise ValueError("No matching brakets found in DOT file")

        # the portion inside braces
        core = data[s+1:e].strip()
        # start with an empty graph
        g = cls.empty()
        name_to_id = {}

        # split lines on ';'
        parts = [ln.strip() for ln in core.split(";") if ln.strip()]

        for line in parts:
            # check for an edge
            if "->" in line:
                # separate any trailing bracketed attributes, e.g. v0->v1 [color=red,mult=2]
                bracket_part = ""
                if "[" in line and "]" in line:
                    # capture bracket text
                    bstart = line.index("[")
                    bend = line.rindex("]")
                    bracket_part = line[bstart+1:bend].strip()
                    line = line[:bstart].strip()

                # now line should be something like 'v0->v1->v2'
                nodes_list = [x.strip() for x in line.split("->") if x.strip()]
                # parse edge attributes if any
                edge_attrs = {}
                if bracket_part:
                    for a in bracket_part.split(","):
                        kv = a.strip().split("=")
                        if len(kv) == 2:
                            k = kv[0].strip()
                            v = kv[1].strip().strip('"')
                            edge_attrs[k] = v

                # handle chain edges
                for i in range(len(nodes_list) - 1):
                    src_name, tgt_name = nodes_list[i], nodes_list[i+1]
                    # if node undefined, create a placeholder
                    if src_name not in name_to_id:
                        name_to_id[src_name] = g.add_node()
                    if tgt_name not in name_to_id:
                        name_to_id[tgt_name] = g.add_node()

                    # repeat the edge multiple times based on occurrences in the file
                    mult = int(edge_attrs.get('mult', 1))
                    g.add_edges([(name_to_id[src_name], name_to_id[tgt_name])] * mult)

                # (ignore any other edge attrs like color, style, etc.)

            else:
                # node definition
                # example: v0 [label="X", input=true, color=red]
                bracket_part = ""
                if "[" in line and "]" in line:
                    bstart = line.index("[")
                    bend = line.rindex("]")
                    bracket_part = line[bstart+1:bend].strip()
                    line = line[:bstart].strip()

                node_name = line.strip()
                # gather attributes
                node_attrs = {}
                if bracket_part:
                    for a in bracket_part.split(","):
                        kv = a.strip().split("=")
                        if len(kv) == 2:
                            k = kv[0].strip()
                            v = kv[1].strip().strip('"')
                            node_attrs[k] = v

                # if node already known, reuse it; else create
                if node_name in name_to_id:
                    n_id = name_to_id[node_name]
                else:
                    n_id = g.add_node()
                    name_to_id[node_name] = n_id

                # set label if found
                if "label" in node_attrs:
                    # update node's label
                    g.nodes[n_id].label = node_attrs["label"]
                # interpret input, output
                if "input" in node_attrs and node_attrs["input"].lower() in ["true","1"]:
                    g.add_input_id(n_id)
                if "output" in node_attrs and node_attrs["output"].lower() in ["true","1"]:
                    g.add_output_id(n_id)

        return g
    
    @classmethod
    def graph_from_adjacency_matrix(self, matrix):
        '''
        convertit une matrice d'adjacence en un multigraphe.
        param: Matrice d'adjacence (liste de listes d'entiers).
        return: Un multigraphe représenté par la matrice d'adjacence.
        convertit une matrice d'adjacence en un multigraphe.
        param: Matrice d'adjacence (liste de listes d'entiers).
        return: Un multigraphe représenté par la matrice d'adjacence.
        '''
        g = open_digraph([], [], [])  # create an empty graph
        n = len(matrix)
        node_ids = {}

        # create nodes
        for _ in range(n):
            n_id = g.add_node()
            node_ids[_] = n_id

        # create edges
        for x in range(n):
            for y in range(n):
                if matrix[x][y] > 0:
                    g.add_edge(node_ids[x], node_ids[y], matrix[x][y])
        return g
    
    @classmethod
    def random_graph(cls, n, bound, inputs=0, outputs= 0, form = "free"):
        ''' 
            return un graphe aléatoire suivant les contraintes spécifiées.

            n: le nombre de nœuds 
            bound: la borne supérieure pour les entiers générés (entier positif).
            inputs: le nombre de nœuds d'entrée et outputs, le nombre de nœuds de sortie (entier positif).
            form: Forme du graphe ("free", "DAG", "oriented", "loop-free", "undirected", "loop-free undirected").

        '''
        

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
        
        graph = cls.graph_from_adjacency_matrix(matrix)
        nodes = list(graph.nodes.keys())
        if inputs > len(nodes) or outputs > len(nodes):
            raise ValueError("depasse le nombre de noeuds")
        
        graph.inputs = random.sample(nodes, inputs)
        remaining_nodes = [n for n in nodes if n not in graph.inputs]
        graph.outputs = random.sample(remaining_nodes, outputs)
        return graph
    
                
#############################################
##            Matrix                       ##
#############################################


def random_int_list(n, bound):
    '''
    return une liste de taille n contenant des entiers aléatoires entre 0 et bound
    '''
    return [random.randrange(0, bound) for _ in range(n)]

def random_int_matrix(n, bound, null_diag=True, number_generator=(lambda : random.random())):
    '''
    return une matrice n x n avec des entiers aléatoires entre 0 et bound
    '''
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
    '''
    return une matrice symétrique n x n avec des entiers aléatoires entre 0 et bound, la diaginale est mise à zéro
    '''
    res = random_int_matrix(n, bound, null_diag)
    for i in range(n):
        for j in range(i + 1, n):
            res[j][i] = res[i][j]      
    return res

def random_oriented_int_matrix(n ,bound, null_diag=True):
    '''
    return une matrice orientée n x n avec des entiers aléatoires entre 0 et bound
    '''
    res = random_int_matrix(n,bound,null_diag)
    for i in range(n):
        for j in range(i + 1,n):
            if random.randrange(0, 2) == 1: 
                res[i][j] = 0
            else:
                res[j][i] = 0
    return res

def random_triangular_int_matrix(n ,bound, null_diag = True):
    '''
    return une matrice triangulaire supérieure n x n avec des entiers aléatoires entre 0
    '''
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            res[i][j] = random.randrange(0, bound+1)
    return res 


#############################################
##                Bool_Circ                ##
#############################################
class bool_circ(open_digraph):
    def __init__(self, graph):
        """
        Constructeur de la classe bool_circ qui hérite de open_digraph.

        Paramètre:
        - graph : un objet de type open_digraph.

        Cette méthode initialise un circuit booléen en s'assurant que le 
        graphe donné est bien formé selon les règles d'un circuit booléen.
        Si le graphe donné n'est pas bien formé, il est remplacé par un graphe vide.
        Une exception est levée si le graphe n'est toujours pas valide.
        """
        graph.assert_is_well_formed()
        super().__init__(graph.get_input_ids().copy(), graph.get_output_ids().copy(), [])
        self.nodes = graph.id_node_map().copy()

        if not self.is_well_formed():
            raise ValueError("Le graphe donné n'est pas un circuit booléen qui me plait ;).")

    def is_well_formed(self):
        """
        Vérifie si le circuit booléen est bien formé.

        Un circuit booléen bien formé doit respecter les critères suivants :
        - Il ne doit pas contenir de cycles.
        - Les nœuds de type 'copie' (label '') doivent avoir exactement un seul parent.
        - Les portes logiques (ET '&', OU '|', XOR '^') doivent avoir exactement une sortie.
        - Les portes NON ('~') doivent avoir exactement une entrée et une sortie.
        - Les constantes '0' et '1' ne doivent avoir aucune sortie.

        Retourne:
        - True si le circuit booléen est valide, False sinon.
        """
        if self.is_cyclic():
            return False

        for node in self.get_nodes():
            label = node.get_label()
            indegree, outdegree = node.indegree(), node.outdegree()

            if label == '':
                if indegree != 1:
                    return False
            elif label in ('&', '|', '^'):
                if outdegree != 1:
                    return False
            elif label == '~':
                if indegree != 1 or outdegree != 1:
                    return False
            elif label in ('0', '1'):
                if outdegree != 0:
                    return False
            else:
                return False
        return True
    