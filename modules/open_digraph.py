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
    
    def __copy__(self):
        return node(self.id, self.label, self.parents, self.children)
    
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
        self.children[id] = multiplicity

    def add_parent_id(self, id, multiplicity=1):
        self.parents[id] = multiplicity

    def remove_parent_once(self, parent_id):
        '''
        parent_id: int
        removes one multiplicity from the parent node
        if multiplicity = 0, the parent is removed
        '''
        if parent_id in self.parents:
            self.parents[parent_id] -= 1
            if self.parents[parent_id] == 0:
                self.parents.pop(parent_id)


    def remove_child_once(self, child_id):
        '''
        child_id: int
        removes one multiplicity from the child node
        if multiplicity is 0, the child is removed
        '''
        if child_id in self.children:
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
    def __copy__(self):
        return open_digraph(self.inputs, self.outputs, self.nodes)
    
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
    
    def add_edge(self, src, tgt):
        '''
        adds edge from src node to tgt node 
        raises ValueError if src / tgt not found
        '''
        if src in self.nodes and tgt in self.nodes: 
            self.nodes[src].add_child_id(tgt)
            self.nodes[tgt].add_parent_id(src)
        else: 
            raise ValueError("Source or target node ID not found in the graph.")
    
    def add_edges(self, edges):
        '''
        edges: (src * tgt) list;
        adds edge from src node to tgt node to each (src * tgt) pair in the given list
        '''
        for src, tgt in edges:
            self.add_edge(src, tgt)
    
    def add_node(self, label='', parents=None, children=None):
        '''
        label: str; node label (default: '').
        parents: int->int dict; parent IDs mapped to multiplicities (default: {}).
        children: int->int dict; child IDs mapped to multiplicities (default: {}).
        adds node with label, parents, and children to graph; returns new node ID.
        '''
        new_id = self.new_id()
        new_node = node(new_id, label, parents or {}, children or {})
        for id, multiplicity in parents.items():
            node = self.get_node_by_id(id)
            if node:
                new_node.add_parent_id(id, multiplicity)
        for id, multiplicity in children.items():
            node = self.get_node_by_id(id)
            if node: 
                new_node.add_child_id(id, multiplicity)
        self.nodes[new_id] = new_node
        return new_id

    # removes edge from src to tgt
    def remove_edge(self, src, tgt):
        '''
        src: int; id of src
        tgt: int; id of tgt node
        removes edge from src to tgt
        '''
        s = self.get_node_by_id(src)
        t = self.get_node_by_id(tgt)
        s.remove_child_once(tgt)
        t.remove_parent_once(src)

    def remove_parallel_edges(self, src, tgt):
        '''
        src: int; id of src node
        tgt: int; id of tgt node
        removes all edges from src to tgt 
        '''
        s = self.get_node_by_id(src)
        t = self.get_node_by_id(tgt)
        s.remove_child_id(tgt)
        t.remove_parent_id(src)

    def remove_node_by_id (self, node_id):
        '''
        node_id: int
        removes node from the graph
        '''
        node = self.nodes.pop(node_id)
        for parent in node.get_parents():
            self.remove_parallel_edges(parent, node_id)
        for child in node.get_children():
            self.remove_parallel_edges(node_id, child)

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

        # single pass to validate input/output nodes exist & their structure
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

        # single pass to validate node consistency and relationships
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

    @classmethod
    def empty(cls):
        return cls([], [], [])
