import copy
class OpenDigraphBaseMixin:
    """base mixin containing core functionality"""
    
    def __init__(self, inputs, outputs, nodes):
        """
        initializes graph with inputs, outputs and nodes
        
        args:
            inputs (list[int]): input node ids
            outputs (list[int]): output node ids
            nodes (list[node]): list of nodes
        """
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = {node.id: node for node in nodes}

    def copy(self):
        return self.__class__(self.get_input_ids().copy(), 
                            self.get_output_ids().copy(), 
                            copy.deepcopy(self.get_nodes()))
    
    def __str__(self):
        nodes_str = "\n  ".join(str(node) for node in self.nodes.values())
        return (f"OpenDigraph(\n  inputs={self.inputs}, \n  outputs={self.outputs}, \n  nodes=\n  {nodes_str}\n)")
    
    def __repr__(self):
        return self.__str__()

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
    
    def min_id(self):
        """
        gets smallest node id in graph
        
        args:
            none
            
        returns:
            int: smallest node id, or None if graph is empty
        """
        return min(self.nodes.keys()) if self.nodes else None


    def max_id(self):
        """
        gets largest node id in graph
        
        args:
            none
            
        returns:
            int: largest node id, or None if graph is empty
        """
        return max(self.nodes.keys()) if self.nodes else None
    
    @classmethod
    def empty(cls):
        """
        creates empty graph with no nodes
        
        args:
            none
            
        returns:
            open_digraph: new empty graph
        """
        return cls([], [], [])

