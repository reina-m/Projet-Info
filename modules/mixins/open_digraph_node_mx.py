from ..node import node

class OpenDigraphNodeMixin:
    """mixin containing node manipulation operations"""
    
    def add_node(self, label='', parents={}, children={}):
        """
        adds node with connections, matching main implementation
        
        args:
            label (str): node label
            parents (dict): parent_id -> multiplicity mapping  
            children (dict): child_id -> multiplicity mapping
            
        returns:
            int: new node id
            
        raises:
            ValueError: if parents/children don't exist
        """
        p_ids = list(parents.keys())
        c_ids = list(children.keys())
        r = p_ids + c_ids

        if r and not all(n in self.nodes for n in r):
            raise ValueError("one or more parents/children don't exist in graph")
        
        n_id = self.new_id()
        new_node = node(n_id, label, {}, {})
        self.nodes[n_id] = new_node
        
        p = [(p, n_id) for p in p_ids]
        c = [(n_id, c) for c in c_ids] 
        edges = p + c
        mult = list(parents.values()) + list(children.values())
        
        self.add_edges(edges, mult)
        return n_id

    def remove_node_by_id(self, node_id):
        '''
        node_id: int
        removes node from the graph along with all its edges
        '''
        n = self.get_node_by_id(node_id)
        edges = [(p, node_id) for p in n.get_parents()] + [(node_id, c) for c in n.get_children()]
        
        self.remove_several_parallel_edges(edges)  # remove all edges linked to node
        self.nodes.pop(node_id)  # remove node itself

    def remove_nodes_by_id(self, nodes):
        """removes multiple nodes and their edges"""
        for n_id in nodes:
            self.remove_node_by_id(n_id)

    def add_input_node(self, id):
        """
        creates new input wrapper node pointing to given node
        
        args:
            id (int): target node id
            
        raises:
            ValueError: if id invalid or already an input
        """
        if id in self.get_nodes_id() and id not in self.get_input_ids():
            self.add_input_id(self.add_node(children={id: 1}))
        else:
            raise ValueError("add_input_node: invalid given id")

    def add_output_node(self, id):
        """
        creates new output wrapper node pointing from given node
        
        args:
            id (int): source node id
            
        raises:
            ValueError: if id invalid or already an output
        """
        if id in self.get_nodes_id() and id not in self.get_output_ids():
            self.add_output_id(self.add_node(parents={id: 1}))
        else:
            raise ValueError("add_output_node: invalid given id")
    
    def node_to_index(self):
        """
        creates mapping from node ids to consecutive indices
        
        args:
            none
            
        returns:
            dict: mapping from node id to index in range(0,n)
            
        notes:
            useful for converting to matrix representation
        """
        return {node: idx for idx, node in enumerate(sorted(self.nodes.keys()))}