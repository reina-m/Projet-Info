class OpenDigraphCompositionMixin:
    """mixin containing graph composition operations"""
    
    def shift_indices(self, n: int):
        """
        shifts all node indices by given offset
        
        args:
            n (int): amount to shift indices by (can be negative)
            
        returns:
            none
            
        raises:
            ValueError: if shift would cause index collisions
            
        notes:
            updates all node ids, parent/child references, inputs and outputs
        """
        new_ids = {node_id: node_id + n for node_id in self.nodes}
        if len(set(new_ids.values())) < len(new_ids):
            raise ValueError("Indice collisions.")
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

    def compose(self, other):
        """
        returns sequential composition of self and other
        
        args:
            other (open_digraph): graph to compose with
            
        returns:
            open_digraph: new graph with self feeding into other
        """
        g = self.copy()
        g.icompose(other)
        return g