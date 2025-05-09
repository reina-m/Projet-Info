class OpenDigraphEdgeMixin:
    """mixin containing edge manipulation operations"""
    
    def add_edge(self, src, tgt, m=1):
        """
        adds edge from src node to tgt node with multiplicity
        
        args:
            src (int): source node id
            tgt (int): target node id
            m (int): edge multiplicity, defaults to 1
            
        raises:
            ValueError: if src or tgt not found in graph
        """
        if src in self.nodes.keys() and tgt in self.nodes.keys(): 
            self.get_node_by_id(src).add_child_id(tgt, m)
            self.get_node_by_id(tgt).add_parent_id(src, m)
        else: 
            raise ValueError("Source or target node ID not found in the graph")

    def add_edges(self, edges, mult=[]):
        """
        adds multiple edges at once
        
        args:
            edges (list[tuple]): list of (src,tgt) pairs
            mult (list[int]): multiplicities for each edge
                must be empty or same length as edges
                
        raises:
            ValueError: if mult length doesn't match edges
        """
        if mult and len(mult) != len(edges):
            raise ValueError("add_edges: mult list must be empty or have the same length as edges")

        for i, (src, tgt) in enumerate(edges):
            m = mult[i] if mult else 1
            self.add_edge(src, tgt, m)

    def remove_edge(self, src, tgt):
        """
        removes single edge between nodes
        
        args:
            src (int): source node id  
            tgt (int): target node id
            
        notes:
            does nothing if nodes don't exist
            removes one multiplicity if multiple edges exist
        """
        if (src in self.get_nodes_id() and tgt in self.get_nodes_id()):
            s = self.get_node_by_id(src)
            t = self.get_node_by_id(tgt)
            s.remove_child_once(tgt)
            t.remove_parent_once(src)

    def remove_edges(self, edges):
        """
        removes multiple edges
        
        args:
            edges (list[tuple]): list of (src,tgt) pairs to remove
        """
        for src, tgt in edges:
            self.remove_edge(src, tgt)

    def remove_parallel_edges(self, src, tgt):
        """
        removes all edges between two nodes
        
        args:
            src (int): source node id
            tgt (int): target node id
        """
        if (src in self.get_nodes_id() and tgt in self.get_nodes_id()):
            s = self.get_node_by_id(src)
            t = self.get_node_by_id(tgt)
            s.remove_child_id(tgt)
            t.remove_parent_id(src)

    def remove_several_parallel_edges(self, edges):
        """
        removes all parallel edges for multiple pairs
        
        args:
            edges (list[tuple]): list of (src,tgt) pairs
        """
        for src, tgt in edges:
            self.remove_parallel_edges(src, tgt)