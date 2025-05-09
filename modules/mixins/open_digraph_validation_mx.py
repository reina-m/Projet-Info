class OpenDigraphValidationMixin:
    """mixin containing graph validation methods"""
    
    def is_cyclic(self):
        """
        checks if graph contains any cycles using topological sort approach
        
        args:
            none
            
        returns:
            bool: true if graph contains cycle, false if acyclic
            
        implementation:
            uses degree counting and node removal to detect cycles
        """
        copy = self.copy()
        in_degree = {node_id: node.indegree() for node_id, node in copy.nodes.items()}
        q = [node_id for node_id, deg in in_degree.items() if deg == 0]

        while q:
           node_id = q.pop(0)  
           for child_id in copy.nodes[node_id].get_children():
               in_degree[child_id] -= 1
               if in_degree[child_id] == 0:
                   q.append(child_id)
           copy.remove_node_by_id(node_id)

        return len(copy.nodes) > 0

    def is_well_formed(self):
        """
        checks if graph follows basic rules
        
        rules checked:
            - input nodes have indegree 0, outdegree 1
            - output nodes have indegree 1, outdegree 0
            - all node ids referenced in edges exist
            - edge multiplicities match between nodes
        
        returns:
            bool: True if well-formed, False otherwise
        """
        for i in self.get_input_ids():
            if i not in self.nodes:
                return False
            node = self.get_node_by_id(i)
            if node.get_parents() or len(node.get_children()) != 1:
                return False

        for o in self.get_output_ids():
            if o not in self.nodes:
                return False
            node = self.get_node_by_id(o)
            if node.get_children() or len(node.get_parents()) != 1:
                return False

        for node in self.get_nodes():
            for p_id, mult in node.get_parents().items():
                if p_id not in self.nodes:
                    return False
                parent = self.get_node_by_id(p_id)
                if node.id not in parent.get_children() or \
                   parent.get_children()[node.id] != mult:
                    return False

            for c_id, mult in node.get_children().items():
                if c_id not in self.nodes:
                    return False
                child = self.get_node_by_id(c_id)
                if node.id not in child.get_parents() or \
                   child.get_parents()[node.id] != mult:
                    return False

        return True

    def assert_is_well_formed(self):
        """
        raises exception if graph is not well-formed
        
        raises:
            AssertionError: with description of why graph is invalid
        """
        for i in self.get_input_ids():
            assert i in self.nodes, f"Input node {i} does not exist"
            node = self.get_node_by_id(i)
            assert not node.get_parents() and len(node.get_children()) == 1, \
                   f"Input node {i} has wrong degree"

        for o in self.get_output_ids():
            assert o in self.nodes, f"Output node {o} does not exist"
            node = self.get_node_by_id(o)
            assert not node.get_children() and len(node.get_parents()) == 1, \
                   f"Output node {o} has wrong degree"

        for node in self.get_nodes():
            for p_id, mult in node.get_parents().items():
                assert p_id in self.nodes, \
                       f"Node {node.id} has edge from non-existent parent {p_id}"
                parent = self.get_node_by_id(p_id)
                assert node.id in parent.get_children() and \
                       parent.get_children()[node.id] == mult, \
                       f"Inconsistent edge multiplicity between nodes {p_id} and {node.id}"

            for c_id, mult in node.get_children().items():
                assert c_id in self.nodes, \
                       f"Node {node.id} has edge to non-existent child {c_id}"
                child = self.get_node_by_id(c_id)
                assert node.id in child.get_parents() and \
                       child.get_parents()[node.id] == mult, \
                       f"Inconsistent edge multiplicity between nodes {node.id} and {c_id}"