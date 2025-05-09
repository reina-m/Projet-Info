class OpenDigraphAlgorithmsMixin:
    """mixin containing graph algorithms"""
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
        
        return [self.__class__(
            [inp for inp in self.get_input_ids() if m[inp] == i],
            [out for out in self.get_output_ids() if m[out] == i],
            sg[i]
        ) for i in range(n)]

    def dijkstra(self, src, tgt=None, direction=None):
        """
        finds shortest paths using dijkstra's algorithm
        
        args:
            src (int): source node id
            tgt (int, optional): target node id
            direction (int, optional): 1 for forward, -1 for backward, None for both
            
        returns:
            tuple(dict, dict):
                - distances from source to all nodes
                - predecessor map for path reconstruction
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
            else:  # bidirectional
                neighbours = list(self.get_node_by_id(u).get_parents().keys()) + \
                           list(self.get_node_by_id(u).get_children().keys())

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
        finds shortest path length between two nodes
        
        args:
            src (int): source node id
            tgt (int): target node id
            
        returns:
            int: length of shortest path from src to tgt
            
        implementation:
            uses dijkstra's algorithm in both directions
        """
        return self.dijkstra(src, tgt, direction=0)[0][tgt]
    
    def ancestors_in_common(self, u, v):
        """
        finds common ancestors of two nodes with distances
        
        args:
            u (int): first node id
            v (int): second node id
            
        returns:
            dict: mapping ancestor -> (dist_to_u, dist_to_v)
            
        implementation:
            uses backward dijkstra search from both nodes
        """
        dist_u, _ = self.dijkstra(u, direction=-1)  
        dist_v, _ = self.dijkstra(v, direction=-1) 
        communs = {}
        for node in dist_u:
            if node in dist_v:
                communs[node] = (dist_u[node], dist_v[node])

        return communs

    def topological_sort(self):
        '''
        sorts a graph topologically 
        '''
        if self.is_cyclic():
            raise ValueError("the graph is cyclic")

        graph_copy = self.copy()
        l = [] 

        while graph_copy.nodes:
            co_feuilles = [node_id for node_id, node in graph_copy.nodes.items() if not node.parents]
            if not co_feuilles:  
                raise ValueError("the graph is cyclic")
            l.append(co_feuilles)
            for node_id in co_feuilles:
                graph_copy.remove_node_by_id(node_id)

        return l

    def node_depth(self, node_id, l=None):
        '''
        @param : the id of a node in the graph (must be cyclic)
        l the topological sort of the graph (optional)
        returns : int , the depth of the node in the graph
        '''
        if node_id not in self.get_nodes_id():
            raise ValueError("node_depth : node not in graph")
        if l is None:
            l = self.topological_sort()
        for i, lvl in enumerate(l):
            if node_id in lvl:
                return i+1 # no such depth as 0
    
    def fusion(self, id1, id2, new_label=None):
        """
        merges two nodes into one, preserving connections
        
        args:
            id1 (int): first node to merge
            id2 (int): second node to merge  
            new_label (str, optional): label for merged node
            
        returns:
            int: id of new merged node
            
        raises:
            ValueError: if nodes don't exist or are the same
            
        notes:
            - inherits all connections from both nodes
            - handles parallel edges properly
            - updates input/output lists
        """
        if id1 not in self.nodes:
            raise ValueError("fusion : the first node id does not exist in the graph")
        if id2 not in self.nodes:
            raise ValueError("fusion : the second node id does not exist in the graph")
        if id1 == id2:
            raise ValueError("fusion : the two node ids are the same")

        fusion_label = new_label or self.nodes[id1].label
        fusion_id = self.add_node(fusion_label)

        fusion_parents = {}
        fusion_children = {}

        for node_id in [id1, id2]:
            for parent_id, mult in self.nodes[node_id].parents.items():
                if parent_id != id1 and parent_id != id2:
                    fusion_parents[parent_id] = fusion_parents.get(parent_id, 0) + mult
            for child_id, mult in self.nodes[node_id].children.items():
                if child_id != id1 and child_id != id2:
                    fusion_children[child_id] = fusion_children.get(child_id, 0) + mult

        for parent_id, mult in fusion_parents.items():
            self.nodes[parent_id].children[fusion_id] = mult
            self.nodes[parent_id].children.pop(id1, None)
            self.nodes[parent_id].children.pop(id2, None)

        for child_id, mult in fusion_children.items():
            self.nodes[child_id].parents[fusion_id] = mult
            self.nodes[child_id].parents.pop(id1, None)
            self.nodes[child_id].parents.pop(id2, None)

        self.nodes[fusion_id].parents = fusion_parents
        self.nodes[fusion_id].children = fusion_children

        for tmp_list in [self.inputs, self.outputs]:
            modified = False
            if id1 in tmp_list:
                tmp_list.remove(id1)
                modified = True
            if id2 in tmp_list:
                tmp_list.remove(id2)
                modified = True
            if modified and fusion_id not in tmp_list:
                tmp_list.append(fusion_id)

        self.remove_nodes_by_id([id1, id2])
        return fusion_id

    def graph_depth(self):
        '''
        returns the depth of the graph (must be acyclic)
        '''
        return len(self.topological_sort())

    def longest_path(self, u, v):
        '''
        returns longest path from u to v in a DAG
        '''
        l = self.topological_sort()
        d = {u: 0}  # longest distance from u
        p = {}      # prev map

        for lvl in l:
            for w in lvl:
                for par in self.get_node_by_id(w).get_parents():
                    if par in d and d[par] + 1 > d.get(w, -1):
                        d[w] = d[par] + 1
                        p[w] = par

        if v not in d:
            return -1, []

        # reconstruct path
        path = [v]
        while path[-1] != u:
            path.append(p[path[-1]])
        path.reverse()
        return d[v], path
    