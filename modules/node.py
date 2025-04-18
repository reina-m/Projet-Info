
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
