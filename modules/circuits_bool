from open_digraph import *

class bool_circ(open_digraph):
    def __init__(self, graph):
        if not graph.is_well_formed():
            graph = open_digraph.empty()
        
        super().__init__(graph.inputs, graph.outputs, graph.nodes)
        
        if not self.is_well_formed():
            raise ValueError("Le graphe donné n'est pas un circuit booléen valide.")

        '''
            g = open_digraph.empty()
            super().__init__(graph.inputs, graph.outputs, list(grph.nodes.values()))'''


    def is_well_formed(self):
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
