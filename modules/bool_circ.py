from .open_digraph import open_digraph

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
    

    @classmethod
    def parse_parentheses(cls, *args):
        g = open_digraph.empty()
        variables = {}
        var_names = []

        for s in args:
            s = '(' + s + ')' 
            current = g.add_node(label="")
            g.add_output_id(current)
            s2 = ''

            for c in s:
                if c == '(':
                    if s2.strip():
                        g.get_node_by_id(current).set_label(s2.strip())
                    new = g.add_node(label="")
                    g.add_edge(new, current)
                    current = new
                    s2 = ''
                elif c == ')':
                    if s2.strip():
                        label = g.get_node_by_id(current).get_label()
                        g.get_node_by_id(current).set_label(label + s2.strip())
                    children = g.get_node_by_id(current).get_children_ids()
                    if children:
                        current = children[0]
                    s2 = ''
                else:
                    s2 += c

        for node in g.get_nodes():
            label = node.get_label()
            if label in ('&', '|', '~', '^', ''):
                continue
            if label in variables:
                g.fusion(variables[label], node.get_id(), label)
            else:
                variables[label] = node.get_id()
                var_names.append(label)
                node.set_label('')
                in_id = g.new_id()
                g.add_node(id=in_id, label=label, children={node.get_id(): 1})
                g.add_input_id(in_id)

        return cls(g), var_names
