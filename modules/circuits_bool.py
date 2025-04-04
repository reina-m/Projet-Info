from open_digraph import open_digraph

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
    


    def parse_parentheses(self, s):
        output_id = self.next_id
        self.nodes[output_id] = {'label': '', 'parents': [], 'children': []}
        self.outputs.append(output_id)
        current_id = output_id
        self.next_id += 1

        s2 = ''

        for char in s:
            if char == '(':
                if s2.strip():
                    self.nodes[current_id]['label'] += s2.strip()
                    s2 = ''
                new_id = self.next_id
                self.next_id += 1
                self.nodes[new_id] = {'label': '', 'parents': [], 'children': [current_id]}
                self.nodes[current_id]['parents'].append(new_id)
                current_id = new_id
            elif char == ')':
                if s2.strip():
                    self.nodes[current_id]['label'] += s2.strip()
                    s2 = ''
                if self.nodes[current_id]['children']:
                    current_id = self.nodes[current_id]['children'][0]
            else:
                s2 += char

        return self

    

    