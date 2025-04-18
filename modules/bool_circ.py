from .open_digraph import open_digraph
import random

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
        outputs = []

        for s in args:
            stack = []
            curr = None
            buf = ''

            for c in s:
                if c == '(':
                    if buf.strip():
                        nid = g.add_node(label=buf.strip())
                        if curr is not None:
                            g.add_edge(nid, curr)
                        curr = nid
                    stack.append(curr)
                    buf = ''
                elif c == ')':
                    if buf.strip():
                        nid = g.add_node(label=buf.strip())
                        g.add_edge(nid, curr)
                    buf = ''
                    curr = stack.pop()
                else:
                    buf += c

            if buf.strip():  # if anything left after loop
                nid = g.add_node(label=buf.strip())
                if curr is not None:
                    g.add_edge(nid, curr)
                curr = nid

            g.add_output_node(curr)
            outputs.append(g.get_output_ids()[-1])

        g.set_outputs(outputs)
        return bool_circ(g)
    
    @classmethod
    def random(cls, n, nin, nout):
        g = open_digraph.random_graph(n, bound=1, form="DAG")
        g.set_inputs([])
        g.set_outputs([])

        ops = ['&', '|', '^']

        # assign logic labels + restructure
        for x in g.get_nodes():
            d_in, d_out = x.indegree(), x.outdegree()

            if d_in == 0:
                x.set_label('')
            elif d_in == 1 and d_out == 1:
                x.set_label(random.choice(['~'])) 
            elif d_in == 1 and d_out > 1:
                x.set_label('')  # copier
            elif d_in > 1 and d_out == 1:
                x.set_label(random.choice(ops))
            elif d_in > 1 and d_out > 1:
                # label it as a proper 2+‑input gate
                x.set_label(random.choice(ops))

                # grab its current outputs
                targets = list(x.get_children().items())
                # keep exactly one of them on x; everything else goes to a copy node
                first_tgt, first_m = targets[0]
                extra = targets[1:]

                # remove only the extra edges from x
                for tgt, m in extra:
                    g.remove_parallel_edges(x.id, tgt)

                # create the “fan‑out” copy node to handle the extras
                cp = g.add_node('', {x.id: 1}, {})
                for tgt, m in extra:
                    g.add_edge(cp, tgt, m)


        # force input structure
        cands = [x.id for x in g.get_nodes() if x.indegree()==0 and x.id not in g.inputs]
        while len(g.inputs) < nin:
            if cands:
                t = cands.pop()
            else:
                # fallback to any zero‑indegree node that isn't already an input
                zeros = [nid for nid in g.get_nodes_id()
                        if g.get_node_by_id(nid).indegree()==0
                        and nid not in g.inputs]
                t = random.choice(zeros)
            g.add_input_node(t)


        # force output structure
        cands = [x.id for x in g.get_nodes() 
                if x.outdegree()==0 and x.indegree()>0 and x.id not in g.outputs]
        # if there aren’t any “clean” sinks, allow any non‑input node as a source
        if not cands:
            cands = [x.id for x in g.get_nodes()
                    if x.indegree()>0 and x.id not in g.outputs]
        while len(g.outputs) < nout:
            if cands:
                s = cands.pop()
            else:
                sinks = [nid for nid in g.get_nodes_id()
                        if g.get_node_by_id(nid).outdegree()==0
                        and nid not in g.outputs]
                s = random.choice(sinks)
            g.add_output_node(s)


        # ensure no numeric labels were inserted by mistake
        def _relabel_all_nodes(g: open_digraph):
            """
            Give every node a fresh label that is consistent with its final
            in‑degree and out‑degree *and* with its role (input / output wrapper).
            """
            for nid, n in g.id_node_map().items():
                din, dout = n.indegree(), n.outdegree()

                if nid in g.inputs or nid in g.outputs:
                    n.set_label('')
                    continue

                if din == 1 and dout >= 1:
                    n.set_label('')
                elif din == 1 and dout == 1:
                    n.set_label('~')
                elif din >= 2 and dout == 1:
                    n.set_label(random.choice(['&', '|', '^']))
                elif din == 0 and dout == 0:
                    n.set_label(random.choice(['0', '1']))
                else:
                    n.set_label('')
        _relabel_all_nodes(g)
        c = cls(g)
        return c
    

