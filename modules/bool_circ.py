from .open_digraph import open_digraph
import random

class bool_circ(open_digraph):
    def __init__(self, graph):
        graph.assert_is_well_formed()
        super().__init__(graph.get_input_ids().copy(), graph.get_output_ids().copy(), [])
        self.nodes = graph.id_node_map().copy()
        if not self.is_well_formed():
            raise ValueError("Le graphe donné n'est pas un circuit booléen bien formé.")

    def is_well_formed(self):
        if self.is_cyclic():
            return False
        for node in self.get_nodes():
            nid = node.get_id()
            indegree, outdegree = node.indegree(), node.outdegree()
            # Input wrapper must have indegree=0, outdegree=1
            if nid in self.inputs:
                if indegree != 0 or outdegree != 1:
                    return False
                continue
            # Output wrapper must have indegree=1, outdegree=0
            if nid in self.outputs:
                if indegree != 1 or outdegree != 0:
                    return False
                continue
            label = node.get_label()
            # Copy nodes: one parent, at least one child
            if label == '':
                if indegree != 1 or outdegree < 1:
                    return False
            # Logic gates: one output
            elif label in ('&', '|', '^'):
                if outdegree != 1:
                    return False
            # NOT gates: one input, one output
            elif label == '~':
                if indegree != 1 or outdegree != 1:
                    return False
            # Constants: no outputs
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
            stack, curr, buf = [], None, ''
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
            if buf.strip():
                nid = g.add_node(label=buf.strip())
                if curr is not None:
                    g.add_edge(nid, curr)
                curr = nid
            # Wrap this subgraph output
            g.add_output_node(curr)
            outputs.append(g.get_output_ids()[-1])
        # Wrap all roots as inputs
        roots = [nid for nid in g.get_nodes_id() if g.get_node_by_id(nid).indegree() == 0]
        for r in roots:
            g.add_input_node(r)
        g.set_outputs(outputs)
        return cls(g)

    @classmethod
    def adder_n(cls, n):
        """
        n‑bit ripple‑carry adder:
        inputs → A₀…Aₙ₋₁, B₀…Bₙ₋₁, Cin
        outputs → Sum₀…Sumₙ₋₁, Cout
        """
        g = open_digraph.empty()

        # create the n data nodes for A and B, plus Cin
        a_nodes = [g.add_node() for _ in range(n)]
        b_nodes = [g.add_node() for _ in range(n)]
        cin    = g.add_node()

        # wrap them as inputs
        for a in a_nodes: g.add_input_node(a)
        for b in b_nodes: g.add_input_node(b)
        g.add_input_node(cin)

        # ripple‑carry logic
        ci = cin
        sum_nodes = []
        for ai, bi in zip(a_nodes, b_nodes):
            # t = ai XOR bi
            t = g.add_node(label='^', parents={ai:1, bi:1})
            # sum bit = t XOR ci
            si = g.add_node(label='^', parents={t:1, ci:1})
            sum_nodes.append(si)

            # carry = (ai AND bi) OR (t AND ci)
            c1 = g.add_node(label='&', parents={ai:1, bi:1})
            c2 = g.add_node(label='&', parents={t:1, ci:1})
            ci = g.add_node(label='|', parents={c1:1, c2:1})

        # wrap outputs: first the final carry, then the sum bits (MSB left)
        g.add_output_node(ci)
        for s in reversed(sum_nodes):
            g.add_output_node(s)

        return cls(g)


    @classmethod
    def half_adder_n(cls, n):
        """
        n‑bit half‑adder (Cin=0):
        inputs → A₀…Aₙ₋₁, B₀…Bₙ₋₁
        outputs → Sum₀…Sumₙ₋₁, Cout (overflow)
        """
        g = open_digraph.empty()

        # data nodes for A and B
        a_nodes = [g.add_node() for _ in range(n)]
        b_nodes = [g.add_node() for _ in range(n)]

        # wrap them as inputs
        for a in a_nodes: g.add_input_node(a)
        for b in b_nodes: g.add_input_node(b)

        # constant‑0 starting carry
        zero = g.add_node(label='0')  

        ci = zero
        sum_nodes = []
        for ai, bi in zip(a_nodes, b_nodes):
            # t = ai XOR bi
            t = g.add_node(label='^', parents={ai:1, bi:1})
            # sum bit = t XOR ci
            si = g.add_node(label='^', parents={t:1, ci:1})
            sum_nodes.append(si)

            # carry = (ai AND bi) OR (t AND ci)
            c1 = g.add_node(label='&', parents={ai:1, bi:1})
            c2 = g.add_node(label='&', parents={t:1, ci:1})
            ci = g.add_node(label='|', parents={c1:1, c2:1})

        # wrap outputs: overflow carry, then sum bits
        g.add_output_node(ci)
        for s in reversed(sum_nodes):
            g.add_output_node(s)

        return cls(g)


    @classmethod
    def random(cls, n, nin, nout):
        """
        Génère un bool_circ aléatoire acyclique de n nœuds,
        exactement nin entrées et nout sorties.
        """
        # Regénérer tant que le nombre de racines/sinks n'est pas correct
        while True:
            base = open_digraph.random_graph(n, bound=1, form="DAG")
            roots = [x.id for x in base.get_nodes() if x.indegree() == 0]
            sinks = [x.id for x in base.get_nodes() if x.outdegree() == 0 and x.indegree() > 0]
            if len(roots) == nin and len(sinks) == nout:
                g = base
                break
        # Nettoyer éventuels wrappers existants
        g.set_inputs([])
        g.set_outputs([])
        ops = ['&', '|', '^']
        # Assigner labels et gérer fan‑out
        for x in g.get_nodes():
            din, dout = x.indegree(), x.outdegree()
            if din == 0:
                x.set_label('')
            elif din == 1 and dout == 1:
                x.set_label('~')
            elif din == 1 and dout > 1:
                x.set_label('')
            elif din > 1 and dout == 1:
                x.set_label(random.choice(ops))
            elif din > 1 and dout > 1:
                x.set_label(random.choice(ops))
                targets = list(x.get_children().items())
                extra = targets[1:]
                for tgt, _ in extra:
                    g.remove_parallel_edges(x.id, tgt)
                cp = g.add_node('', {x.id: 1}, {})
                for tgt, m in extra:
                    g.add_edge(cp, tgt, m)
        # Wrap exactly those roots and sinks
        for r in roots:
            g.add_input_node(r)
        for s in sinks:
            g.add_output_node(s)
        # Relabel pour consistance
        def _relabel_all(g):
            for nid, node in g.id_node_map().items():
                din, dout = node.indegree(), node.outdegree()
                if nid in g.inputs or nid in g.outputs:
                    node.set_label('')
                elif din == 1 and dout >= 1:
                    node.set_label('')
                elif din == 1 and dout == 1:
                    node.set_label('~')
                elif din >= 2 and dout == 1:
                    node.set_label(random.choice(ops))
                elif din == 0 and dout == 0:
                    node.set_label(random.choice(['0','1']))
                else:
                    node.set_label('')
        _relabel_all(g)
        return cls(g)
