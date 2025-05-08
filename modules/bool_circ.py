from .open_digraph import open_digraph
import random

class bool_circ(open_digraph):
    def __init__(self, graph):
        graph.assert_is_well_formed()
        if graph.is_cyclic():
            raise ValueError("Le graphe ne peut pas être cyclique pour un circuit booléen")
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

    @classmethod
    def cla4(cls):
        """
        Construit un Carry-Lookahead Adder (CLA) 4 bits
        Entrées : a0-a3, b0-b3, c0 (bit de retenue initial)
        Sorties : r0-r3, c4 (carry out)
        """
        g = open_digraph.empty()
        a = [g.add_node() for _ in range(4)]
        b = [g.add_node() for _ in range(4)]
        c0 = g.add_node()
        
        for nid in a + b + [c0]:
            g.add_input_node(nid)
        
        # g_i = a_i & b_i ; p_i = a_i ^ b_i
        g_nodes = [g.add_node(label='&', parents={ai:1, bi:1}) for ai, bi in zip(a,b)]
        p_nodes = [g.add_node(label='^', parents={ai:1, bi:1}) for ai, bi in zip(a,b)]

        # Calcul des c_i = g_i | (p_i & c_i-1)
        carries = [c0]  # c0 est donné
        for i in range(4):
            pi = p_nodes[i]
            gi = g_nodes[i]
            t = g.add_node(label='&', parents={pi:1, carries[-1]:1})
            ci = g.add_node(label='|', parents={gi:1, t:1})
            carries.append(ci)

        # r_i = p_i ^ c_i
        r = [g.add_node(label='^', parents={p_nodes[i]:1, carries[i]:1}) for i in range(4)]

        g.add_output_node(carries[4])  # c4
        for ri in reversed(r):
            g.add_output_node(ri)

        return cls(g)

    @classmethod
    def cla4n(cls, n):
        """
        Construit un additionneur CLA composé de blocs de 4 bits, pour 4n bits
        Entrées : a0-a(4n-1), b0-b(4n-1), c0
        Sorties : r0-r(4n-1), c_{4n}
        """
        if n < 1:
            raise ValueError("n doit être ≥ 1")

        g = open_digraph.empty()
        A = [g.add_node() for _ in range(4*n)]
        B = [g.add_node() for _ in range(4*n)]
        c0 = g.add_node()

        for x in A + B + [c0]:
            g.add_input_node(x)

        carry = c0
        sums = []

        for i in range(n):
            cla = cls.cla4()
            shift = g.iparallel(cla)
            a_blk = A[4*i:4*i+4]
            b_blk = B[4*i:4*i+4]

            # Connect inputs
            for src, tgt in zip(a_blk, cla.get_input_ids()[0:4]):
                g.add_edge(src, tgt + shift)
            for src, tgt in zip(b_blk, cla.get_input_ids()[4:8]):
                g.add_edge(src, tgt + shift)
            g.add_edge(carry, cla.get_input_ids()[8] + shift)  # carry-in

            carry = cla.get_output_ids()[0] + shift
            sums.extend([x + shift for x in cla.get_output_ids()[1:]])

        g.add_output_node(carry)
        for s in reversed(sums):
            g.add_output_node(s)

        return cls(g)

    @staticmethod
    def estimate_depth_and_gates(circ):
        """
        Retourne la profondeur et le nombre de portes logiques d'un circuit.
        → Profondeur = profondeur max (longest path)
        → Nombre de portes = nombre de noeuds logiques (&, |, ^, ~)
        """
        logic_nodes = {'&', '|', '^', '~'}
        gates = 0
        
        # Vérifier si le graphe est cyclique
        if circ.is_cyclic():
            # Dans le cas d'un graphe cyclique, on ne peut pas calculer la profondeur
            max_depth = float('inf')
        else:
            max_depth = 0
            for nid in circ.get_nodes_id():
                node = circ.get_node_by_id(nid)
                if node.label in logic_nodes:
                    d = circ.node_depth(nid)
                    max_depth = max(max_depth, d)
        
        # Comptage des portes logiques
        for nid in circ.get_nodes_id():
            node = circ.get_node_by_id(nid)
            if node.label in logic_nodes:
                gates += 1
                
        return max_depth, gates

    @staticmethod
    def adjust_io(graph, nin, nout):
        """
        Ajuste les entrées/sorties du graphe donné pour correspondre à nin et nout.
        """
        while len(graph.get_input_ids()) < nin:
            tgt = random.choice(graph.get_nodes_id())
            graph.add_input_node(tgt)
        while len(graph.get_input_ids()) > nin:
            i1, i2 = random.sample(graph.get_input_ids(), 2)
            join = graph.add_node()
            graph.add_edge(join, i1)
            graph.add_edge(join, i2)
            graph.set_inputs([i for i in graph.get_input_ids() if i not in (i1, i2)] + [join])

        while len(graph.get_output_ids()) < nout:
            src = random.choice(graph.get_nodes_id())
            graph.add_output_node(src)
        while len(graph.get_output_ids()) > nout:
            o1, o2 = random.sample(graph.get_output_ids(), 2)
            join = graph.add_node()
            graph.add_edge(o1, join)
            graph.add_edge(o2, join)
            graph.set_outputs([o for o in graph.get_output_ids() if o not in (o1, o2)] + [join])

        return graph


    @classmethod
    def from_int(cls, value, size=8):
        """
        Construit un circuit représentant un entier binaire fixé (non variable).
        Chaque bit est un nœud avec label '0' ou '1'.
        """
        g = open_digraph.empty()
        bits = bin(value)[2:].zfill(size)
        for bit in bits:
            nid = g.add_node(label=bit)
            g.add_output_node(nid)
        return cls(g)

    def simplify_not(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '~':
            return False
        child = next(iter(node.get_children()))
        label = self.get_node_by_id(child).label
        if label in ('0', '1'):
            new_val = '1' if label == '0' else '0'
            new_id = self.add_node(label=new_val)
            for pid in node.get_parents():
                self.add_edge(pid, new_id)
            self.remove_nodes_by_id([nid, child])
            return True
        return False

    def simplify_and(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '&':
            return False
        values = [self.get_node_by_id(cid).label for cid in node.get_parents()]
        if '0' in values:
            zero = self.add_node(label='0')
            for cid in node.get_children():
                self.add_edge(zero, cid)
            self.remove_node_by_id(nid)
            return True
        elif all(v == '1' for v in values):
            one = self.add_node(label='1')
            for cid in node.get_children():
                self.add_edge(one, cid)
            self.remove_node_by_id(nid)
            return True
        return False

    def simplify_or(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '|':
            return False
        values = [self.get_node_by_id(cid).label for cid in node.get_parents()]
        if '1' in values:
            one = self.add_node(label='1')
            for cid in node.get_children():
                self.add_edge(one, cid)
            self.remove_node_by_id(nid)
            return True
        elif all(v == '0' for v in values):
            zero = self.add_node(label='0')
            for cid in node.get_children():
                self.add_edge(zero, cid)
            self.remove_node_by_id(nid)
            return True
        return False

    def simplify_xor(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '^':
            return False
        values = [self.get_node_by_id(cid).label for cid in node.get_parents()]
        if all(v in ('0', '1') for v in values):
            result = 0
            for v in values:
                result ^= int(v)
            new_id = self.add_node(label=str(result))
            for cid in node.get_children():
                self.add_edge(new_id, cid)
            self.remove_node_by_id(nid)
            return True
        return False

    def evaluate(self):
        """
        Applique les règles de simplification tant qu'il y a des co-feuilles
        qui ne sont pas connectées directement aux sorties.
        """
        changed = True
        while changed:
            changed = False
            leaves = [n.id for n in self.get_nodes() if not n.get_children() and n.id not in self.get_output_ids()]
            for nid in leaves:
                node = self.get_node_by_id(nid)
                if node.label in ('0', '1'):
                    self.remove_node_by_id(nid)
                    changed = True
                    continue
                if node.label == '~':
                    changed |= self.simplify_not(nid)
                elif node.label == '&':
                    changed |= self.simplify_and(nid)
                elif node.label == '|':
                    changed |= self.simplify_or(nid)
                elif node.label == '^':
                    changed |= self.simplify_xor(nid)

        return self
