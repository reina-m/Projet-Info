from .open_digraph import open_digraph
import random

class bool_circ(open_digraph):
    def __init__(self, graph):
        graph.assert_is_well_formed()
        if graph is None:
            graph = open_digraph.empty()
        #super().__init__(graph.inputs, graph.outputs, graph.nodes)
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
    def empty(cls):
        return cls()

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
            g.add_output_node(curr)
            outputs.append(g.get_output_ids()[-1])
        roots = [nid for nid in g.get_nodes_id() if g.get_node_by_id(nid).indegree() == 0]
        for r in roots:
            g.add_input_node(r)
        g.set_outputs(outputs)
        return cls(g)

    @classmethod
    def adder_n(cls, n):
        g = open_digraph.empty()
        a_nodes = [g.add_node() for _ in range(n)]
        b_nodes = [g.add_node() for _ in range(n)]
        cin = g.add_node()

        for a in a_nodes: g.add_input_node(a)
        for b in b_nodes: g.add_input_node(b)
        g.add_input_node(cin)

        ci = cin
        sum_nodes = []
        for ai, bi in zip(a_nodes, b_nodes):
            t = g.add_node(label='^', parents={ai:1, bi:1})
            si = g.add_node(label='^', parents={t:1, ci:1})
            sum_nodes.append(si)

            c1 = g.add_node(label='&', parents={ai:1, bi:1})
            c2 = g.add_node(label='&', parents={t:1, ci:1})
            ci = g.add_node(label='|', parents={c1:1, c2:1})

        cout_output = g.add_node()
        g.add_edge(ci, cout_output)
        g.add_output_node(cout_output)
        
        for s in reversed(sum_nodes):
            sum_output = g.add_node()
            g.add_edge(s, sum_output)
            g.add_output_node(sum_output)
        g.assert_is_well_formed()

        return cls(g)

    @classmethod
    def half_adder_n(cls, n):
        g = open_digraph.empty()
        a_nodes = [g.add_node() for _ in range(n)]
        b_nodes = [g.add_node() for _ in range(n)]

        for a in a_nodes: g.add_input_node(a)
        for b in b_nodes: g.add_input_node(b)

        zero = g.add_node(label='0')  
        ci = zero
        sum_nodes = []
        for ai, bi in zip(a_nodes, b_nodes):
            t = g.add_node(label='^', parents={ai:1, bi:1})
            si = g.add_node(label='^', parents={t:1, ci:1})
            sum_nodes.append(si)

            c1 = g.add_node(label='&', parents={ai:1, bi:1})
            c2 = g.add_node(label='&', parents={t:1, ci:1})
            ci = g.add_node(label='|', parents={c1:1, c2:1})

        cout_output = g.add_node()
        g.add_edge(ci, cout_output)
        g.add_output_node(cout_output)
        
        for s in reversed(sum_nodes):
            sum_output = g.add_node()
            g.add_edge(s, sum_output)
            g.add_output_node(sum_output)

        return cls(g)

    @classmethod
    def random(cls, n, nin, nout):
        while True:
            base = open_digraph.random_graph(n, bound=1, form="DAG")
            roots = [x.id for x in base.get_nodes() if x.indegree() == 0]
            sinks = [x.id for x in base.get_nodes() if x.outdegree() == 0 and x.indegree() > 0]
            if len(roots) == nin and len(sinks) == nout:
                g = base
                break
        g.set_inputs([])
        g.set_outputs([])
        ops = ['&', '|', '^']
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
        for r in roots:
            g.add_input_node(r)
        for s in sinks:
            g.add_output_node(s)
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
        g = open_digraph.empty()
        a = [g.add_node() for _ in range(4)]
        b = [g.add_node() for _ in range(4)]
        c0 = g.add_node()
        
        for nid in a + b:
            g.add_input_node(nid)
        g.add_input_node(c0)
        
        g_nodes = [g.add_node(label='&', parents={ai:1, bi:1}) for ai, bi in zip(a,b)]
        p_nodes = [g.add_node(label='^', parents={ai:1, bi:1}) for ai, bi in zip(a,b)]

        carries = [c0]
        for i in range(4):
            pi = p_nodes[i]
            gi = g_nodes[i]
            t = g.add_node(label='&', parents={pi:1, carries[-1]:1})
            ci = g.add_node(label='|', parents={gi:1, t:1})
            carries.append(ci)

        r = [g.add_node(label='^', parents={p_nodes[i]:1, carries[i]:1}) for i in range(4)]

        cout_output = g.add_node(label='')
        g.add_edge(carries[4], cout_output)
        g.add_output_node(cout_output)
        
        for ri in reversed(r):
            sum_output = g.add_node(label='')
            g.add_edge(ri, sum_output)
            g.add_output_node(sum_output)
        g.assert_is_well_formed()
        
        return cls(g)

    @classmethod
    def cla4n(cls, n):
        if n < 1:
            raise ValueError("n doit être ≥ 1")

        g = open_digraph.empty()
        A = [g.add_node() for _ in range(4*n)]
        B = [g.add_node() for _ in range(4*n)]
        c0 = g.add_node()

        for x in A + B:
            g.add_input_node(x)
        g.add_input_node(c0)

        carry = c0
        sums = []

        for i in range(n):
            block = open_digraph.empty()
            a_block = [block.add_node() for _ in range(4)]
            b_block = [block.add_node() for _ in range(4)]
            cin_block = block.add_node()
            
            for nid in a_block + b_block:
                block.add_input_node(nid)
            block.add_input_node(cin_block)
            
            g_nodes = [block.add_node(label='&', parents={ai:1, bi:1}) 
                      for ai, bi in zip(a_block, b_block)]
            p_nodes = [block.add_node(label='^', parents={ai:1, bi:1}) 
                      for ai, bi in zip(a_block, b_block)]

            carries_block = [cin_block]
            for j in range(4):
                pj = p_nodes[j]
                gj = g_nodes[j]
                t = block.add_node(label='&', parents={pj:1, carries_block[-1]:1})
                cj = block.add_node(label='|', parents={gj:1, t:1})
                carries_block.append(cj)

            r = [block.add_node(label='^', parents={p_nodes[j]:1, carries_block[j]:1}) 
                for j in range(4)]

            cout_output = block.add_node()
            block.add_edge(carries_block[4], cout_output)
            block.add_output_node(cout_output)
            
            for rj in reversed(r):
                sum_output = block.add_node()
                block.add_edge(rj, sum_output)
                block.add_output_node(sum_output)
            
            shift = g.iparallel(block)
            
            a_blk = A[4*i:4*i+4]
            b_blk = B[4*i:4*i+4]
            
            for j in range(4):
                g.add_edge(a_blk[j], a_block[j] + shift)
                g.add_edge(b_blk[j], b_block[j] + shift)
            g.add_edge(carry, cin_block + shift)
            
            carry = cout_output + shift
            
            for s in block.get_output_ids()[1:]:
                sums.append(s + shift)

        out_carry = g.add_node()
        g.add_edge(carry, out_carry)
        g.add_output_node(out_carry)
        
        for s in reversed(sums):
            out_sum = g.add_node()
            g.add_edge(s, out_sum)
            g.add_output_node(out_sum)
        g.assert_is_well_formed()

        return cls(g)

    @staticmethod
    def estimate_depth_and_gates(circ):
        logic_nodes = {'&', '|', '^', '~'}
        gates = 0
        
        if circ.is_cyclic():
            max_depth = float('inf')
        else:
            max_depth = 0
            for nid in circ.get_nodes_id():
                node = circ.get_node_by_id(nid)
                if node.label in logic_nodes:
                    d = circ.node_depth(nid)
                    max_depth = max(max_depth, d)
        
        for nid in circ.get_nodes_id():
            node = circ.get_node_by_id(nid)
            if node.label in logic_nodes:
                gates += 1
                
        return max_depth, gates

    @staticmethod
    def adjust_io(graph, nin, nout):
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
        g = open_digraph.empty()
        bits = bin(value)[2:].zfill(size)
        
        # Create constant nodes with proper labels
        for bit in reversed(bits):
            const_node = g.add_node(label=bit)
            output_node = g.add_node(label=bit)  # Output node gets same label
            g.add_edge(const_node, output_node)
            g.add_output_node(output_node)
        g.assert_is_well_formed()    
        return cls(g)

    def simplify_not(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '~':
            return False
        
        # Get the single child (constant node)
        child_id = next(iter(node.get_children()))
        child_node = self.get_node_by_id(child_id)
        
        if child_node.label in ('0', '1'):
            new_val = '1' if child_node.label == '0' else '0'
            new_id = self.add_node(label=new_val)
            
            # Redirect parents
            for pid in node.get_parents():
                self.add_edge(pid, new_id)
                
            # Handle outputs
            for out_id in self.outputs:
                if out_id == child_id:
                    self.set_outputs([new_id if x == out_id else x for x in self.outputs])
                    
            self.remove_nodes_by_id([nid, child_id])
            return True
        return False

    def simplify_and(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '&':
            return False
            
        values = [self.get_node_by_id(pid).label for pid in node.get_parents()]
        
        if '0' in values:
            zero = self.add_node(label='0')
            for cid in node.get_children():
                self.add_edge(zero, cid)
            if nid in self.outputs:
                self.set_outputs([zero if x == nid else x for x in self.outputs])
            self.remove_node_by_id(nid)
            return True
            
        elif all(v == '1' for v in values):
            one = self.add_node(label='1')
            for cid in node.get_children():
                self.add_edge(one, cid)
            if nid in self.outputs:
                self.set_outputs([one if x == nid else x for x in self.outputs])
            self.remove_node_by_id(nid)
            return True
        return False

    def simplify_or(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '|':
            return False
            
        values = [self.get_node_by_id(pid).label for pid in node.get_parents()]
        
        if '1' in values:
            one = self.add_node(label='1')
            for cid in node.get_children():
                self.add_edge(one, cid)
            if nid in self.outputs:
                self.set_outputs([one if x == nid else x for x in self.outputs])
            self.remove_node_by_id(nid)
            return True
            
        elif all(v == '0' for v in values):
            zero = self.add_node(label='0')
            for cid in node.get_children():
                self.add_edge(zero, cid)
            if nid in self.outputs:
                self.set_outputs([zero if x == nid else x for x in self.outputs])
            self.remove_node_by_id(nid)
            return True
        return False

    def simplify_xor(self, nid):
        node = self.get_node_by_id(nid)
        if node.label != '^':
            return False
            
        values = [self.get_node_by_id(pid).label for pid in node.get_parents()]
        
        if all(v in ('0', '1') for v in values):
            result = 0
            for v in values:
                result ^= int(v)
                
            new_id = self.add_node(label=str(result))
            for cid in node.get_children():
                self.add_edge(new_id, cid)
            if nid in self.outputs:
                self.set_outputs([new_id if x == nid else x for x in self.outputs])
            self.remove_node_by_id(nid)
            return True
        return False

    def evaluate(self):
        changed = True
        while changed:
            changed = False
            nodes_to_check = [n.id for n in self.get_nodes() if n.id not in self.outputs]
            
            for nid in nodes_to_check:
                if nid not in self.nodes:
                    continue
                    
                node = self.get_node_by_id(nid)
                if node.label == '~':
                    changed |= self.simplify_not(nid)
                elif node.label == '&':
                    changed |= self.simplify_and(nid)
                elif node.label == '|':
                    changed |= self.simplify_or(nid)
                elif node.label == '^':
                    changed |= self.simplify_xor(nid)

        return self


    @classmethod
    def hamming_encoder(cls):
        g = cls(inputs=[], outputs=[], nodes=[])
        
        # Input nodes (4 bits de données)
        d0 = g.add_node(label='')
        d1 = g.add_node(label='')
        d2 = g.add_node(label='') 
        d3 = g.add_node(label='')
        
        # Ajout des inputs
        g.add_input_node(d0)
        g.add_input_node(d1)
        g.add_input_node(d2)
        g.add_input_node(d3)
        
        # Calcul des bits de parité
        p0 = g.add_node(label='^', parents={d0:1, d1:1, d2:1})
        p1 = g.add_node(label='^', parents={d0:1, d1:1, d3:1})
        p2 = g.add_node(label='^', parents={d0:1, d2:1, d3:1})
        
        # Output nodes (7 bits: p0, p1, d0, p2, d1, d2, d3)
        outputs_order = [p0, p1, d0, p2, d1, d2, d3]
        for out in outputs_order:
            out_node = g.add_node()
            g.add_edge(out, out_node)
            g.add_output_node(out_node)
            
        return g

    @classmethod
    def hamming_decoder(cls):
        """Décodeur Hamming(7,4) - 7 bits reçus → 4 bits corrigés"""
        g = cls.empty()
        
        # Input nodes (7 bits reçus)
        r = [g.add_node(label='') for _ in range(7)]
        for node_id in r:
            g.add_input_node(node_id)
        
        # Calcul des syndromes (s0, s1, s2)
        s0 = g.add_node(label='^', parents={r[0]:1, r[2]:1, r[4]:1, r[6]:1})
        s1 = g.add_node(label='^', parents={r[1]:1, r[2]:1, r[5]:1, r[6]:1})
        s2 = g.add_node(label='^', parents={r[3]:1, r[4]:1, r[5]:1, r[6]:1})
        
        # Calcul de la position d'erreur (bit index)
        pos = g.add_node(label='|', parents={s0:1, s1:1, s2:1})
        
        # Correction des bits de données
        corrected = [
            g.add_node(label='^', parents={r[2]:1, pos:1}),  # d0
            g.add_node(label='^', parents={r[4]:1, pos:1}),  # d1
            g.add_node(label='^', parents={r[5]:1, pos:1}),  # d2
            g.add_node(label='^', parents={r[6]:1, pos:1})   # d3
        ]
        
        # Output nodes (4 bits de données corrigés)
        for data in corrected:
            out_node = g.add_node()
            g.add_edge(data, out_node)
            g.add_output_node(out_node)
        
        return g

    # Règles de réécriture supplémentaires
    def apply_xor_associativity(self, nid):
        """Associativité du XOR"""
        """Règle d'associativité du XOR."""
        node = self.get_node_by_id(nid)
        if node.label != '^':
            return False
        xor_parents = [pid for pid in node.parents 
                    if self.get_node_by_id(pid).label == '^']
        if not xor_parents:
            return False
        pid = xor_parents[0]
        parent = self.get_node_by_id(pid)

        for cid, mult in parent.children.items():
            if cid != nid:
                self.add_edge(pid, cid, mult)
                self.remove_parallel_edges(parent.id, cid)
        for gpid, mult in parent.parents.items():
            self.add_edge(gpid, nid, mult)
            self.remove_parallel_edges(gpid, pid)

        self.remove_node_by_id(pid)
        return True 


    def apply_copy_associativity(self, nid):
        """Règle d'associativité des copies."""
        node = self.get_node_by_id(nid)
        if node.label != '':
            return False
        copy_children = [cid for cid in node.children 
                        if self.get_node_by_id(cid).label == '']
        if not copy_children:
            return False
        cid = copy_children[0]
        child = self.get_node_by_id(cid)
        for pcid, mult in child.parents.items():
            if pcid != nid:
                self.add_edge(pcid, nid, mult)
                self.remove_parallel_edges(pcid, cid)
        for gcid, mult in child.children.items():
            self.add_edge(nid, gcid, mult)
            self.remove_parallel_edges(cid, gcid)
        self.remove_node_by_id(cid)
        return True

    def apply_xor_involution(self, nid):
        """XOR(x, x) = 0 (involution)"""
        node = self.get_node_by_id(nid)
        if node.label != '^':
            return False
        for pid in list(node.parents.keys()):
            mult = node.parents[pid]
            if mult % 2 == 0:
                self.remove_several_parallel_edges([(pid, nid)])
            else:
                self.remove_several_parallel_edges([(pid, nid)])
                self.add_edge(pid, nid, 1)
            return True
        return False

    '''

    def apply_xor_involution(self, nid):
        """Règle: XOR(x,x,...,x) = x si nombre impair, 0 sinon"""
        node = self.get_node_by_id(nid)
        if node.label != '^':
            return False
        
        # Compter les arêtes multiples entre mêmes nœuds
        edge_counts = {}
        for pid in node.parents:
            if pid in edge_counts:
                edge_counts[pid] += node.parents[pid]
            else:
                edge_counts[pid] = node.parents[pid]
        
        changed = False
        for pid, count in list(edge_counts.items()):
            if count >= 2:
                changed = True
                # Supprimer toutes les arêtes entre ces nœuds
                self.remove_several_parallel_edges([(pid, nid)])
                # Si nombre impair, remettre une arête
                if count % 2 == 1:
                    self.add_edge(pid, nid, 1)
        
        return changed

'''
    def apply_not_involution(self, nid):
        """~~x = x (double négation)"""
        node = self.get_node_by_id(nid)
        if node.label != '~':
            return False
        if len(node.parents) != 1 or len(node.children) != 1:
            return False
        parent_id = next(iter(node.parents))
        child_id = next(iter(node.children))
        if self.get_node_by_id(parent_id).label == '~':
            for gpid, mult in self.get_node_by_id(parent_id).parents.items():
                self.add_edge(gpid, child_id, mult)
            self.remove_nodes_by_id([parent_id, nid])
            return True
        return False

    def apply_erasure(self, nid):
        """Effacement de constantes inutiles."""
        node = self.get_node_by_id(nid)
        if node.label not in ['0', '1']:
            return False
        if len(node.children) > 1:
            child_to_keep = next(iter(node.children))
            for cid in list(node.children.keys()):
                if cid != child_to_keep:
                    self.remove_parallel_edges(nid, cid)
            return True
        return False

    def simplify_all(self):
        """Applique toutes les règles de simplification jusqu'à point fixe"""
        changed = True
        while changed:
            changed = False
            nodes = self.get_nodes_id()
            for nid in nodes:
                if nid not in self.nodes:
                    continue
                node = self.get_node_by_id(nid)
                if node.label == '^':
                    changed |= self.apply_xor_involution(nid)
                    changed |= self.apply_xor_associativity(nid)
                    changed |= self.simplify_xor(nid)
                elif node.label == '':
                    changed |= self.apply_copy_associativity(nid)
                elif node.label == '~':
                    changed |= self.apply_not_involution(nid)
                    changed |= self.simplify_not(nid)
                elif node.label in ['0', '1']:
                    changed |= self.apply_erasure(nid)
                elif node.label == '&':
                    changed |= self.simplify_and(nid)
                elif node.label == '|':
                    changed |= self.simplify_or(nid)
 

    def evaluate_with_inputs(self, inputs):
        """Évalue le circuit avec les inputs donnés"""
        if len(inputs) != len(self.inputs):
            raise ValueError("Nombre d'entrées incorrect")
        
        for i, val in zip(self.inputs, inputs):
            self.get_node_by_id(i).label = str(val)
        self.evaluate()
        
     
        return [int(self.get_node_by_id(o).label) for o in self.outputs]

    def display_hamming(self):
        """Affiche le circuit Hamming"""
        self.display(verbose=True, filename_prefix="hamming_circuit")