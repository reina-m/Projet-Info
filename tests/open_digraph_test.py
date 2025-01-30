import sys
import os
root = os.path.normpath(os.path.join(__file__, './../..'))
sys.path.append(root) # allow us to fetch files from the project root
import unittest
from modules.open_digraph import *

# classe pour le test des methodes __init__ : 

class InitTest(unittest.TestCase):
    def test_init_node(self):
        n0 = node(0, 'i', {}, {1:1})
        self.assertEqual(n0.id, 0)
        self.assertEqual(n0.label, 'i')
        self.assertEqual(n0.parents, {})
        self.assertEqual(n0.children, {1:1})
        self.assertIsInstance(n0, node)

        self.assertIsNot(n0, n0.copy())

    def test_init_open_digraph(self):
        # a -> b -> c
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertEqual(g.inputs, [0])
        self.assertEqual(g.outputs, [2])
        self.assertEqual(len(g.nodes), 3)

        # no need to test the rest as __init__ nodes has been tested
        self.assertEqual(g.nodes[0].id, 0)
        self.assertEqual(g.nodes[1].id, 1)
        self.assertEqual(g.nodes[2].id, 2)

        # test pour la copie : 
        self.assertIsNot(g, g.copy)
        for node_id in g.nodes:
            self.assertIsNot(g.copy().nodes[node_id], g.nodes[node_id])

if __name__ == '__main__': # the following code is called only when
    unittest.main()        # precisely this file is run