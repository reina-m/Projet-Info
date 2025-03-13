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

class StrReprTest(unittest.TestCase):
    def test_str_node(self):
        n = node(1, 'test', {0:1}, {2:1})
        expected = "Node(id=1, label='test', parents={0: 1}, children={2: 1})"
        actual = str(n)
        print("Expected:", expected)
        print("Actual  :", actual)
        self.assertEqual(actual, expected)

    def test_str_open_digraph(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        expected = (
            "OpenDigraph(\n"
            "  inputs=[0], \n"
            "  outputs=[2], \n"
            "  nodes=\n"
            "  Node(id=0, label='a', parents={}, children={1: 1})\n"
            "  Node(id=1, label='b', parents={0: 1}, children={2: 1})\n"
            "  Node(id=2, label='c', parents={1: 1}, children={})\n"
            ")"
        )
        actual = str(g)
        print("Expected:\n", expected)
        print("Actual:\n", actual)
        self.assertEqual(actual, expected)

class EmptyGraphTest(unittest.TestCase):
    def test_empty_graph(self):
        g = open_digraph.empty()
        
        # Vérifications attendues
        self.assertEqual(g.inputs, [])  # Vérifie que les entrées sont vides
        self.assertEqual(g.outputs, [])  # Vérifie que les sorties sont vides
        self.assertEqual(g.nodes, {})  # Vérifie que le dictionnaire de nœuds est vide

class CopyTest(unittest.TestCase):
    def test_copy_node(self):
        n = node(1, 'test', {0:1}, {2:1})
        n_copy = n.copy()

        self.assertEqual(n.id, n_copy.id)
        self.assertEqual(n.label, n_copy.label)
        self.assertEqual(n.parents, n_copy.parents)
        self.assertEqual(n.children, n_copy.children)
        self.assertIsNot(n, n_copy)  # ensure they are different objects

    def test_copy_graph(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])
        g_copy = g.copy()

        self.assertEqual(g.inputs, g_copy.inputs)
        self.assertEqual(g.outputs, g_copy.outputs)
        self.assertEqual(len(g.nodes), len(g_copy.nodes))
        self.assertIsNot(g, g_copy)  # ensure they are different objects

class GettersTest(unittest.TestCase):
    def test_getters(self):
        n = node(1, 'test', {0:1}, {2:1})

        self.assertEqual(n.get_id(), 1)
        self.assertEqual(n.get_label(), 'test')
        self.assertEqual(n.get_parents(), {0: 1})
        self.assertEqual(n.get_children(), {2: 1})

class SettersTest(unittest.TestCase):
    def test_setters(self):
        n = node(1, 'test', {0:1}, {2:1})

        n.set_id(10)
        self.assertEqual(n.get_id(), 10)

        n.set_label('new_label')
        self.assertEqual(n.get_label(), 'new_label')

        n.set_parents({5:2})
        self.assertEqual(n.get_parents(), {5: 2})

        n.set_children({7:3})
        self.assertEqual(n.get_children(), {7: 3})

class NewIDTest(unittest.TestCase):
    def test_new_id(self):
        g = open_digraph.empty()
        self.assertEqual(g.new_id(), 1)  # graph is empty, should return 1

        g.nodes = {1: node(1, 'A', {}, {}), 2: node(2, 'B', {}, {})}
        self.assertEqual(g.new_id(), 3)  # should return max id + 1

class AddEdgeTest(unittest.TestCase):
    def test_add_edge(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        g.add_edge(0, 2)

        self.assertIn(2, g.nodes[0].get_children())  # 0 should have 2 as child
        self.assertIn(0, g.nodes[2].get_parents())   # 2 should have 0 as parent

    def test_add_edge_invalid(self):
        g = open_digraph.empty()
        with self.assertRaises(ValueError):
            g.add_edge(1, 2)  # Should fail since 1 and 2 are not in the graph

class AddNodeTest(unittest.TestCase):
    def test_add_node(self):
        g = open_digraph.empty()
        new_id = g.add_node(label="test", parents={}, children={})

        self.assertIn(new_id, g.nodes)  # ensure the node was added
        self.assertEqual(g.nodes[new_id].get_label(), "test")

    def test_add_node_with_parents_children(self):
        g = open_digraph.empty()
        g.add_node("A")  # First node
        g.add_node("B")  # Second node

        new_id = g.add_node("C", parents={1:1}, children={2:1})

        self.assertIn(new_id, g.nodes)  # new node exists
        self.assertIn(new_id, g.nodes[1].get_children())  # parent-child link
        self.assertIn(1, g.nodes[new_id].get_parents())  # reverse link

if __name__ == '__main__': # the following code is called only when
    unittest.main()        # precisely this file is run