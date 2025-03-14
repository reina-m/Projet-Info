import sys
import os
import random

root = os.path.normpath(os.path.join(__file__, './../..'))
sys.path.append(root) # allow us to fetch files from the project root
import unittest
from modules.open_digraph import *

# classe pour le test des methodes __init__ : 
#############################################
##             TESTS POUR TD1              ##
#############################################

# EXO 3

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

# EXO 4
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

# EXO 5
class EmptyGraphTest(unittest.TestCase):
    def test_empty_graph(self):
        g = open_digraph.empty()
        self.assertEqual(g.inputs, [])
        self.assertEqual(g.outputs, [])
        self.assertEqual(g.nodes, {})

# EXO 6
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

# EXOS 7 - 8
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

# EXO 10
class NewIDTest(unittest.TestCase):
    def test_new_id(self):
        g = open_digraph.empty()
        self.assertEqual(g.new_id(), 1)  # graph is empty, should return 1
        g.nodes = {1: node(1, 'A', {}, {}), 2: node(2, 'B', {}, {})}
        self.assertEqual(g.new_id(), 3)  # should return max id + 1

# EXO 11
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
            g.add_edge(1, 2)  # should fail since 1 and 2 are not in the graph

class AddEdgesTest(unittest.TestCase):
    def test_add_edges(self):
        # create a small graph
        n0 = node(0, 'A', {}, {1: 1})
        n1 = node(1, 'B', {0: 1}, {2: 1})
        n2 = node(2, 'C', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        # add multiple edges at once
        edges = [(0, 2), (1, 2)]
        multiplicities = [2, 3]

        g.add_edges(edges, multiplicities)

        # check if the edges are correctly added with their multiplicities
        self.assertEqual(g.nodes[0].get_children()[2], 2)  # (0 -> 2) with multiplicity 2
        self.assertEqual(g.nodes[2].get_parents()[0], 2)   # (2 <- 0) with multiplicity 2

        self.assertEqual(g.nodes[1].get_children()[2], 3)  # (1 -> 2) with multiplicity 3
        self.assertEqual(g.nodes[2].get_parents()[1], 3)   # (2 <- 1) with multiplicity 3

    def test_add_edges_default_multiplicity(self):
        # create a small graph
        n0 = node(0, 'A', {}, {1: 1})
        n1 = node(1, 'B', {0: 1}, {2: 1})
        n2 = node(2, 'C', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        # add multiple edges with default multiplicity (1)
        edges = [(0, 2), (1, 2)]
        g.add_edges(edges)

        # check if the edges are correctly added with default multiplicity 1
        self.assertEqual(g.nodes[0].get_children()[2], 1)
        self.assertEqual(g.nodes[2].get_parents()[0], 1)

        self.assertEqual(g.nodes[1].get_children()[2], 1)
        self.assertEqual(g.nodes[2].get_parents()[1], 1)

    def test_add_edges_invalid_multiplicity_length(self):
        g = open_digraph([], [], [])

        # should raise ValueError when mult list length doesn't match edges list length
        with self.assertRaises(ValueError):
            g.add_edges([(0, 1), (1, 2)], [1])  # only one multiplicity given for two edges

# EXO 12
class AddNodeTest(unittest.TestCase):
    def test_add_node(self):
        g = open_digraph.empty()
        new_id = g.add_node(label="test", parents={}, children={})

        self.assertIn(new_id, g.nodes)  # ensure the node was added
        self.assertEqual(g.nodes[new_id].get_label(), "test")

    def test_add_node_with_parents_children(self):
        g = open_digraph.empty()
        g.add_node("A")
        g.add_node("B")
        new_id = g.add_node("C", parents={1:1}, children={2:1})
        self.assertIn(new_id, g.nodes)  # new node exists
        self.assertIn(new_id, g.nodes[1].get_children())  # parent-child link
        self.assertIn(1, g.nodes[new_id].get_parents())  # reverse link

#############################################
##          FIN TESTS POUR TD1             ##
#############################################

#############################################
##            TESTS POUR TD2               ##
#############################################

# EXO 1
class TestNodeMethods(unittest.TestCase):

    def setUp(self):
        """Initial setup before each test case."""
        self.node = node(1, "A", {2: 2, 3: 1}, {4: 2, 5: 1})  # example node

    def test_remove_parent_once(self):
        """Test that remove_parent_once correctly decreases multiplicity or removes parent."""
        self.node.remove_parent_once(2)
        self.assertEqual(self.node.parents, {2: 1, 3: 1})  # 2 should have multiplicity 1

        self.node.remove_parent_once(2)
        self.assertEqual(self.node.parents, {3: 1})  # 2 should be fully removed

        self.node.remove_parent_once(3)
        self.assertEqual(self.node.parents, {})  # 3 removed, no parents left

    def test_remove_child_once(self):
        """Test that remove_child_once correctly decreases multiplicity or removes child."""
        self.node.remove_child_once(4)
        self.assertEqual(self.node.children, {4: 1, 5: 1})  # 4 should have multiplicity 1

        self.node.remove_child_once(4)
        self.assertNotIn(4, self.node.children)  # 4 should be fully removed

    def test_remove_parent_id(self):
        """Test that remove_parent_id removes all occurrences of a parent."""
        self.node.remove_parent_id(2)
        self.assertEqual(self.node.parents, {3: 1})  # 2 should be fully removed

    def test_remove_child_id(self):
        """Test that remove_child_id removes all occurrences of a child."""
        self.node.remove_child_id(4)
        self.assertEqual(self.node.children, {5: 1})  # 4 should be fully removed

# EXO 2 : 
class TestGraphEdgeRemoval(unittest.TestCase):

    def setUp(self):
        """Initial setup before each test case."""
        n0 = node(0, "A", {}, {1: 1, 2: 1})
        n1 = node(1, "B", {0: 1}, {2: 2})
        n2 = node(2, "C", {0: 1, 1: 2}, {})
        self.graph = open_digraph([0], [2], [n0, n1, n2])

    def test_remove_edge(self):
        """Test that remove_edge removes one occurrence of an edge."""
        self.graph.remove_edge(1, 2)
        self.assertEqual(self.graph.get_node_by_id(1).get_children(), {2: 1})  # one edge left
        self.graph.remove_edge(1, 2)
        self.assertNotIn(2, self.graph.get_node_by_id(1).get_children())  # no edge left

    def test_remove_parallel_edges(self):
        """Test that remove_parallel_edges removes all occurrences of an edge."""
        self.graph.remove_parallel_edges(1, 2)
        self.assertNotIn(2, self.graph.get_node_by_id(1).get_children())  # 1 -> 2 should be gone

    def test_remove_node_by_id(self):
        """Test that remove_node_by_id removes a node and all its edges."""
        self.graph.remove_node_by_id(1)
        self.assertNotIn(1, self.graph.nodes)  # node 1 should be removed
        self.assertNotIn(1, self.graph.get_node_by_id(0).get_children())  # 0 -> 1 should be gone
        self.assertNotIn(1, self.graph.get_node_by_id(2).get_parents())  # 2 <- 1 should be gone

    def test_remove_edges(self):
        """Test that remove_edges removes multiple individual edges."""
        self.graph.remove_edges([(0, 1), (1, 2)])
        self.assertNotIn(1, self.graph.get_node_by_id(0).get_children())  # 0 -> 1 should be gone
        self.assertEqual(self.graph.get_node_by_id(1).get_children()[2], 1)  # 1 -> 2 should have one left

    def test_remove_several_parallel_edges(self):
        """Test that remove_several_parallel_edges removes multiplse edges completely."""
        self.graph.remove_several_parallel_edges([(0, 1), (1, 2)])
        self.assertNotIn(1, self.graph.get_node_by_id(0).get_children())  # 0 -> 1 gone
        self.assertNotIn(2, self.graph.get_node_by_id(1).get_children())  # 1 -> 2 gone

# EXO 4 : 
class TestAddInputOutputNodes(unittest.TestCase):

    def setUp(self):
        """Initial setup before each test case."""
        self.graph = open_digraph([], [], [])
        self.graph.add_node("A")  # id = 0
        self.graph.add_node("B")  # id = 1
        self.graph.add_node("C")  # id = 2

    def test_add_output_node(self):
        """Test that add_output_node correctly creates an output node."""
        self.graph.add_output_node(1)  # create an output node pointing to node 1
        last_id = max(self.graph.nodes.keys())  # new node id

        self.assertIn(last_id, self.graph.get_output_ids())  # should be in outputs
        self.assertEqual(self.graph.get_node_by_id(last_id).get_parents(), {1: 1})  # should point to 1

    def test_add_input_node(self):
        """Test that add_input_node correctly creates an input node."""
        self.graph.add_input_node(2)  # create an input node pointing to node 2
        last_id = max(self.graph.nodes.keys())  # new node id

        self.assertIn(last_id, self.graph.get_input_ids())  # should be in inputs
        self.assertEqual(self.graph.get_node_by_id(last_id).get_children(), {2: 1})  # should point to 2

    def test_add_output_node_invalid(self):
        """Test that add_output_node raises ValueError if id is invalid."""
        with self.assertRaises(ValueError):
            self.graph.add_output_node(10)  # id 10 does not exist

    def test_add_input_node_invalid(self):
        """Test that add_input_node raises ValueError if id is invalid."""
        with self.assertRaises(ValueError):
            self.graph.add_input_node(10)  # id 10 does not exist

# EXOS 3 ET 5
class IsWellFormedTest(unittest.TestCase):
    """
    Tests for the is_well_formed method.
    Ensures that:
      - is_well_formed accepts valid graphs and rejects invalid ones.
      - Adding or removing a node keeps a well-formed graph.
      - Adding or removing an edge keeps a well-formed graph (if it doesn't 
        break input/output node constraints).
      - Adding an input or output node remains well-formed.
    """

    def test_accepts_valid_graph(self):
        """A properly connected, labeled graph with valid inputs/outputs should be accepted."""
        # Example: a -> b
        n0 = node(0, 'a', {}, {1:1})
        n1 = node(1, 'b', {0:1}, {})
        g = open_digraph([0], [1], [n0, n1])
        self.assertTrue(g.is_well_formed(), "A simple valid graph should be well-formed.")

    def test_rejects_invalid_graph(self):
        """
        Create a graph that is obviously malformed (e.g., an output node isn't 
        properly connected, or references a non-existent node).
        """
        n0 = node(0, 'X', {}, {})
        n1 = node(1, 'Y', {}, {})
        g = open_digraph([0], [1], [n0, n1])  # No edge between input and output
        self.assertFalse(g.is_well_formed(), "Graph with unconnected output node should be malformed.")

    def test_add_remove_node_keeps_well_formed(self):
        """
        Ensure that adding/removing a node maintains a well-formed graph.
        """
        # Start with a valid graph
        n0 = node(0, 'A', {}, {1: 1})
        n1 = node(1, 'B', {0: 1}, {})
        g = open_digraph([0], [1], [n0, n1])
        self.assertTrue(g.is_well_formed())

        # Add a new node safely
        new_id = g.add_node("C")
        self.assertTrue(g.is_well_formed(), "Adding a node without edges should keep the graph well-formed.")

        # Now remove that same node
        g.remove_node_by_id(new_id)
        self.assertTrue(g.is_well_formed(), "Removing a node without breaking constraints should be fine.")


    def test_add_input_output_nodes_keeps_well_formed(self):
        """
        Ensure that adding input/output nodes correctly maintains a well-formed graph.
        """
        g = open_digraph([], [], [])

        # Add a simple node
        a_id = g.add_node("A")
        self.assertTrue(g.is_well_formed())

        # Add an output node that points to 'A'
        g.add_output_node(a_id)
        self.assertTrue(g.is_well_formed(), "Adding an output node pointing to an existing node should stay well-formed.")

        # Add an input node that 'A' points to
        g.add_input_node(a_id)
        self.assertTrue(g.is_well_formed(), "Adding an input node from an existing node should stay well-formed.")




class TestGraph(unittest.TestCase):
    def test_random_int_list(self):
        lst = random_int_list(5, 10)
        self.assertEqual(len(lst), 5)
        self.assertTrue(all(0 <= x < 10 for x in lst))

    def test_random_int_matrix(self):
        n, bound = 4, 10
        matrix = random_int_matrix(n, bound)
        self.assertEqual(len(matrix), n)
        self.assertTrue(all(len(row) == n for row in matrix))

    def test_random_symetric_int_matrix(self):
        n, bound = 4, 10
        matrix = random_symetric_int_matrix(n, bound)
        for i in range(n):
            for j in range(n):
                self.assertEqual(matrix[i][j], matrix[j][i])
        
class TestIsCyclic(unittest.TestCase):

    def test_acyclic_graph(self):
        # Créer un graphe acyclique simple : a -> b -> c
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertFalse(g.is_cyclic(), "Le graphe ne devrait pas être cyclique.")

    def test_cyclic_graph(self):
        # Créer un graphe cyclique : a -> b -> c -> a
        n0 = node(0, 'a', {2: 1}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {0: 1})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertTrue(g.is_cyclic(), "Le graphe devrait être cyclique.")

    def test_empty_graph(self):
        # Graphe vide
        g = open_digraph([], [], [])
        self.assertFalse(g.is_cyclic(), "Un graphe vide ne devrait pas être cyclique.")


class TestIsWellFormed(unittest.TestCase):

    def test_well_formed_graph(self):
        # Créer un graphe bien formé : a -> b -> c
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertTrue(g.is_well_formed(), "Le graphe devrait être bien formé.")

    def test_malformed_graph(self):
        # Créer un graphe mal formé : a -> b, mais b n'a pas de parent
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {}, {})
        g = open_digraph([0], [1], [n0, n1])

        self.assertFalse(g.is_well_formed(), "Le graphe ne devrait pas être bien formé.")

    def test_empty_graph(self):
        # Graphe vide
        g = open_digraph([], [], [])
        self.assertTrue(g.is_well_formed(), "Un graphe vide devrait être bien formé.")

    def test_bool_circ_invalid_initialization(self):
    # Graphe invalide : nœud de copie avec degré entrant != 1
        n0 = node(0, ' ', {}, {1: 1})  # Nœud de copie sans parent
        n1 = node(1, 'a', {0: 1}, {})
        g = open_digraph([], [], [n0, n1])
        with self.assertRaises(ValueError):
            bool_circ(g)
            
class TestShiftIndices(unittest.TestCase):

    def test_shift_indices(self):
        # Créer un graphe simple : a -> b -> c
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        # Décale les indices de 5
        g.shift_indices(5)

        # Vérifie que les indices ont été décalés
        self.assertEqual(g.get_input_ids(), [5])
        self.assertEqual(g.get_output_ids(), [7])
        self.assertEqual(g.get_nodes_id(), [5, 6, 7])

        # Vérifie que les relations entre les nœuds sont conservées
        self.assertEqual(g.get_node_by_id(5).get_children(), {6: 1})
        self.assertEqual(g.get_node_by_id(6).get_parents(), {5: 1})
        self.assertEqual(g.get_node_by_id(6).get_children(), {7: 1})
        self.assertEqual(g.get_node_by_id(7).get_parents(), {6: 1})


if __name__ == '__main__': # the following code is called only when
    unittest.main()        # precisely this file is rung