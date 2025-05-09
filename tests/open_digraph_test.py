import sys
import os
import random

root = os.path.normpath(os.path.join(__file__, './../..'))
sys.path.append(root) 
import unittest
from modules.node import node
from modules.open_digraph import open_digraph
from modules.bool_circ import bool_circ
from modules.matrix import *


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
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertEqual(g.inputs, [0])
        self.assertEqual(g.outputs, [2])
        self.assertEqual(len(g.nodes), 3)

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
        self.assertEqual(g.inputs, [])
        self.assertEqual(g.outputs, [])
        self.assertEqual(g.nodes, {})

class CopyTest(unittest.TestCase):
    def test_copy_node(self):
        n = node(1, 'test', {0:1}, {2:1})
        n_copy = n.copy()
        self.assertEqual(n.id, n_copy.id)
        self.assertEqual(n.label, n_copy.label)
        self.assertEqual(n.parents, n_copy.parents)
        self.assertEqual(n.children, n_copy.children)
        self.assertIsNot(n, n_copy)

    def test_copy_graph(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])
        g_copy = g.copy()
        self.assertEqual(g.inputs, g_copy.inputs)
        self.assertEqual(g.outputs, g_copy.outputs)
        self.assertEqual(len(g.nodes), len(g_copy.nodes))
        self.assertIsNot(g, g_copy)

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
        self.assertEqual(g.new_id(), 1) 
        g.nodes = {1: node(1, 'A', {}, {}), 2: node(2, 'B', {}, {})}
        self.assertEqual(g.new_id(), 3)

class AddNodeTest(unittest.TestCase):
    def test_add_node(self):
        g = open_digraph.empty()
        new_id = g.add_node(label="test", parents={}, children={})

        self.assertIn(new_id, g.nodes)
        self.assertEqual(g.nodes[new_id].get_label(), "test")

    def test_add_node_with_parents_children(self):
        g = open_digraph.empty()
        g.add_node("A")
        g.add_node("B")
        new_id = g.add_node("C", parents={1:1}, children={2:1})
        self.assertIn(new_id, g.nodes) 
        self.assertIn(new_id, g.nodes[1].get_children()) 
        self.assertIn(1, g.nodes[new_id].get_parents()) 

class TestNodeMethods(unittest.TestCase):

    def setUp(self):
        """Initial setup before each test case."""
        self.node = node(1, "A", {2: 2, 3: 1}, {4: 2, 5: 1}) 

    def test_remove_parent_once(self):
        """Test that remove_parent_once correctly decreases multiplicity or removes parent."""
        self.node.remove_parent_once(2)
        self.assertEqual(self.node.parents, {2: 1, 3: 1}) 

        self.node.remove_parent_once(2)
        self.assertEqual(self.node.parents, {3: 1})

        self.node.remove_parent_once(3)
        self.assertEqual(self.node.parents, {})

    def test_remove_child_once(self):
        """Test that remove_child_once correctly decreases multiplicity or removes child."""
        self.node.remove_child_once(4)
        self.assertEqual(self.node.children, {4: 1, 5: 1})

        self.node.remove_child_once(4)
        self.assertNotIn(4, self.node.children) 

    def test_remove_parent_id(self):
        """Test that remove_parent_id removes all occurrences of a parent."""
        self.node.remove_parent_id(2)
        self.assertEqual(self.node.parents, {3: 1}) 

    def test_remove_child_id(self):
        """Test that remove_child_id removes all occurrences of a child."""
        self.node.remove_child_id(4)
        self.assertEqual(self.node.children, {5: 1})

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
        self.assertEqual(self.graph.get_node_by_id(1).get_children(), {2: 1}) 
        self.graph.remove_edge(1, 2)
        self.assertNotIn(2, self.graph.get_node_by_id(1).get_children()) 

    def test_remove_parallel_edges(self):
        """Test that remove_parallel_edges removes all occurrences of an edge."""
        self.graph.remove_parallel_edges(1, 2)
        self.assertNotIn(2, self.graph.get_node_by_id(1).get_children()) 

    def test_remove_node_by_id(self):
        """Test that remove_node_by_id removes a node and all its edges."""
        self.graph.remove_node_by_id(1)
        self.assertNotIn(1, self.graph.nodes)
        self.assertNotIn(1, self.graph.get_node_by_id(0).get_children()) 
        self.assertNotIn(1, self.graph.get_node_by_id(2).get_parents()) 

    def test_remove_edges(self):
        """Test that remove_edges removes multiple individual edges."""
        self.graph.remove_edges([(0, 1), (1, 2)])
        self.assertNotIn(1, self.graph.get_node_by_id(0).get_children()) 
        self.assertEqual(self.graph.get_node_by_id(1).get_children()[2], 1) 

    def test_remove_several_parallel_edges(self):
        """Test that remove_several_parallel_edges removes multiplse edges completely."""
        self.graph.remove_several_parallel_edges([(0, 1), (1, 2)])
        self.assertNotIn(1, self.graph.get_node_by_id(0).get_children())
        self.assertNotIn(2, self.graph.get_node_by_id(1).get_children()) 
 
class TestAddInputOutputNodes(unittest.TestCase):

    def setUp(self):
        """Initial setup before each test case."""
        self.graph = open_digraph([], [], [])
        self.graph.add_node("A")
        self.graph.add_node("B") 
        self.graph.add_node("C")

    def test_add_output_node(self):
        """Test that add_output_node correctly creates an output node."""
        self.graph.add_output_node(1)
        last_id = max(self.graph.nodes.keys())  

        self.assertIn(last_id, self.graph.get_output_ids())
        self.assertEqual(self.graph.get_node_by_id(last_id).get_parents(), {1: 1})

    def test_add_input_node(self):
        """Test that add_input_node correctly creates an input node."""
        self.graph.add_input_node(2)
        last_id = max(self.graph.nodes.keys())

        self.assertIn(last_id, self.graph.get_input_ids())
        self.assertEqual(self.graph.get_node_by_id(last_id).get_children(), {2: 1}) 

    def test_add_output_node_invalid(self):
        """Test that add_output_node raises ValueError if id is invalid."""
        with self.assertRaises(ValueError):
            self.graph.add_output_node(10)

    def test_add_input_node_invalid(self):
        """Test that add_input_node raises ValueError if id is invalid."""
        with self.assertRaises(ValueError):
            self.graph.add_input_node(10)

class IsWellFormedTest(unittest.TestCase):

    def test_accepts_valid_graph(self):
        n0 = node(0, 'a', {}, {1:1})
        n1 = node(1, 'b', {0:1}, {})
        g = open_digraph([0], [1], [n0, n1])
        self.assertTrue(g.is_well_formed(), "A simple valid graph should be well-formed.")

    def test_rejects_invalid_graph(self):
        n0 = node(0, 'X', {}, {})
        n1 = node(1, 'Y', {}, {})
        g = open_digraph([0], [1], [n0, n1])
        self.assertFalse(g.is_well_formed(), "Graph with unconnected output node should be malformed.")

    def test_add_remove_node_keeps_well_formed(self):
        n0 = node(0, 'A', {}, {1: 1})
        n1 = node(1, 'B', {0: 1}, {})
        g = open_digraph([0], [1], [n0, n1])
        self.assertTrue(g.is_well_formed())

        new_id = g.add_node("C")
        self.assertTrue(g.is_well_formed(), "Adding a node without edges should keep the graph well-formed.")

        g.remove_node_by_id(new_id)
        self.assertTrue(g.is_well_formed(), "Removing a node without breaking constraints should be fine.")


    def test_add_input_output_nodes_keeps_well_formed(self):
        g = open_digraph([], [], [])

        a_id = g.add_node("A")
        self.assertTrue(g.is_well_formed())

        g.add_output_node(a_id)
        self.assertTrue(g.is_well_formed(), "Adding an output node pointing to an existing node should stay well-formed.")

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




    
class Test_random(unittest.TestCase):
    def test_random_graph_loop_free(self):
        g = open_digraph.random_graph(n=5, bound=3, form="loop-free")
        mat = g.adjacency_matrix()
        for i in range(len(mat)):
            self.assertEqual(mat[i][i], 0)

    def test_random_graph_undirected(self):
        g = open_digraph.random_graph(n=5, bound=3, form="undirected")
        mat = g.adjacency_matrix()
        for i in range(len(mat)):
            for j in range(len(mat)):
                self.assertEqual(mat[i][j], mat[j][i])

    def test_random_graph_DAG(self):
        g = open_digraph.random_graph(n=5, bound=2, form="DAG")
        mat = g.adjacency_matrix()
        for i in range(len(mat)):
            for j in range(i):
                self.assertEqual(mat[i][j], 0)

    def test_random_graph_oriented(self):
        g = open_digraph.random_graph(n=5, bound=1, form="oriented")
        mat = g.adjacency_matrix()
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i][j] > 0:
                    self.assertEqual(mat[j][i], 0)

class TestDijkstra(unittest.TestCase):
    def setUp(self):
        n0 = node(0, 'a', {}, {1: 1, 2: 1})
        n1 = node(1, 'b', {0: 1}, {3: 1})
        n2 = node(2, 'c', {0: 1}, {3: 1})
        n3 = node(3, 'd', {1: 1, 2: 1}, {})
        self.graph = open_digraph([0], [3], [n0, n1, n2, n3])

    def test_dijkstra_distances(self):
        dist, prev = self.graph.dijkstra(0, direction=1) 
        expected_dist = {0: 0, 1: 1, 2: 1, 3: 2}
        self.assertEqual(dist, expected_dist)

    def test_dijkstra_prev(self):
        dist, prev = self.graph.dijkstra(0, direction=1)
        self.assertIn(prev[1], [0])
        self.assertIn(prev[2], [0]) 
        self.assertIn(prev[3], [1, 2]) 

    def test_dijkstra_target_stop(self):
        dist, prev = self.graph.dijkstra(0, tgt=3, direction=1)
        self.assertEqual(dist[3], 2)

class TestIsCyclic(unittest.TestCase):

    def test_acyclic_graph(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertFalse(g.is_cyclic(), "Le graphe ne devrait pas être cyclique.")

    def test_cyclic_graph(self):
        n0 = node(0, 'a', {2: 1}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {0: 1})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertTrue(g.is_cyclic(), "Le graphe devrait être cyclique.")

    def test_empty_graph(self):
        g = open_digraph([], [], [])
        self.assertFalse(g.is_cyclic(), "Un graphe vide ne devrait pas être cyclique.")

class TestIsWellFormed(unittest.TestCase):

    def test_well_formed_graph(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        self.assertTrue(g.is_well_formed(), "Le graphe devrait être bien formé.")

    def test_malformed_graph(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {}, {})
        g = open_digraph([0], [1], [n0, n1])

        self.assertFalse(g.is_well_formed(), "Le graphe ne devrait pas être bien formé.")

    def test_empty_graph(self):
        g = open_digraph([], [], [])
        self.assertTrue(g.is_well_formed(), "Un graphe vide devrait être bien formé.")
            
class TestShiftIndices(unittest.TestCase):

    def test_shift_indices(self):
        n0 = node(0, 'a', {}, {1: 1})
        n1 = node(1, 'b', {0: 1}, {2: 1})
        n2 = node(2, 'c', {1: 1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])

        g.shift_indices(5)

        self.assertEqual(g.get_input_ids(), [5])
        self.assertEqual(g.get_output_ids(), [7])
        self.assertEqual(g.get_nodes_id(), [5, 6, 7])

        self.assertEqual(g.get_node_by_id(5).get_children(), {6: 1})
        self.assertEqual(g.get_node_by_id(6).get_parents(), {5: 1})
        self.assertEqual(g.get_node_by_id(6).get_children(), {7: 1})
        self.assertEqual(g.get_node_by_id(7).get_parents(), {6: 1})

class TestOpenDigraphComposition(unittest.TestCase):

    def setUp(self):
        self.g1 = open_digraph.empty()
        n1 = self.g1.add_node(label="A")
        n2 = self.g1.add_node(label="B")
        self.g1.add_edge(n1, n2)
        self.g1.add_input_id(n1)
        self.g1.add_output_id(n2)

        self.g2 = open_digraph.empty()
        n3 = self.g2.add_node(label="C")
        n4 = self.g2.add_node(label="D")
        self.g2.add_edge(n3, n4)
        self.g2.add_input_id(n3)
        self.g2.add_output_id(n4)

    

    def test_identity(self):
        n = 3
        id_graph = open_digraph.identity(n)
        self.assertEqual(len(id_graph.get_input_ids()), n)
        self.assertEqual(len(id_graph.get_output_ids()), n)
        self.assertEqual(len(id_graph.nodes), 2 * n)
        for input_id in id_graph.get_input_ids():
            node = id_graph.get_node_by_id(input_id)
            children = node.get_children()
            self.assertEqual(len(children), 1)
            output_id = list(children.keys())[0]
            self.assertIn(output_id, id_graph.get_output_ids())
    def test_parallel_basic(self):
        g_parallel = open_digraph.parallel(self.g1, self.g2)
        self.assertEqual(len(g_parallel.nodes), len(self.g1.nodes) + len(self.g2.nodes))
        self.assertEqual(len(g_parallel.get_input_ids()), len(self.g1.get_input_ids()) + len(self.g2.get_input_ids()))
        self.assertEqual(len(g_parallel.get_output_ids()), len(self.g1.get_output_ids()) + len(self.g2.get_output_ids()))
 
        self.assertEqual(len(self.g1.nodes), 2)
        self.assertEqual(len(self.g2.nodes), 2)

    def test_icompose_valid(self):
        id_graph = open_digraph.identity(len(self.g1.get_input_ids()))    
        original_node_count = len(self.g1.nodes)
        self.g1.icompose(id_graph)
        self.assertEqual(len(self.g1.nodes), original_node_count + len(id_graph.nodes))
        self.assertEqual(len(self.g1.get_input_ids()), len(id_graph.get_input_ids()))
        
    
    def test_icompose_invalid(self):
        bad_graph = open_digraph.empty()
        bad_graph.add_input_id(bad_graph.add_node())
        bad_graph.add_output_id(bad_graph.add_node())
        bad_graph.add_output_id(bad_graph.add_node())  
        
        with self.assertRaises(ValueError):
            self.g1.icompose(bad_graph)

    def test_compose_valid(self):
        id_graph = open_digraph.identity(len(self.g1.get_input_ids()))
        composed = open_digraph.compose(self.g1, id_graph)
        self.assertEqual(len(composed.nodes), len(self.g1.nodes) + len(id_graph.nodes))
        self.assertEqual(len(self.g1.nodes), 2)
        self.assertEqual(len(id_graph.nodes), 2 * len(self.g1.get_input_ids()))

class TopologicalSortTest(unittest.TestCase):
    def test_topological_sort_acyclic(self):
        n0 = node(0, 'a', {}, {1:1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1:1}, {})
        g = open_digraph([0], [2], [n0, n1, n2])
        
        layers = g.topological_sort()
        self.assertEqual(layers, [[0], [1], [2]])
        
    def test_topological_sort_multiple_paths(self):
        n0 = node(0, 'a', {}, {1:1, 2:1})
        n1 = node(1, 'b', {0:1}, {3:1})
        n2 = node(2, 'c', {0:1}, {3:1})
        n3 = node(3, 'd', {1:1, 2:1}, {})
        g = open_digraph([0], [3], [n0, n1, n2, n3])
        
        layers = g.topological_sort()
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0], [0])
        self.assertEqual(set(layers[1]), {1, 2})
        self.assertEqual(layers[2], [3])
        
    def test_topological_sort_empty_graph(self):
        g = open_digraph.empty()
        layers = g.topological_sort()
        self.assertEqual(layers, [])

    def test_topological_sort_cyclic_graph(self):
        n0 = node(0, 'a', {2:1}, {1:1})
        n1 = node(1, 'b', {0:1}, {2:1})
        n2 = node(2, 'c', {1:1}, {0:1})
        g = open_digraph([0], [2], [n0, n1, n2])
        
        with self.assertRaises(ValueError):
            g.topological_sort()
            
    def test_topological_sort_complex_acyclic(self):
        n0 = node(0, 'a', {}, {1:1, 2:1})
        n1 = node(1, 'b', {0:1}, {3:1})
        n2 = node(2, 'c', {0:1}, {4:1})
        n3 = node(3, 'd', {1:1}, {5:1})
        n4 = node(4, 'e', {2:1}, {5:1})
        n5 = node(5, 'f', {3:1, 4:1}, {})
        g = open_digraph([0], [5], [n0, n1, n2, n3, n4, n5])
        
        layers = g.topological_sort()
        self.assertEqual(len(layers), 4)
        self.assertEqual(layers[0], [0])
        self.assertEqual(set(layers[1]), {1, 2})
        self.assertEqual(set(layers[2]), {3, 4})
        self.assertEqual(layers[3], [5])
      
class TestFusion(unittest.TestCase):
    def setUp(self):
        self.n0 = node(0, 'a', {}, {1:1})
        self.n1 = node(1, 'b', {0:1, 3:1}, {2:1})
        self.n2 = node(2, 'c', {1:1}, {})
        self.n3 = node(3, 'd', {}, {1:1})
        self.graph = open_digraph([0, 3], [2], [self.n0, self.n1, self.n2, self.n3])

    def test_fusion_basic(self):
        fusion_id = self.graph.fusion(0, 3, new_label="a+d")
        self.assertIn(fusion_id, self.graph.nodes)
        self.assertEqual(self.graph.nodes[fusion_id].label, "a+d")
        self.assertEqual(self.graph.nodes[fusion_id].children, {1:2})
        self.assertEqual(self.graph.nodes[1].parents, {fusion_id:2})
        self.assertNotIn(0, self.graph.nodes)
        self.assertNotIn(3, self.graph.nodes)
        self.assertIn(fusion_id, self.graph.inputs)
        self.assertEqual(len(self.graph.inputs), 1)

    def test_fusion_with_outputs(self):
        fusion_id = self.graph.fusion(1, 2, new_label="b+c")
        self.assertIn(fusion_id, self.graph.outputs)
        self.assertEqual(self.graph.nodes[0].children, {fusion_id:1})
        self.assertEqual(self.graph.nodes[3].children, {fusion_id:1})
        self.assertEqual(self.graph.nodes[fusion_id].parents, {0:1, 3:1})

    def test_fusion_invalid_nodes(self):
        with self.assertRaises(ValueError):
            self.graph.fusion(0, 99)
        with self.assertRaises(ValueError):
            self.graph.fusion(99, 1)
        with self.assertRaises(ValueError):
            self.graph.fusion(0, 0)

    def test_fusion_label_handling(self):
        fusion_id1 = self.graph.fusion(0, 3)
        self.assertEqual(self.graph.nodes[fusion_id1].label, "a")

        self.setUp()
        fusion_id2 = self.graph.fusion(0, 3, new_label="fusion")
        self.assertEqual(self.graph.nodes[fusion_id2].label, "fusion")



    def test_fusion_edge_multiplicities(self):
        self.graph.add_edge(0, 1)
        self.graph.add_edge(3, 1)
        fusion_id = self.graph.fusion(0, 3)
        self.assertEqual(self.graph.nodes[fusion_id].children, {1:4})
        self.assertEqual(self.graph.nodes[1].parents, {fusion_id:4})


class TestCLA(unittest.TestCase):
    def test_cla4(self):
        cla = bool_circ.cla4()
        self.assertEqual(len(cla.inputs), 9)  
        self.assertEqual(len(cla.outputs), 5)  
        self.assertTrue(cla.is_well_formed())

    def test_cla4n(self):
        cla = bool_circ.cla4n(2)
        self.assertEqual(len(cla.inputs), 17)  
        self.assertEqual(len(cla.outputs), 9)  
        self.assertTrue(cla.is_well_formed())

class TestFromInt(unittest.TestCase):
    def test_from_int(self):
        bc = bool_circ.from_int(5, 4)
        self.assertEqual(len(bc.outputs), 4)
        self.assertEqual([bc.get_node_by_id(nid).label for nid in bc.outputs], ['0','1','0','1'])
    


class TestSimplifications(unittest.TestCase):
    def setUp(self):
        self.bc = bool_circ.empty()
        self.output_node = self.bc.add_node('')
        self.bc.add_output_node(self.output_node)
    
    def test_simplify_not(self):
        n = self.bc.add_node('~')
        c = self.bc.add_node('0')
        self.bc.add_edge(n, c)
        self.bc.add_edge(n, self.output_node)
        
        self.assertTrue(self.bc.simplify_not(n))

    def test_simplify_and(self):
        n  = self.bc.add_node('&')
        c1 = self.bc.add_node('0')
        c2 = self.bc.add_node('1')
    
        self.bc.add_edge(n, self.output_node)
        self.bc.add_edge(n, c1)
        self.bc.add_edge(n, c2)
        
        self.assertTrue(self.bc.simplify_and(n))
        self.assertEqual(self.bc.get_node_by_id(self.bc.outputs[0]).label, '0')


class TestEvaluate(unittest.TestCase):
    def test_evaluate_adder(self):
        adder = bool_circ.adder_n(2)
        for i, val in enumerate([0,1,0,1,0]):
            nid = adder.inputs[i]
            adder.get_node_by_id(nid).children = {}
            const = adder.add_node(str(val))
            adder.add_edge(const, nid)
        
        adder.evaluate()
        outputs = [adder.get_node_by_id(nid).label for nid in adder.outputs]
        self.assertEqual(outputs, ['0','1'])


class TestHammings(unittest.TestCase):
    def test_hamming_full_cycle(self):
        message = [1, 0, 1, 0]
        
        encoder = bool_circ.hamming_encoder()
        encoded = encoder.evaluate_with_inputs(message)
        
        error_pos = random.randint(0, 6)
        corrupted = encoded.copy()
        corrupted[error_pos] = 1 - corrupted[error_pos]
        
        decoder = bool_circ.hamming_decoder()
        decoded = decoder.evaluate_with_inputs(corrupted)
        
        self.assertEqual(decoded, message)

    def test_rewrite_rules(self):
        g = bool_circ.empty()
        a = g.add_node(label='')

        not1 = g.add_node(label='~', parents={a:1})
        not2 = g.add_node(label='~', parents={not1:1})
        out = g.add_node()
        g.add_edge(not2, out)
        
        g.simplify_all()
        
        self.assertEqual(len(g.get_nodes()), 2)



if __name__ == '__main__': 
    unittest.main()        