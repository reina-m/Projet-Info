from modules.open_digraph import *
import inspect

# QUESTION 9 DU TD 1
print("methods of node class:")
print(dir(node))
print("\nmethods of open_digraph class:")
print(dir(open_digraph))
print("\nfile location of open_digraph:")
print(inspect.getfile(open_digraph))

print("\nsource code of open_digraph.get_nodes:")
print(inspect.getsource(open_digraph.get_nodes))
print("\ndocstring of node.__init__:")
print(inspect.getdoc(node.__init__))

# graphe simple
g = open_digraph.empty()
n0 = g.add_node(label="A")
n1 = g.add_node(label="B")
n2 = g.add_node(label="C")
g.add_edge(n0, n1)
g.add_edge(n1, n2)

dot_path = "graphe1.dot"
g.save_as_dot_file(dot_path)
print(f"Graph saved as: {dot_path}")
# load the graph back from the .dot file
g_loaded = open_digraph.from_dot_file(dot_path)
print("\nLoaded simple graph from DOT file:")
print(g_loaded)  # should use __str__

# inputs / outputs
g2 = open_digraph.empty()
a = g2.add_node(label="A")
b = g2.add_node(label="B")
c = g2.add_node(label="C")

g2.add_input_id(a)
g2.add_output_id(c)
g2.add_edge(a, b)
g2.add_edge(b, c)

dot_path_io = "graphe2.dot"
g2.save_as_dot_file(dot_path_io)
print(f"\nGraph (with inputs/outputs) saved as: {dot_path_io}")
g2_loaded = open_digraph.from_dot_file(dot_path_io)
print("\nLoaded graph (inputs/outputs) from DOT file:")
print(g2_loaded)

# 3) (multiplicity)
g3 = open_digraph.empty()
a = g3.add_node(label="A")
b = g3.add_node(label="B")

# add 3 parallel edges from M0 to M1
g3.add_edge(a, b, 3)
dot_path3 = "graphe3.dot"
g3.save_as_dot_file(dot_path3)
print(f"\nGraph (multiple edges) saved as: {dot_path3}")
g3_loaded = open_digraph.from_dot_file(dot_path3)
print("\nLoaded graph (multiple edges) from DOT file:")
print(g3_loaded)

# chain
g4 = open_digraph.empty()
ch_nodes = [g4.add_node(label=f"v{i}") for i in range(5)]
# create a chain v0 -> v1 -> v2 -> v3 -> v4
for i in range(len(ch_nodes) - 1):
    g4.add_edge(ch_nodes[i], ch_nodes[i+1])
dot_path4 = "graphe4.dot"
g4.save_as_dot_file(dot_path4)
print(f"\nChain graph saved as: {dot_path4}")
g4_loaded = open_digraph.from_dot_file(dot_path4)
print("\nLoaded chain graph from DOT file:")
print(g4_loaded)

g5 = open_digraph.empty()
ch_nodes = [g5.add_node(label=f"v{i}") for i in range(5)]
# create a chain v0 -> v1 -> v2 -> v3 -> v4
g5.add_edge(ch_nodes[0], ch_nodes[1])
g5.add_edge(ch_nodes[0], ch_nodes[2])
g5.add_edge(ch_nodes[2], ch_nodes[4])
g5.add_edge(ch_nodes[2], ch_nodes[4])
dot_path5 = "graphe5.dot"
g5.save_as_dot_file(dot_path5)
print(f"\nChain graph saved as: {dot_path5}")
g5_loaded = open_digraph.from_dot_file(dot_path5)
print("\nLoaded chain graph from DOT file:")
print(g5_loaded)

print("\nAll graphs have been saved, loaded, and printed.\n")
print("\nDisplaying graph...")
g5.display(verbose=True) 