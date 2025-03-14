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

# 1) SIMPLE GRAPH
g = open_digraph.empty()
n0 = g.add_node(label="A")
n1 = g.add_node(label="B")
n2 = g.add_node(label="C")
g.add_edge(n0, n1)
g.add_edge(n1, n2)

dot_path = "graph_simple.dot"
g.save_as_dot_file(dot_path)
print(f"Graph saved as: {dot_path}")

# load the graph back from the .dot file
g_loaded = open_digraph.from_dot_file(dot_path)
print("\nLoaded simple graph from DOT file:")
print(g_loaded)  # should use __str__

# 2) GRAPH WITH INPUTS & OUTPUTS
g_io = open_digraph.empty()
a = g_io.add_node(label="X")
b = g_io.add_node(label="Y")
c = g_io.add_node(label="Z")

# mark some as input/output
g_io.add_input_id(a)
g_io.add_output_id(c)

# connect the nodes
g_io.add_edge(a, b)
g_io.add_edge(b, c)

dot_path_io = "graph_io.dot"
g_io.save_as_dot_file(dot_path_io)
print(f"\nGraph (with inputs/outputs) saved as: {dot_path_io}")

g_io_loaded = open_digraph.from_dot_file(dot_path_io)
print("\nLoaded graph (inputs/outputs) from DOT file:")
print(g_io_loaded)

# 3) GRAPH WITH MULTIPLE EDGES (multiplicity)
g_mult = open_digraph.empty()
m0 = g_mult.add_node(label="M0")
m1 = g_mult.add_node(label="M1")

# add 3 parallel edges from M0 to M1
g_mult.add_edge(m0, m1, 3)

dot_path_mult = "graph_mult.dot"
g_mult.save_as_dot_file(dot_path_mult)
print(f"\nGraph (multiple edges) saved as: {dot_path_mult}")

g_mult_loaded = open_digraph.from_dot_file(dot_path_mult)
print("\nLoaded graph (multiple edges) from DOT file:")
print(g_mult_loaded)

# 4) CHAIN GRAPH (longer chain of nodes)
g_chain = open_digraph.empty()
ch_nodes = [g_chain.add_node(label=f"N{i}") for i in range(5)]
# create a chain N0 -> N1 -> N2 -> N3 -> N4
for i in range(len(ch_nodes) - 1):
    g_chain.add_edge(ch_nodes[i], ch_nodes[i+1])

dot_path_chain = "graph_chain.dot"
g_chain.save_as_dot_file(dot_path_chain)
print(f"\nChain graph saved as: {dot_path_chain}")

g_chain_loaded = open_digraph.from_dot_file(dot_path_chain)
print("\nLoaded chain graph from DOT file:")
print(g_chain_loaded)

# 5) A MIXED GRAPH (inputs, outputs, multiplicities, chain)
g_mix = open_digraph.empty()

mix0 = g_mix.add_node(label="Mix0")
mix1 = g_mix.add_node(label="Mix1")
mix2 = g_mix.add_node(label="Mix2")
mix3 = g_mix.add_node(label="Mix3")

# mark some as input, output
g_mix.add_input_id(mix0)
g_mix.add_input_id(mix1)
g_mix.add_output_id(mix2)
g_mix.add_output_id(mix3)

# edges with multiplicity
g_mix.add_edge(mix0, mix1, 3)
# single edges
g_mix.add_edge(mix0, mix3)

dot_path_mix = "graph_mix.dot"
g_mix.save_as_dot_file(dot_path_mix)
print(f"\nMixed graph saved as: {dot_path_mix}")

g_mix_loaded = open_digraph.from_dot_file(dot_path_mix)
print("\nLoaded mixed graph from DOT file:")
print(g_mix_loaded)

print("\nAll graphs have been saved, loaded, and printed.\n")
print("\nDisplaying the mixed graph...")
g_mix.display(verbose=True) 