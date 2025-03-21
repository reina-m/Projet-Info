from modules.open_digraph import *

# Create an empty open_digraph
g = open_digraph([], [], [])

# Add nodes manually
n0 = g.add_node(label="A")  # Component 1
n1 = g.add_node(label="B")
n2 = g.add_node(label="C")
n3 = g.add_node(label="X")  # Component 2
n4 = g.add_node(label="Y")

# Add edges to form two separate components
g.add_edge(n0, n1)
g.add_edge(n1, n2)

g.add_edge(n3, n4)

# Display the original graph
print("\nOriginal Graph:")
print(g)
g.display(verbose=True, filename_prefix="original_graph")

# Test connected_components
num_components, component_mapping = g.connected_components()
print(f"\nNumber of Connected Components: {num_components}")
print("Component Mapping (Node ID -> Component ID):")
print(component_mapping)

# Test connected_components_subgraphs
subgraphs = g.connected_components_list()
print("\nSubgraphs corresponding to connected components:")

for i, subgraph in enumerate(subgraphs):
    print(f"\nSubgraph {i}:")
    print(subgraph)
    subgraph.display(verbose=True, filename_prefix=f"subgraph_{i}")
