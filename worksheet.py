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

