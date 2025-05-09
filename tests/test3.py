from modules.bool_circ import bool_circ
from modules.open_digraph import open_digraph

def analyze_half_adder_complexity(max_n=8):
    print("Analyse de complexité du Half-Adder:")
    print("\nn\tProfondeur\tPortes\tChemin le plus court")
    print("-" * 45)
    
    results = []
    for n in [2, 4, 6, 8]:
        ha = bool_circ.half_adder_n(n)
        
        depth = ha.graph_depth()
        gates = sum(1 for node in ha.get_nodes() 
                   if node.get_label() in ['^', '&', '|'])
        
        min_path = float('inf')
        for inp in ha.get_input_ids():
            for out in ha.get_output_ids():
                distances, _ = ha.dijkstra(inp, out)
                if out in distances:
                    min_path = min(min_path, distances[out])
        
        results.append((n, depth, gates, min_path))
        print(f"{n}\t{depth}\t\t{gates}\t\t{min_path}")
    
    return results

if __name__ == "__main__":
    results = analyze_half_adder_complexity()