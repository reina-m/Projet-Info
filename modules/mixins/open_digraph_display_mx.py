import os
import tempfile
import webbrowser

class OpenDigraphDisplayMixin:
    """mixin containing display and conversion methods"""

    def save_as_dot_file(self, path, verbose=False, layout_hints=True):
        """saves graph in DOT format with improved layout"""
        with open(path, "w") as f:
            f.write("digraph {\n")
            if layout_hints:
                f.write("  // Layout settings\n")
                f.write("  rankdir=TB;\n")  
                f.write("  splines=ortho;\n")  
                f.write("  nodesep=0.5;\n")
                f.write("  ranksep=0.5;\n\n")
            
           
            f.write("  // Input nodes\n")
            f.write("  subgraph cluster_inputs {\n")
            f.write("    label=\"Inputs\";\n")
            f.write("    style=rounded;\n")
            f.write("    color=blue;\n")
            for i, nid in enumerate(self.inputs):
                label = f"{'A' if i < len(self.inputs)//2 else 'B'}[{i%(len(self.inputs)//2)}]"
                if verbose:
                    label += f"\n{nid}"
                f.write(f"    {nid} [label=\"{label}\"];\n")
            f.write("  }\n\n")

           
            f.write("  // Logic gates\n")
            for node in self.get_nodes():
                if node.id not in self.inputs and node.id not in self.outputs:
                    label = node.get_label()
                    if verbose:
                        label += f"\n{node.id}"
                    shape = "circle" if label in ["&", "|", "^"] else "box"
                    f.write(f"  {node.id} [label=\"{label}\",shape={shape}];\n")

            f.write("\n  // Output nodes\n")
            f.write("  subgraph cluster_outputs {\n")
            f.write("    label=\"Outputs\";\n")
            f.write("    style=rounded;\n")
            f.write("    color=red;\n")
            for i, nid in enumerate(self.outputs):
                label = f"{'Cout' if i==0 else f'S[{len(self.outputs)-i-1}]'}"
                if verbose:
                    label += f"\n{nid}"
                f.write(f"    {nid} [label=\"{label}\"];\n")
            f.write("  }\n\n")

            f.write("  // Edges\n")
            for node in self.get_nodes():
                for child, mult in node.get_children().items():
                    f.write(f"  {node.id} -> {child}")
                    if mult > 1:
                        f.write(f" [label=\"{mult}\"]")
                    f.write(";\n")
            
            f.write("}\n")

    def display(self, verbose=False, filename_prefix="my_graph"):
        """
        visualizes graph using graphviz
        
        args:
            verbose (bool): if true, includes node ids in labels
            filename_prefix (str): prefix for temporary files
        """
        dot_file = tempfile.NamedTemporaryFile(delete=False, 
                                             prefix=f"{filename_prefix}_", 
                                             suffix=".dot")
        png_file = dot_file.name[:-4] + ".png"
        dot_file.close()
        self.save_as_dot_file(dot_file.name, verbose)
        os.system(f'dot -Tpng {dot_file.name} -o {png_file}')
        webbrowser.open(f'file://{png_file}')
        os.unlink(dot_file.name)