import os
import tempfile
import webbrowser

class OpenDigraphDisplayMixin:
    """mixin containing display and conversion methods"""

    def save_as_dot_file(self, path, verbose=False):
        """
        saves graph in DOT format
        
        args:
            path (str): output file path, must end in .dot
            verbose (bool): if true, includes node ids in labels
            
        returns:
            none
            
        raises:
            AssertionError: if path doesn't end in .dot
            
        format details:
            - inputs marked with input="true" 
            - outputs marked with output="true"
            - node labels included if present
        """
        assert path.endswith(".dot"), "path must end with .dot"
        s = "digraph G {\n\n"

        # Write out all nodes
        # Write out all nodes
        #    - If it's an input node, add input="true"
        #    - If it's an output node, add output="true"
        #    - If it has a label, use label="..."
        #    - If verbose, also show \nid=ID
        for node_id, node in self.nodes.items():
            # Gather attributes in a small dict
            attr_dict = {}
            if node.label:
                # e.g. label="A"
                # if verbose we add the ID on a second line
                if verbose:
                    attr_dict["label"] = f'{node.label}\\nid={node_id}'
                else:
                    attr_dict["label"] = node.label
            else:
                # no label
                if verbose:
                    attr_dict["label"] = f'{node_id}'
                else:
                    attr_dict["label"] = ""

            # mark as input or output
            if node_id in self.inputs:
                attr_dict["input"] = "true"
            if node_id in self.outputs:
                attr_dict["output"] = "true"
            
            if node_id in self.inputs:
                attr_dict["shape"] = "diamond"
            elif node_id in self.outputs:
                attr_dict["shape"] = "box"
            else:
                attr_dict["shape"] = "circle"
            
            # build final bracket string: e.g. [label="A", input="true"]
            if attr_dict:
                # turn dict into a list of key="value"
                attributes_str = ", ".join(f'{k}="{v}"' for k,v in attr_dict.items())
                s += f'    v{node_id} [{attributes_str}];\n'
            else:
                # if truly no attributes, just v{node_id}
                s += f'    v{node_id};\n'

        s += "\n"

        #  write edges
        for node_id, node in self.nodes.items():
            for child_id, multiplicity in node.get_children().items():
                if multiplicity <= 1:
                    # single edge
                    s += f'    v{node_id} -> v{child_id};\n'
                else:
                    # multiple edges
                    for _ in range(multiplicity):
                        s += f'    v{node_id} -> v{child_id};\n'

        s += "\n}\n"

        with open(path, "w") as f:
            f.write(s)

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