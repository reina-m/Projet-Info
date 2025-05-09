from modules.bool_circ import bool_circ
import os

def visualize_half_adder(n=4):
    os.makedirs("images", exist_ok=True)
    
    ha = bool_circ.half_adder_n(n)
    
    inputs = [0,1,0,1, 0,0,1,1] 
    
    ha.save_as_dot_file("images/adder_structure.dot", verbose=False)
    
    result = ha.evaluate_with_inputs(inputs)
    print(f"Adding {inputs[:n]} ({sum(b<<i for i,b in enumerate(reversed(inputs[:n])))})")
    print(f"   to {inputs[n:]} ({sum(b<<i for i,b in enumerate(reversed(inputs[n:])))})")
    print(f"Result: {result} ({sum(b<<i for i,b in enumerate(reversed(result)))})")

    os.system('dot -Tpng -Gdpi=300 images/adder_structure.dot -o images/adder_structure.png')

if __name__ == "__main__":
    visualize_half_adder(4)