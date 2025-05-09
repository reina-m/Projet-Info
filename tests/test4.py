from modules.bool_circ import bool_circ

def test_hamming_properties():
    print("Test des propriétés du code de Hamming(7,4)\n")
    
    encoder = bool_circ.hamming_encoder()
    decoder = bool_circ.hamming_decoder()
    
    original = [1,0,1,1]
    print(f"Message original: {original}")
    
    print("\n1. Test sans erreur:")
    encoded = encoder.evaluate_with_inputs(original)
    decoded = decoder.evaluate_with_inputs(encoded)
    print(f"Encodé: {encoded}")
    print(f"Décodé: {decoded}")
    print(f"Identité préservée: {decoded == original}")
    
    print("\n2. Tests avec une erreur:")
    for pos in range(7):
        encoded_error = encoded.copy()
        encoded_error[pos] ^= 1 
        decoded_error = decoder.evaluate_with_inputs(encoded_error)
        print(f"Erreur bit {pos}: {encoded_error} -> {decoded_error} "
              f"(correct: {decoded_error == original})")
    
    print("\n3. Test avec deux erreurs:")
    encoded_2errors = encoded.copy()
    encoded_2errors[0] ^= 1 
    encoded_2errors[3] ^= 1
    decoded_2errors = decoder.evaluate_with_inputs(encoded_2errors)
    print(f"Original:  {original}")
    print(f"2 erreurs: {encoded_2errors}")
    print(f"Décodé:    {decoded_2errors}")
    print(f"Correct:   {decoded_2errors == original}")

if __name__ == "__main__":
    test_hamming_properties()