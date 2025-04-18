import random 


def random_int_list(n, bound):
    '''
    return une liste de taille n contenant des entiers aléatoires entre 0 et bound
    '''
    return [random.randrange(0, bound) for _ in range(n)]

def random_int_matrix(n, bound, null_diag=True, number_generator=(lambda : random.random())):
    '''
    return une matrice n x n avec des entiers aléatoires entre 0 et bound
    '''
    res = []
    for i in range(n):
        line = []
        for j in range(n):
            line.append(int(bound*number_generator()))
        res.append(line)
        if null_diag:
            res[i][i] = 0
    return res

def random_symetric_int_matrix(n, bound, null_diag = True):
    '''
    return une matrice symétrique n x n avec des entiers aléatoires entre 0 et bound, la diaginale est mise à zéro
    '''
    res = random_int_matrix(n, bound, null_diag)
    for i in range(n):
        for j in range(i + 1, n):
            res[j][i] = res[i][j]      
    return res

def random_oriented_int_matrix(n ,bound, null_diag=True):
    '''
    return une matrice orientée n x n avec des entiers aléatoires entre 0 et bound
    '''
    res = random_int_matrix(n,bound,null_diag)
    for i in range(n):
        for j in range(i + 1,n):
            if random.randrange(0, 2) == 1: 
                res[i][j] = 0
            else:
                res[j][i] = 0
    return res

def random_triangular_int_matrix(n ,bound, null_diag = True):
    '''
    return une matrice triangulaire supérieure n x n avec des entiers aléatoires entre 0
    '''
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            res[i][j] = random.randrange(0, bound+1)
    return res 