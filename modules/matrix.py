import random 


def random_int_list(n, bound):
    '''
    return a list of length n with random integers between 0 and bound (excluded)
    '''
    return [random.randrange(0, bound) for _ in range(n)]

def random_int_matrix(n, bound, null_diag=True, number_generator=(lambda : random.random())):
    '''
    return an n × n matrix with random integers between 0 and bound

    null_diag: if True, sets the diagonal to 0
    number_generator: function returning float in [0,1); defaults to random.random.-
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
    return an n × n symmetric matrix with random integers between 0 and bound
    the diagonal is set to 0 if null_diag is True
    '''
    res = random_int_matrix(n, bound, null_diag)
    for i in range(n):
        for j in range(i + 1, n):
            res[j][i] = res[i][j]      
    return res

def random_oriented_int_matrix(n ,bound, null_diag=True):
    '''
    return an n × n oriented matrix with random integers between 0 and bound
    the matrix is made asymmetric by zeroing one direction for each (i, j)
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
    return an n × n upper triangular matrix with random integers between 0 and bound
    the diagonal is set to 0 if null_diag is True
    '''
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            res[i][j] = random.randrange(0, bound+1)
    return res 