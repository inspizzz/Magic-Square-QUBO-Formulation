# pyqubo stuff
from pyqubo import Array, Constraint
from mpl_toolkits.mplot3d import Axes3D

import math
import matplotlib.pyplot as plt
import numpy as np
import neal
import re

# get the vector index given the i, j, k magic square matrix identifiers
def get_index(i, j, k, SQUARE_SIZE, BITS):
    
    # we need to know the length of the magic square matrix horizontally
    row_length = BITS * SQUARE_SIZE
    
    return int((i*row_length) + (j*BITS) + k)


def implement_row(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for row1 in range(SQUARE_SIZE):
        for row2 in range(SQUARE_SIZE):
            if row1 == row2:
                continue

            H_row1 = 0
            H_row2 = 0

            for j in range(SQUARE_SIZE):
                H_row1 += Constraint(penalty * np.sum([list(variables)[get_index(row1,j,k,SQUARE_SIZE,BITS)] * np.pow(2, BITS-k) for k in range(BITS)]), label=f"row_{row1}_{j}")

            for j in range(SQUARE_SIZE):
                H_row2 += Constraint(penalty * np.sum([list(variables)[get_index(row2,j,k,SQUARE_SIZE,BITS)] * np.pow(2, BITS-k) for k in range(BITS)]), label=f"row_{row2}_{j}")


            H += (H_row1 - H_row2) ** 2
        
    return H


def implement_col(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for col1 in range(SQUARE_SIZE):
        for col2 in range(SQUARE_SIZE):
            if col1 == col2:
                continue

            H_col1 = 0
            H_col2 = 0

            for i in range(SQUARE_SIZE):
                H_col1 += Constraint(penalty * np.sum([list(variables)[get_index(i,col1,k,SQUARE_SIZE,BITS)] * np.pow(2, BITS-k) for k in range(BITS)]), label=f"row_{i}_{col1}")

            for i in range(SQUARE_SIZE):
                H_col2 += Constraint(penalty * np.sum([list(variables)[get_index(i,col2,k,SQUARE_SIZE,BITS)] * np.pow(2, BITS-k) for k in range(BITS)]), label=f"row_{i}_{col2}")


            H += (H_col1 - H_col2) ** 2
        
    return H


def implement_empty(H, variables, SQUARE_SIZE, BITS, penalty=1):

    def xor(var1, var2):
        return (var1 + var2) - (2 * var1 * var2)

    for i in range(SQUARE_SIZE):
        for j in range(SQUARE_SIZE):
                H += Constraint(penalty * np.prod([1 - xor(var, 0) for var in list(variables)[get_index(i,j,0,SQUARE_SIZE,BITS):get_index(i,j,BITS,SQUARE_SIZE,BITS)]]), label=f"empty_{i}_{j}")

    return H
    
                
def implement_unique(H, variables, SQUARE_SIZE, BITS, penalty=1):

    for i1 in range(SQUARE_SIZE):
        for j1 in range(SQUARE_SIZE):
            for i2 in range(i1, SQUARE_SIZE):
                for j2 in range(SQUARE_SIZE):
                    if i1 == i2 and j1 <= j2:
                        continue

                    # print(f"Comparing {i1},{j1} -> {i2},{j2}")
                    H_tmp = 1

                    for k in range(BITS):
                        var1 = variables[get_index(i1,j1,k,SQUARE_SIZE,BITS)]
                        var2 = variables[get_index(i2,j2,k,SQUARE_SIZE,BITS)]
                        
                        H_tmp *= Constraint(1 - ((var1 + var2) - (2 * var1 * var2)), label=f"unique_{i1}_{j1}_{i2}_{j2}_{k}")

                    H += penalty * H_tmp
    return H
                

NUMBER_OF_SAMPLES = 5000
unique_penalty_weight = 1
empty_penalty_weight = 1
row_penalty_weight = 1
col_penalty_weight = 1


for SQUARE_SIZE in range(2, 4):

    # debug
    print(f"[EXPERIMENT] - starting experiment for square size {SQUARE_SIZE}")

    # calculate the number of bits and vars
    BITS = math.floor(math.log2(SQUARE_SIZE**2) + 1)
    VARS = SQUARE_SIZE * SQUARE_SIZE * BITS

    # create binary variables
    array = Array.create('x', VARS, 'BINARY')

    # apply constraints
    H = 0
    H += implement_empty(H, array, SQUARE_SIZE, BITS, empty_penalty_weight)
    H += implement_unique(H, array, SQUARE_SIZE, BITS, unique_penalty_weight)
    # H += implement_row(H, array, row_penalty_weight)
    # H += implement_col(H, array, col_penalty_weight)

    model = H.compile()
    qubo, offset = model.to_qubo()

    Q = np.zeros((SQUARE_SIZE * SQUARE_SIZE * BITS, SQUARE_SIZE * SQUARE_SIZE * BITS))

    for key, value in qubo.items():
        i = int(re.search(r'\[(\d+)\]', key[0]).group(1))
        j = int(re.search(r'\[(\d+)\]', key[1]).group(1))

        Q[i][j] = value

    plt.imshow(Q, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

    # compile and convert to binary quadratic model
    model = H.compile()
    bqm = model.to_bqm()

    # simulate
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(bqm, num_reads=NUMBER_OF_SAMPLES)
    decoded_samples = model.decode_sampleset(sampleset)

    # decode
    solutions = []

    for solution in decoded_samples:
        
        # get the sample and energy
        sample = solution.sample
        energy = solution.energy

        # print(sample)
        
        sample = np.array([sample[f"x[{i}]"] for i in range(SQUARE_SIZE * SQUARE_SIZE * BITS)]).reshape((SQUARE_SIZE, SQUARE_SIZE, BITS))  

        solutions.append({"sample": sample, "energy": energy})

    # save the solutions
    np.save(f"annealing analysis/binary/solutions_{SQUARE_SIZE}.npy", {"solutions": solutions, "penalty_weights": {"penalty_unique": unique_penalty_weight, "penalty_empty": empty_penalty_weight, "penalty_row": row_penalty_weight, "penalty_col": col_penalty_weight}, "samples": NUMBER_OF_SAMPLES, "time": sampleset.info["timing"]})
    