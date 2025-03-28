# pyqubo stuff
from pyqubo import Array, Constraint

import matplotlib.pyplot as plt
import numpy as np
import neal

# get the vector index given the i, j, k magic square matrix identifiers
def get_index(i, j, k, SQUARE_SIZE, BITS):
    
    # we need to know the length of the magic square matrix horizontally
    row_length = BITS * SQUARE_SIZE
    
    return int((i*row_length) + (j*BITS) + k)

def penalty_sum_row(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for row1 in range(SQUARE_SIZE):
        for row2 in range(SQUARE_SIZE):
            if row1 == row2:
                continue
            
            H_tmp = 0
            
            for j in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp += Constraint(variables[get_index(row2,j,k,SQUARE_SIZE,BITS)], label=f"row2_{row2}_{j}_{k}")

            for j in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp -= Constraint(variables[get_index(row1,j,k,SQUARE_SIZE,BITS)], label=f"row1_{row1}_{j}_{k}")

            H_tmp **= 2
            H += H_tmp

    H *= penalty

    return H


def penalty_sum_col(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for col1 in range(SQUARE_SIZE):
        for col2 in range(SQUARE_SIZE):
            if col1 == col2:
                continue
            
            H_tmp = 0
            
            for i in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp += Constraint(variables[get_index(i, col2, k,SQUARE_SIZE,BITS)], label=f"col1_{i}_{col2}_{k}")

            for i in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp -= Constraint(variables[get_index(i, col1, k,SQUARE_SIZE,BITS)], label=f"col2_{i}_{col1}_{k}")

            H_tmp **= 2
            H += H_tmp

    H *= penalty

    return H


def penalty_domain_wall(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for i in range(SQUARE_SIZE):
        for j in range(SQUARE_SIZE):
            for k in range(BITS-1):
                H += penalty * Constraint(variables[get_index(i,j,k+1,SQUARE_SIZE,BITS)] - (variables[get_index(i,j,k+1,SQUARE_SIZE,BITS)] * variables[get_index(i,j,k,SQUARE_SIZE,BITS)]), label=f"domain_wall_{i}_{j}_{k}")
    return H   


def penalty_unique(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for k in range(BITS):
        H += Constraint(penalty * (sum(variables[get_index(i,j,k,SQUARE_SIZE,BITS)] for i in range(SQUARE_SIZE) for j in range(SQUARE_SIZE)) * (1/(BITS-k)) - 1) ** 2, label=f"unique_{k}")

    return H


NUMBER_OF_SAMPLES = 5000
unique_penalty_weight = 1
unary_penalty_weight = 1
row_penalty_weight = 1
col_penalty_weight = 1

global iteration

for SQUARE_SIZE in range(6, 7):

    iteration = 1

    print(f"[EXPERIMENT] - starting experiment for square size {SQUARE_SIZE}")

    BITS = SQUARE_SIZE * SQUARE_SIZE

    array = Array.create('x', SQUARE_SIZE * SQUARE_SIZE * BITS, 'BINARY')

    # apply constraints
    H = 0
    H += penalty_domain_wall(H, array, SQUARE_SIZE, BITS, unary_penalty_weight)
    H += penalty_unique(H, array, SQUARE_SIZE, BITS, unique_penalty_weight)
    H += penalty_sum_col(H, array, SQUARE_SIZE, BITS, col_penalty_weight)
    H += penalty_sum_row(H, array, SQUARE_SIZE, BITS, row_penalty_weight)

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

        # split the sample values into bit sized chunks
        # sample = np.array([list(sample.values())[i:i+BITS] for i in range(0, len(sample), BITS)]).reshape((SQUARE_SIZE, SQUARE_SIZE, BITS))
        sample = np.array([sample[f"x[{i}]"] for i in range(SQUARE_SIZE * SQUARE_SIZE * BITS)]).reshape((SQUARE_SIZE, SQUARE_SIZE, BITS))  

        solutions.append({"sample": sample, "energy": energy})

    np.save(f"annealing analysis/domain_wall/solutions_{SQUARE_SIZE}.npy", {"solutions": solutions, "penalty_weights": {"penalty_unique": unique_penalty_weight, "penalty_unary": unary_penalty_weight, "penalty_row": row_penalty_weight, "penalty_col": col_penalty_weight}, "samples": NUMBER_OF_SAMPLES, "time": sampleset.info["timing"]})
    