# pyqubo stuff
from pyqubo import Array, Constraint
from mpl_toolkits.mplot3d import Axes3D

import math
import matplotlib.pyplot as plt
import numpy as np
import neal
import re
from itertools import product
import os

# get the vector index given the i, j, k magic square matrix identifiers
def get_index(i, j, k, SQUARE_SIZE, BITS):
    
    # we need to know the length of the magic square matrix horizontally
    row_length = BITS * SQUARE_SIZE
    
    return int((i*row_length) + (j*BITS) + k)


def implement_row(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for row1 in range(SQUARE_SIZE):
        for row2 in range(row1, SQUARE_SIZE):
            if row1 == row2:
                continue

            H_row1 = 0
            H_row2 = 0

            for j in range(SQUARE_SIZE):
                H_row1 += Constraint(penalty * np.sum([list(variables)[get_index(row1, j, k, SQUARE_SIZE, BITS)] * np.pow(2, BITS-k-1) for k in range(BITS)]), label=f"row_{row1}_{j}")

            for j in range(SQUARE_SIZE):
                H_row2 += Constraint(penalty * np.sum([list(variables)[get_index(row2, j, k, SQUARE_SIZE, BITS)] * np.pow(2, BITS-k-1) for k in range(BITS)]), label=f"row_{row2}_{j}")


            H += (H_row1 - H_row2) ** 2
        
    return H
            

def implement_col(H, variables, SQUARE_SIZE, BITS, penalty=1):
    for col1 in range(SQUARE_SIZE):
        for col2 in range(col1, SQUARE_SIZE):
            if col1 == col2:
                continue

            H_col1 = 0
            H_col2 = 0

            for i in range(SQUARE_SIZE):
                H_col1 += Constraint(penalty * np.sum([list(variables)[get_index(i, col1, k, SQUARE_SIZE, BITS)] * np.pow(2, BITS-k-1) for k in range(BITS)]), label=f"row_{i}_{col1}")

            for i in range(SQUARE_SIZE):
                H_col2 += Constraint(penalty * np.sum([list(variables)[get_index(i, col2, k, SQUARE_SIZE, BITS)] * np.pow(2, BITS-k-1) for k in range(BITS)]), label=f"row_{i}_{col2}")


            H += (H_col1 - H_col2) ** 2
        
    return H
            
            
def implement_unique(H, variables, SQUARE_SIZE, BITS, penalty=1):
    
    for i1 in range(SQUARE_SIZE):
        for j1 in range(SQUARE_SIZE):
            for i2 in range(i1, SQUARE_SIZE):
                for j2 in range(SQUARE_SIZE):
                    if i1 == i2 and j2 <= j1:
                        continue

                    H_tmp = 0

                    for k in range(BITS):
                        var1 = variables[get_index(i1, j1, k, SQUARE_SIZE, BITS)]
                        var2 = variables[get_index(i2, j2, k, SQUARE_SIZE, BITS)]
                        
                        H_tmp += 1 - ((var1 + var2) - (2 * var1 * var2))

                    H +=  Constraint(H_tmp, label=f"unique_{i1}_{j1}_{i2}_{j2}")
                    
    return penalty * H

                

NUMBER_OF_SAMPLES = 10

# create a list of all possible weight combos
penalty_weights = list(product(range(1, 5), repeat=3))

global iteration



for SQUARE_SIZE in range(4, 6):

    iteration = 1

    for unique_penalty_weight, row_penalty_weight, col_penalty_weight in penalty_weights:

        # check if this experiment has already been run
        if os.path.exists(f"annealing analysis/binary/solutions_{SQUARE_SIZE}.npy"):

            # load the existing data
            existing_data = np.load(f"annealing analysis/binary/solutions_{SQUARE_SIZE}.npy", allow_pickle=True)
            
            if len(existing_data) > 0:
                if [unique_penalty_weight, row_penalty_weight, col_penalty_weight] in [list(i["penalty_weights"].values()) for i in existing_data]:
                    print(f"[EXPERIMENT {iteration}] - skipping experiment for square size {SQUARE_SIZE} with penalty weights {unique_penalty_weight, row_penalty_weight, col_penalty_weight}")
                    iteration += 1
                    continue

        # debug
        print(f"[EXPERIMENT {iteration}] - starting experiment for square size {SQUARE_SIZE}")

        # calculate the number of bits and vars
        BITS = math.floor(math.log2(SQUARE_SIZE**2) + 1)

        # create binary variables
        array = Array.create('x', SQUARE_SIZE * SQUARE_SIZE * BITS, 'BINARY')

        # apply constraints
        H = 0
        H += implement_unique(H, array, SQUARE_SIZE, BITS, unique_penalty_weight)
        H += implement_row(H, array, SQUARE_SIZE, BITS, row_penalty_weight)
        H += implement_col(H, array, SQUARE_SIZE, BITS, col_penalty_weight)

        # compile and convert to binary quadratic model
        model = H.compile()
        bqm = model.to_bqm()

        # simulate
        sa = neal.SimulatedAnnealingSampler()
        sampleset = sa.sample(bqm, num_reads=NUMBER_OF_SAMPLES)
        decoded_samples = model.decode_sampleset(sampleset)

        # decode
        solutions = []

        # save the samples outcome and energies
        for solution in decoded_samples:
            
            # get the sample and energy
            sample = solution.sample
            energy = solution.energy
            
            sample = np.array([sample[f"x[{i}]"] for i in range(SQUARE_SIZE * SQUARE_SIZE * BITS)]).reshape((SQUARE_SIZE, SQUARE_SIZE, BITS))  
            solutions.append({"sample": sample, "energy": energy})

        # find the file and create data json
        filename = f"Annealing Analysis/binary/solutions_{SQUARE_SIZE}.npy"
        new_data = {"solutions": solutions, "penalty_weights": {"penalty_unique": unique_penalty_weight, "penalty_row": row_penalty_weight, "penalty_col": col_penalty_weight}, "samples": NUMBER_OF_SAMPLES, "time": sampleset.info["timing"]}

        # create list if file does not exist, else append to it
        if os.path.exists(filename):
            existing_data = np.load(filename, allow_pickle=True)
            combined = np.append(existing_data, np.array([new_data]))
        else:
            combined = [new_data]

        # save the data
        np.save(filename, combined)

        # increment the iteration
        iteration += 1
    