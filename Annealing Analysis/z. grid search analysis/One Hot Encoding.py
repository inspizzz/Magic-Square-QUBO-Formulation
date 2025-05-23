# pyqubo stuff
from pyqubo import Array, Constraint

import matplotlib.pyplot as plt
import numpy as np
import neal
from itertools import product
import os


# get the vector index given the i, j, k magic square matrix identifiers
def get_index(i, j, k, SQUARE_SIZE, BITS):
    
    # we need to know the length of the magic square matrix horizontally
    row_length = BITS * SQUARE_SIZE
    
    return int((i*row_length) + (j*BITS) + k)


def implement_row(H, variables, SQUARE_SIZE, BITS, penalty):
    """Ensure the all rows sum to the magic constant

    Args:
        variables (Array -> Binary): the binary variables that are being used to encode the results
        penalty (Int): a quantity which scales the impact of the penalty on the objective function
    """
    for i1 in range(SQUARE_SIZE):
        for i2 in range(i1+1, SQUARE_SIZE):

            H_tmp = 0
            
            for j in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp += variables[get_index(i1, j, k, SQUARE_SIZE, BITS)] * k

            for j in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp -= variables[get_index(i2, j, k, SQUARE_SIZE, BITS)] * k

            H_tmp **= 2
            H += Constraint(H_tmp, label="row_constraint")
    
    return penalty * H


def implement_column(H, variables, SQUARE_SIZE, BITS, penalty=1):
    """Aim is to ensure all columns sum to the magic constant

    
    Args:
        variables (_type_): _description_
        penalty (_type_): _description_
    """
    
    for j1 in range(SQUARE_SIZE):
        for j2 in range(j1+1, SQUARE_SIZE):

            H_tmp = 0
            
            for i in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp += variables[get_index(i, j2, k, SQUARE_SIZE, BITS)] * k

            for i in range(SQUARE_SIZE):
                for k in range(BITS):
                    H_tmp -= variables[get_index(i, j1, k, SQUARE_SIZE, BITS)] * k

            H_tmp **= 2
            H += Constraint(H_tmp, label="column_constraint")
    
    return penalty * H


def implement_unique(H, variables, SQUARE_SIZE, BITS, penalty):
    """Aim is to make sure the each number in the magic square is unique

    Args:
        variables (Array -> Binary): the binary variables that are being used to encode the results
        penalty (Int): a quantity which scales the impact of the penalty on the objective function
    """

    for k in range(BITS):
        H += Constraint((sum(variables[get_index(i, j, k, SQUARE_SIZE, BITS)] for i in range(SQUARE_SIZE) for j in range(SQUARE_SIZE)) - 1) ** 2, label="unique")

    return penalty * H


def implement_ohe(H, variables, SQUARE_SIZE, BITS, penalty):
    """Aim is to ensure that the results produced only have one bit set to 1 for a valid encoding


    Args:
        variables (Array -> Binary): the binary variables that are being used to encode the results
        penalty (Int): a quantity which scales the impact of the penalty on the objective function
    """
    
    for i in range(SQUARE_SIZE):
        for j in range(SQUARE_SIZE):
            H_tmp = 0
            
            for k in range(BITS):
                H_tmp += variables[get_index(i, j, k, SQUARE_SIZE, BITS)]

            # H += Constraint(penalty * (H_tmp - 1) ** 2, label="one_hot_encoding")
            H += Constraint((H_tmp**2) - (2 * H_tmp) + 1, label="one_hot_encoding")

    return penalty * H



NUMBER_OF_SAMPLES = 10

# create a list of all possible weight combos
penalty_weights = list(product(range(1, 5), repeat=4))

global iteration


for SQUARE_SIZE in range(5, 6):

    iteration = 1

    for unique_penalty_weight, ohe_penalty_weight, row_penalty_weight, col_penalty_weight in penalty_weights:

        # check if this experiment has already been run
        if os.path.exists(f"annealing analysis/ohe/solutions_{SQUARE_SIZE}.npy"):

            # load the existing data
            existing_data = np.load(f"annealing analysis/ohe/solutions_{SQUARE_SIZE}.npy", allow_pickle=True)
            
            if len(existing_data) > 0:
                if [unique_penalty_weight, ohe_penalty_weight, row_penalty_weight, col_penalty_weight] in [list(i["penalty_weights"].values()) for i in existing_data]:
                    print(f"[EXPERIMENT {iteration}] - skipping experiment for square size {SQUARE_SIZE} with penalty weights {unique_penalty_weight, ohe_penalty_weight, row_penalty_weight, col_penalty_weight}")
                    iteration += 1
                    continue

        # log progress and define bit size
        print(f"[EXPERIMENT {iteration}] - starting experiment for square size {SQUARE_SIZE}")
        BITS = SQUARE_SIZE * SQUARE_SIZE

        array = Array.create('x', SQUARE_SIZE * SQUARE_SIZE * BITS, 'BINARY')

        # apply constraints
        H = 0
        H += implement_ohe(H, array, SQUARE_SIZE, BITS, ohe_penalty_weight)
        H += implement_unique(H, array, SQUARE_SIZE, BITS, unique_penalty_weight)
        H += implement_column(H, array, SQUARE_SIZE, BITS, col_penalty_weight)
        H += implement_row(H, array, SQUARE_SIZE, BITS, row_penalty_weight)

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

        filename = f"Annealing Analysis/ohe/solutions_{SQUARE_SIZE}.npy"
        new_data = {"solutions": solutions, "penalty_weights": {"penalty_unique": unique_penalty_weight, "penalty_unary": ohe_penalty_weight, "penalty_row": row_penalty_weight, "penalty_col": col_penalty_weight}, "samples": NUMBER_OF_SAMPLES, "time": sampleset.info["timing"]}


        if os.path.exists(filename):
            existing_data = np.load(filename, allow_pickle=True)
            combined = np.append(existing_data, np.array([new_data]))
        else:
            combined = [new_data]

        np.save(filename, combined)

        iteration += 1