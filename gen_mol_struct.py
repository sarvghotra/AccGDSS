import os
from utils.mol_utils import gen2dmol
from utils.mol_utils import load_smiles, canonicalize_smiles
from rdkit import Chem, RDLogger
from rdkit import DataStructs
import argparse
import numpy as np

from parsers.parser import Parser
from parsers.config import get_config
from sampler import Sampler, Sampler_mol
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity


def select_most_similar_sample(cand_samples, refers):
    # cand_samples: list of smiles
    # refers: list of smiles
    # Select the most similar sample to the reference based on the Tanimoto similarity between two molecules
    # Return most_similar_cand_sample, most_similar_refer_sample, most_similar_score

    most_similar_score = 0
    for cand_sample in cand_samples:
        cand_mol = Chem.MolFromSmiles(cand_sample)
        cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2, nBits=1024) # Chem.RDKFingerprint(cand_mol) #
        cand_fp = np.array(cand_fp)
        cand_fp = cand_fp.reshape(1, -1)

        for refer in refers:
            refer_mol = Chem.MolFromSmiles(refer)
            refer_fp = AllChem.GetMorganFingerprintAsBitVect(refer_mol, 2, nBits=1024)  # Chem.RDKFingerprint(refer_mol)    #
            refer_fp = np.array(refer_fp)
            refer_fp = refer_fp.reshape(1, -1)

            score = cosine_similarity(cand_fp, refer_fp)[0][0]    # DataStructs.FingerprintSimilarity(cand_fp, refer_fp)   #
            if score > most_similar_score:
                most_similar_score = score
                most_similar_cand_sample = cand_sample
                most_similar_refer_sample = refer

    return most_similar_score, most_similar_cand_sample, most_similar_refer_sample


if __name__ == '__main__':
    output_dir = "plots/mol_structs_tanimoto_morgan/"
    n_samples = 50

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    if config.data.data in ['QM9', 'ZINC250k']:
        sampler = Sampler_mol(config)
    else:
        sampler = Sampler(config)

    print("starting")
    train_smiles, test_smiles = load_smiles(config.data.data)
    _, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
    print("finished loading smiles")

    # FIXME
    test_smiles = test_smiles[:500]

    # read the file line by line
    best_samples = []
    highest_score = -1

    for nb_sample_steps in [10, 30, 100, 500, 1000]:

        samples = sampler.custom_sampling(nb_sample_steps, n_samples)

        print("nb_sample_steps: ", nb_sample_steps)
        most_sim_sample = select_most_similar_sample(samples, test_smiles)
        score, gen_sample, test_sample = most_sim_sample
        print("similarty matching finished")

        if score > highest_score:
            highest_score = score
            best_samples = [nb_sample_steps, test_sample]

        print(f"{nb_sample_steps} {score}")
        print()

        output_file = os.path.join(output_dir, f"{nb_sample_steps}_{score}.png")
        gen2dmol(gen_sample, output_file)

    print(f"Best score: {highest_score} steps: {best_samples[0]}")

    best_nb_sample_steps, best_test_sample = best_samples
    output_file = os.path.join(output_dir, f"best_{best_nb_sample_steps}_{highest_score}_test.png")
    gen2dmol(best_test_sample, output_file)
