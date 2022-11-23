import random
import torch
import gc

import transformers
from transformers import logging
logging.set_verbosity_error() # disable transformers logging

import warnings
warnings.filterwarnings('ignore')

from textattack.attack_recipes import SynBA2022

from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
    InputColumnModification
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import BERT
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapMultimodal
from textattack import Attacker
from textattack import AttackArgs

from textattack.metrics.quality_metrics import (
    SBERTMetric,
    ContradictionMetric
)
from textattack.metrics.attack_metrics import (
    AttackSuccessRate
)
import pandas as pd

# Import HyperOpt Library
from hyperopt import tpe, hp, fmin

# Import the model
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Import the dataset
dataset_name = "yelp_polarity"
dataset = HuggingFaceDataset(dataset_name, None, "test")

# whether to make the attack reproducible or set it random for each run
random_seed = 765

# set the number of samples to attack
num_examples = 500

# create dataframe where we will store the results
df = pd.DataFrame(columns=['w1', 'w2', 'w3', 'succ', 'fail', 'skip', 'attack_contradiction_rate', 'avg_sentence_bert_similarity', 'loss'])

def build_synba(model_wrapper, lambda_mlm, lambda_thesaurus, lambda_we):
    """
    Build the attack, with the given hyperparameters
    """
    #
    # Candidate size K is set to 30 for all data-sets.
    transformation = WordSwapMultimodal(max_candidates=30, multimodal_weights=(lambda_mlm, lambda_thesaurus, lambda_we), min_confidence=0.1)
    #
    # Don't modify the same word twice or stopwords.
    #
    constraints = [RepeatModification(), StopwordModification()]
    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )
    constraints.append(input_column_modification)
    constraints.append(MaxModificationRate(max_rate=0.2, min_threshold=4))
    constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
    constraints.append(WordEmbeddingDistance(min_cos_sim=0.6, include_unknown_words=True))
    constraints.append(BERT(model_name="stsb-mpnet-base-v2", threshold=0.7, metric="cosine"))
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model_wrapper)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    search_method = GreedyWordSwapWIR(wir_method="gradient")

    return Attack(goal_function, constraints, transformation, search_method)

def perform_attack(model, dataset, num_examples, random_seed, lambda_mlm, lambda_thesaurus, lambda_we):
    """
    Set up the attack and perform it on the dataset.
    Returns the metrics that we want to optimize.
    """
    attack = build_synba(model, lambda_mlm, lambda_thesaurus, lambda_we)
    attack_args = AttackArgs(num_examples=num_examples, shuffle=False, random_seed=random_seed, disable_stdout=True, silent=True)
    attacker = Attacker(attack, dataset, attack_args)
    attack_results = attacker.attack_dataset()

    # compute metrics
    attack_success_stats = AttackSuccessRate().calculate(attack_results)
    sbert_stats = SBERTMetric().calculate(attack_results)["avg_sentence_bert_similarity"]
    contradiction_stats = ContradictionMetric(by_sentence=True).calculate(attack_results)["attack_contradiction_rate"]

    # free memory after performing the attack
    gc.collect()
    torch.cuda.empty_cache()
    return attack_success_stats, sbert_stats, contradiction_stats


def optimize(args):
    """
    This is the function that hyperopt will maximize.
    """
    # make sure that the hyperparameters sum up to 1
    sum = args["w1"] + args["w2"] + args["w3"]
    w1 = args["w1"]/sum
    w2 = args["w2"]/sum
    w3 = args["w3"]/sum

    print(f"Now tring --> lambda_mlm: {round(w1, 4)}, lambda_thesaurus: {round(w2, 4)}, lambda_we: {round(w3, 4)}")
    attack_success_stats, sbert_similarity, contraddiction_rate = perform_attack(model_wrapper, dataset, num_examples, random_seed, w1, w2, w3)
    
    print(f"Succ/Fail/Skip: {attack_success_stats['successful_attacks']}/{attack_success_stats['failed_attacks']}/{attack_success_stats['skipped_attacks']}")
    print(f"--> SBERT Similarity: {sbert_similarity}, Contraddiction Rate: {contraddiction_rate}")
    
    # compute penality for failed attacks
    penalty = 0.2 * (attack_success_stats["failed_attacks"] / (attack_success_stats["failed_attacks"]+attack_success_stats["successful_attacks"]))

    # objective to maximize (minimize the negative)
    loss = -(sbert_similarity * (1-contraddiction_rate)) + penalty
    print(f"--> Loss: {loss}\n")

    # store the results in the dataframe
    df.loc[len(df)] = [w1, w2, w3, attack_success_stats["successful_attacks"], attack_success_stats["failed_attacks"], attack_success_stats["skipped_attacks"], contraddiction_rate, sbert_similarity, loss]
    
    # write partial results to file
    df.to_csv("results_rotten_1.csv", index=False)
    return loss

# define the search space across the hyperparameters
space = hp.choice(
    "weights",
    [
        {
            "w1": hp.uniform("w1", 0, 1),
            "w2": hp.uniform("w2", 0, 1),
            "w3": hp.uniform("w3", 0, 1),
        },
    ],
)

best = fmin(
    optimize,
    space,
    algo=tpe.suggest,
    max_evals=120,
)

print("\nThe best combination of hyperparameters is:")
sum = best["w1"] + best["w2"] + best["w3"]
w1 = best["w1"]/sum
w2 = best["w2"]/sum
w3 = best["w3"]/sum
print(f"lambda_mlm: {round(w1, 4)}, lambda_thesaurus: {round(w2, 4)}, lambda_we: {round(w3, 4)}")
