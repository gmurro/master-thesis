from pprint import pprint
from pyparsing import col
import transformers
from transformers import pipeline

from textattack.attack_recipes import SynBA2022, BAEGarg2019, TextFoolerJin2019
from textattack.datasets import Dataset
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack import Attacker
from textattack import AttackArgs

from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
    AttackTimer,
    AttackMRR
)
from textattack.metrics.quality_metrics import ( 
    Perplexity, 
    SBERTMetric,
    ContradictionMetric
)

import os
import pandas as pd
import tensorflow as tf
import torch
import gc
import pickle
from pprint import pprint
dataset_name = "rotten_tomatoes"
attack_class = SynBA2022

# create dataframe where we will store the results
try:
    stats = pd.read_csv(os.path.join("logs", dataset_name, f"stats_{dataset_name}.csv"), index_col=[0])
except FileNotFoundError:
    stats = pd.DataFrame(columns=['successful_attacks', 'failed_attacks', 'skipped_attacks', 'original_accuracy', 
                            'attack_accuracy_perc', 'attack_success_rate', 'avg_word_perturbed', 'avg_word_perturbed_perc', 
                            'max_words_changed', 'avg_num_queries', 'avg_attack_time', 'avg_word_ranking_time', 
                            'avg_transformation_time', 'avg_num_transformations', 'avg_constraints_time', 'avg_num_constraints', 
                            'avg_batch_query_time', 'avg_num_batch_queries', 
                            'avg_original_perplexity', 
                            'avg_attack_perplexity', 
                            'avg_sentence_bert_similarity', 
                            'attack_contradiction_rate', 'attack_contradiction_rate_by_sentence'])
                            #'avg_mean_rank', 'avg_std_rank', 'avg_min_rank', 'avg_max_rank', 'avg_mrr'])


log_file_name = os.path.join("logs", dataset_name, f"log_{attack_class.__name__}_{dataset_name}.csv")
print(f"Log file: {log_file_name}")

# read attack results from file using pickle
attack_results = pickle.load(open(os.path.join("logs", dataset_name, f"attack_results_{attack_class.__name__}_{dataset_name}.pkl"), "rb"))

# compute metrics
attack_success_stats = AttackSuccessRate().calculate(attack_results)
words_perturbed_stats = WordsPerturbed().calculate(attack_results)
attack_query_stats = AttackQueries().calculate(attack_results)
attack_timer_stats = AttackTimer().calculate(attack_results)
perplexity_stats = Perplexity().calculate(attack_results)
sbert_stats = SBERTMetric().calculate(attack_results)
contraciction_metric = ContradictionMetric(by_sentence=False).calculate(attack_results)
contraciction_metric_by_sentence = ContradictionMetric(by_sentence=True).calculate(attack_results)
#rank_metric = AttackMRR().calculate(attack_results)

dict_stats = {**attack_success_stats, **words_perturbed_stats, **attack_query_stats, **attack_timer_stats,  **perplexity_stats, **sbert_stats,**contraciction_metric, "attack_contradiction_rate_by_sentence": contraciction_metric_by_sentence['attack_contradiction_rate']} 
del dict_stats['num_words_changed_until_success']
dict_stats['avg_constraints_time'] = str(dict_stats['avg_constraints_time'])

# add new row to the stats dataframe
row = pd.DataFrame(dict_stats, index=[attack_class.__name__])
stats =pd.concat([stats, row], axis=0)
stats.to_csv(os.path.join("logs", dataset_name, f"stats_{dataset_name}.csv"))
pprint(dict_stats)