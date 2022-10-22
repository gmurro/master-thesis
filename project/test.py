from tqdm import tqdm # tqdm provides us a nice progress bar.
import pandas as pd
import random

import time

import transformers
from transformers import pipeline

from textattack.attack_recipes import SynBA2022, CustomGeneticAttack, TextFoolerJin2019

from textattack.datasets import Dataset
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack.shared import utils
import os

from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs

from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
    AttackTimer
)
from textattack.metrics.quality_metrics import ( 
    Perplexity, 
    USEMetric,
    SBERTMetric,
    BERTScoreMetric,
    MeteorMetric,
    ContradictionMetric
)

from pprint import pprint

# Import the model
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes")
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


# Import the dataset
dataset_name = "rotten_tomatoes"
dataset = HuggingFaceDataset(dataset_name, None, "test")
# dataset_name = "custom"
# custom_dataset = [
#     ('Funny and quirky and rough in spots, just like an indie movie should be.', 1),
#     ('How many times can you say BORING? This is very close to sickening boring.', 0),
#     ('Cute movie. Getting second chances when things don\'t go the way we have planned.', 1),
#     ('Could be the worst movie ever made, I know it\'s the worst I\'ve ever seen! That\'s two hours of my life I won\'t get back, what a waste!', 0),
# ]
# dataset = Dataset(custom_dataset)
#dataset = Dataset([(data['text'], data['label']) for data in dataset._dataset if len(data['text']) > 100 and  dataset._dataset if len(data['text']) < 200])
#text = "I like the vibe here.  Got my turkey burger and caesar salad.  The two things I really love in this world!  You should try their mac n cheese bites, steak and frites, and their huge chocolate cake.   Tasty, great service."
#text = """Personally, I think this place is more hype than anything... I've been here about three times to try different flavors. Let's see - red velvet, chocolate marshmallow, and carrot... all of them were just \""meh\"" to me. \n\nI like how the place looks and how the cupcakes were displayed, also the dog cookies were cute. I probably liked the packaging more than anything."""
#text = "The waiter last night sucked a big one. 08/05/2011 \n\nI'll never go back. Too bad because its just around the corner from me and we used to be regulars."
#text = """I have eaten here many a times and I highly recommend.\n\nI have never had a problem and the food is pretty good."""
#dataset = Dataset([(text, 0)])

seed = 7685 # random.randint(0, 10000)
print("Seed: ",seed)


attack = SynBA2022.build(model_wrapper)
attack_args = AttackArgs(num_examples=2, shuffle=True, random_seed=seed, use_timer=True)
attacker = Attacker(attack, dataset, attack_args)
attack_results = attacker.attack_dataset()

contraciction_metric = ContradictionMetric().calculate(attack_results)
print("Contradiction Metric: ", contraciction_metric)

sbert_metric = SBERTMetric().calculate(attack_results)
print("BERT similarity: ", sbert_metric)

attack_timer_stats = AttackTimer().calculate(attack_results)
print("Attack timer stats:")
pprint(attack_timer_stats)

# attack_success_stats = AttackSuccessRate().calculate(attack_results)
# words_perturbed_stats = WordsPerturbed().calculate(attack_results)
# attack_query_stats = AttackQueries().calculate(attack_results)
# #perplexity_stats = Perplexity().calculate(attack_results)
# #use_stats = USEMetric().calculate(attack_results)
# sbert_stats = SBERTMetric().calculate(attack_results)
# bert_score_stats = BERTScoreMetric().calculate(attack_results)
# meteor_stats = MeteorMetric().calculate(attack_results)

# print(f"Attack Success Rate: {attack_success_stats}")
# print(f"Words Perturbed: {words_perturbed_stats}")
# print(f"Attack Queries: {attack_query_stats}")
# #print(f"Perplexity: {perplexity_stats}")
# #print(f"Universal Sentence Encoder: {use_stats}")
# print(f"SBERT: {sbert_stats}")
# print(f"BERTScore: {bert_score_stats}")
# print(f"Meteor: {meteor_stats}")