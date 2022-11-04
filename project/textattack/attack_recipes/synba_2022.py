"""

Synba
===================================================
(SynBA: A contextualized Synonim-Based adversarial Attack for Text Classification)

"""

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


from .attack_recipe import AttackRecipe


class SynBA2022(AttackRecipe):
    """Murro, G. (2022).

    SynBA: A contextualized Synonim-Based adversarial Attack for Text Classification
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Candidate size K is set to 30 for all data-sets.
        transformation = WordSwapMultimodal(max_candidates=30, multimodal_weights=(0.284067, 0.107318,0.608615), min_confidence=0.1) #(0.3019, 0.2604, 0.4377), min_confidence=0.1)
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