from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.grammaticality.language_models import (
    Google1BillionWordsLanguageModel, LearningToWriteLanguageModel
)
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.bert_score import BERTScore
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import CustomGeneticSearch
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import WordSwapEmbedding

from textattack.attack_recipes.attack_recipe import AttackRecipe


class CustomGeneticAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their embedding nearest-neighbors.
        #
        # Embedding: Counter-fitted Paragram Embeddings.
        #
        # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
        #
        transformation = WordSwapEmbedding(max_candidates=8)
        #
        # Don't modify the same word twice or stopwords
        #
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        constraints = [RepeatModification(), 
                        StopwordModification(stopwords=stopwords),
                       MaxModificationRate(max_rate=0.5, min_threshold=4), 
                       PartOfSpeech(allow_verb_noun_swap=True),
                       WordEmbeddingDistance(max_mse_dist=0.6, include_unknown_words=True),
                       BERT(model_name="bert-base-nli-mean-tokens", threshold=0.7)]
        #
        # Language Model
        #
        constraints.append(
            LearningToWriteLanguageModel(
                window_size=6,
                max_log_prob_diff=5.0,
                compare_against_original=True
            )
        )
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = CustomGeneticSearch(
            pop_size=30,
            max_iters=15,
            temp=0.2985,
            post_crossover_check=False,
            give_up_if_no_improvement=True,
            max_crossover_retries=20
        )
        return Attack(goal_function, constraints, transformation, search_method)
