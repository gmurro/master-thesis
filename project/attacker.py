# add path to use local version of textattack library
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db import User

from app.schemas_analysis import AdversarialAttackConfig, AnalysisRawResult

import random
import transformers

import textattack
from textattack.attack_recipes import SynBA2022, TextFoolerJin2019, CustomGeneticAttack

from textattack.datasets import Dataset
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack import Attacker
from textattack import AttackArgs

from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult

from textattack.shared import utils


def show_model_output(attack_result):
    """
    Return html version of model output and confidence score with a newline in between
    """
    html = attack_result.get_colored_output("html")
    return "<br/>(".join(html.split(" ("))


def run_analysis_adversarial_attack(config: AdversarialAttackConfig, user: User, record: AnalysisRawResult):
    """
    Run adversarial attack analysis, using TextAttack library
    """    
    try:

        # Import the model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(config.attacked_model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.attacked_model)
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

        # Import the dataset
        dataset = HuggingFaceDataset(config.dataset_used, None, "test")

        # Build the attack
        if config.type_of_attack == "SynBA":
            attack = SynBA2022.build(model_wrapper)
        elif config.type_of_attack == "TextFooler":
            attack = TextFoolerJin2019.build(model_wrapper)
        elif config.type_of_attack == "GeneticAttack":
            attack = CustomGeneticAttack.build(model_wrapper)
        else:
            raise ValueError("Attack method not supported")
        
        # Build the attacker
        if config.is_random_seed:
            seed = random.randint(0, 100000)
        else:
            seed = 7685

        use_timer = True if config.is_advanced_metrics else False

        attack_args = AttackArgs(num_examples=config.samples_under_attack, shuffle=True, disable_stdout=True, random_seed=seed, use_timer=use_timer)
        attacker = Attacker(attack, dataset, attack_args)
        attack_results = attacker.attack_dataset()

        attack_success_stats = textattack.metrics.attack_metrics.AttackSuccessRate().calculate(attack_results)
        words_perturbed_stats = textattack.metrics.attack_metrics.WordsPerturbed().calculate(attack_results)
        attack_query_stats = textattack.metrics.attack_metrics.AttackQueries().calculate(attack_results)

        if config.is_advanced_metrics: 
            timer_stats = textattack.metrics.attack_metrics.AttackTimer().calculate(attack_results)
            sbert_stats = textattack.metrics.quality_metrics.SBERTMetric().calculate(attack_results)
            nli_stats = textattack.metrics.quality_metrics.ContradictionMetric(by_sentence=True).calculate(attack_results)
            perplexity_stats = textattack.metrics.quality_metrics.Perplexity().calculate(attack_results)
        
        output = {
            "successful_attacks": attack_success_stats["successful_attacks"],
            "failed_attacks": attack_success_stats["failed_attacks"],
            "skipped_attacks": attack_success_stats["skipped_attacks"]
        }
        
        results = []
        for i, res in enumerate(attack_results):

            if isinstance(res, SuccessfulAttackResult):
                original_output = show_model_output(res.original_result)
                attacked_output = show_model_output(res.perturbed_result)
                original_text = res.diff_color("html")[0]
                attacked_text = res.diff_color("html")[1]
            elif isinstance(res, FailedAttackResult):
                original_output = show_model_output(res.original_result)
                attacked_output = utils.color_text("FAILED", "red", "html")
                original_text = res.original_result.attacked_text.text
                attacked_text = ""
            else:
                original_output = show_model_output(res.original_result)
                attacked_output = utils.color_text("SKIPPED", "black", "html")
                original_text = res.original_result.attacked_text.text
                attacked_text = ""

            results.append({
                "n": i+1,
                "original_text": original_text,
                "attacked_text": attacked_text,
                "original_words": res.original_result.attacked_text.words,
                "attacked_words": res.perturbed_result.attacked_text.words,            
                "original_output": original_output,
                "attacked_output": attacked_output,
                "ground_truth_output": res.original_result.ground_truth_output,
                "ranking_indices": [],
                "perturbed_indices": res.perturbed_result.attacked_text.attack_attrs["modified_indices"] if res.perturbed_result.attacked_text.attack_attrs["modified_indices"] else [],
            })

        if config.is_advanced_metrics: 
            res = {
                "original_accuracy": attack_success_stats["original_accuracy"],
                "attacked_accuracy": attack_success_stats["attack_accuracy_perc"],
                "attack_success_rate": attack_success_stats["attack_success_rate"],
                "avg_word_perturbed_perc": words_perturbed_stats["avg_word_perturbed_perc"],
                "avg_word_input": words_perturbed_stats["avg_word_perturbed"],
                "avg_num_queries": attack_query_stats["avg_num_queries"],
                "avg_attack_time": timer_stats["avg_attack_time"],
                "avg_query_time": timer_stats["avg_batch_query_time"],
                "avg_sentence_bert_similarity": sbert_stats["avg_sentence_bert_similarity"],
                "avg_original_perplexity": perplexity_stats["avg_original_perplexity"],
                "avg_attack_perplexity": perplexity_stats["avg_attack_perplexity"],
                "attack_contradiction_rate": round(nli_stats["attack_contradiction_rate"]*100, 2),
                "results": results
            }
        else:
            res = {
                "original_accuracy": attack_success_stats["original_accuracy"],
                "attacked_accuracy": attack_success_stats["attack_accuracy_perc"],
                "attack_success_rate": attack_success_stats["attack_success_rate"],
                "avg_word_perturbed_perc": words_perturbed_stats["avg_word_perturbed_perc"],
                "avg_word_input": words_perturbed_stats["avg_word_perturbed"],
                "avg_num_queries": attack_query_stats["avg_num_queries"],
                "results": results
            }
        
        return output, res
    except Exception as e:
        import torch
        import gc

        # free memory in case of error
        del model
        del tokenizer
        del model_wrapper
        del dataset
        gc.collect()
        torch.cuda.empty_cache()
        raise e

    
