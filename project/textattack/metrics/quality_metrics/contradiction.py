"""

ContradictionMetric class:
-------------------------------------------------------
Class for calculating the contradiction metric on AttackResults

"""

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.constraints.semantics.nli import Contradiction
from textattack.metrics import Metric


class ContradictionMetric(Metric):
    def __init__(
            self,
            model_name="cross-encoder/nli-deberta-v3-base", 
            by_sentence=False,
            **kwargs
            ):

        self.model = Contradiction(model_name)
        self.by_sentence = by_sentence
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates rate of contraddictions on all successfull attacks.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset

        Example::


            >> import textattack
            >> import transformers
            >> model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
            >> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
            >> dataset = textattack.datasets.HuggingFaceDataset("glue", "sst2", split="train")
            >> attack_args = textattack.AttackArgs(
                num_examples=1,
                log_to_csv="log.csv",
                checkpoint_interval=5,
                checkpoint_dir="checkpoints",
                disable_stdout=True
            )
            >> attacker = textattack.Attacker(attack, dataset, attack_args)
            >> results = attacker.attack_dataset()
            >> contradiction_rate = textattack.metrics.quality_metrics.ContradictionMetric().calculate(results)
        """

        self.results = results

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text)
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text)

        # count number of contradictions in successful candidates
        contradiction_counter = 0
        for c in range(len(self.original_candidates)):
            
            if self.model._is_contradiction(
                self.original_candidates[c], self.successful_candidates[c], self.by_sentence
            ):
                contradiction_counter += 1


        self.all_metrics["attack_contradiction_rate"] = round(
            contradiction_counter / len(self.original_candidates), 3
        ) if len(self.original_candidates) > 0 else 0.0

        return self.all_metrics
