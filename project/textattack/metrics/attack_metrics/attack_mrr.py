"""

Metrics on AttackMRR
---------------------------------------------------------------------

"""

import numpy as np

from textattack.attack_results import SkippedAttackResult, FailedAttackResult
from textattack.metrics import Metric


class AttackMRR(Metric):
    def __init__(self):
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates all metrics related to mean recipocal rank of word substitution during an attack.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """

        self.results = results
        self.mean_ranks = []
        self.mean_reciprocal_ranks = []
        for result in self.results:
            if isinstance(result, FailedAttackResult) or isinstance(
                result, SkippedAttackResult
            ):
                continue
            
            # compute mean rank and mean reciprocal rank only for successful attacks
            transformations_rank = result.perturbed_result.attacked_text.attack_attrs['transformations_rank']
            self.mean_ranks.append(np.mean(transformations_rank))
            self.mean_reciprocal_ranks.append(self.mean_reciprocal_rank(transformations_rank))

        self.all_metrics["avg_mean_rank"] = self.avg_mean_rank()
        self.all_metrics["avg_mrr"] = self.avg_mrr()

        return self.all_metrics

    def mean_reciprocal_rank(self, rankings):
        """Calculates the mean reciprocal rank of a list of rankings.

        Args:
            rankings (list): A list of rankings.

        Returns:
            float: The mean reciprocal rank of the list of rankings.
        """
        return np.mean([1.0 / r for r in rankings])

    def avg_mrr(self):
        """Calculates the average mean reciprocal rank of all successful attacks."""

        mean_reciprocal_ranks = np.array(self.mean_reciprocal_ranks)
        avg_mrr = mean_reciprocal_ranks.mean() if len(mean_reciprocal_ranks) > 0 else 0.0
        avg_mrr = round(avg_mrr, 4)
        return avg_mrr

    def avg_mean_rank(self):
        """Calculates the average mean rank of all successful attacks."""

        mean_ranks = np.array(self.mean_ranks)
        avg_mean_rank = mean_ranks.mean() if len(mean_ranks) > 0 else 0.0
        avg_mean_rank = round(avg_mean_rank, 4)
        return avg_mean_rank