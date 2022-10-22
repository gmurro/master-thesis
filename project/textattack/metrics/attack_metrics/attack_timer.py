"""

Metrics on AttackTimer
---------------------------------------------------------------------

"""

import numpy as np

from textattack.attack_results import SkippedAttackResult
from textattack.metrics import Metric


class AttackTimer(Metric):
    def __init__(self):
        self.all_metrics =  {}

    def calculate(self, results):
        """Calculates all metrics related to the time enlapsed during attack.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """

        self.results = results

        self.all_metrics["avg_attack_time"]  = np.array(
            [
                r.timer.attack_time
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)

        self.all_metrics["avg_word_ranking_time"]  = np.array(
            [
                r.timer.word_ranking_time
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)
        
        self.all_metrics["avg_transformation_time"] = np.array(
             [
                r.timer.avg_transformation_time
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)

        self.all_metrics["avg_num_transformations"] = np.array(
             [
                r.timer.n_transformations
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)
        
        dict_list = [
                r.timer.avg_constraints_time
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        self.all_metrics["avg_constraints_time"] = self.compute_avg_dict(dict_list)

        self.all_metrics["avg_num_constraints"] = np.array(
             [
                r.timer.n_constraints
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)
        
        self.all_metrics["avg_batch_query_time"] = np.array(
             [
                r.timer.avg_query_time
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)

        self.all_metrics["avg_num_batch_queries"] = np.array(
             [
                r.timer.n_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        ).mean().round(3)

        return self.all_metrics

    def compute_avg_dict(self, dict_list):
        avg_dict = {}
        for key in dict_list[0].keys():
            avg_dict[key] = round(sum(d[key] for d in dict_list) / len(dict_list), 3) if len(dict_list) > 0 else 0.0
        return avg_dict
