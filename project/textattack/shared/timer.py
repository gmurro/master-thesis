import time
from textattack.shared import utils

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    """ 
    Class used to compute the execution time of a each step during the attack.
    """
    def __init__(self):
        # auxiliary variables to compute the average incrementally
        self.n_transformations = 0
        self.n_constraints = 0
        self.n_queries = 0

        self.attack_time = 0
        self.word_ranking_time = 0                             
        self.avg_transformation_time = 0 
        self.avg_constraints_time = {}
        self.avg_query_time = 0

    def start(self):
        """Start a new timer"""
        return time.perf_counter()

    def stop(self, start_time):
        """Stop the timer, and return the elapsed time in seconds"""
        return time.perf_counter() - start_time
    
    def update_attack_time(self, time_elapsed):
        """Update the time needed to perform the attack on one input"""
        self.attack_time = time_elapsed
    
    def update_word_ranking_time(self, time_elapsed):
        """ Update the total time needed to rank words to perturb from the text (WIR, random, etc.)"""
        self.word_ranking_time = time_elapsed
    
    def update_transformation_time(self, time_elapsed):
        """Update the transformation time computing the average incrementally"""
        self.avg_transformation_time += (time_elapsed - self.avg_transformation_time) / float(self.n_transformations + 1)
        self.n_transformations += 1

    def update_constraints_time(self, time_elapsed, constraint_class):
        """Update the post transformation constraints time computing the average incrementally"""
        self.avg_constraints_time.setdefault(constraint_class, 0)
        self.avg_constraints_time[constraint_class] += (time_elapsed - self.avg_constraints_time[constraint_class]) / float(self.n_constraints + 1)
        self.n_constraints += 1

    def update_query_time(self, time_elapsed):
        """Update the query time computing the average incrementally"""
        self.avg_query_time += (time_elapsed - self.avg_query_time) / float(self.n_queries + 1)
        self.n_queries += 1
    
    def __str__(self) -> str:
        main_str = "Timer" + "("
        lines = []
        lines.append(utils.add_indent(f"(word_ranking_time): {self.word_ranking_time}", 2))
        lines.append(utils.add_indent(f"(avg_transformation_time): {self.avg_transformation_time}", 2))
        lines.append(utils.add_indent(f"(avg_constraints_time): {self.avg_constraints_time}", 2))
        lines.append(utils.add_indent(f"(avg_query_time): {self.avg_query_time}", 2))
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str