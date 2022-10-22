import numpy as np

from textattack.search_methods import GeneticAlgorithm, PopulationMember


class CustomGeneticSearch(GeneticAlgorithm):
    """Attacks a model with word substiutitions using a genetic algorithm.

    Args:
        pop_size (int): The population size. Defaults to 60.
        max_iters (int): The maximum number of iterations to use. Defaults to 20.
        temp (float): Temperature for softmax function used to normalize probability dist when sampling parents.
            Higher temperature increases the sensitivity to lower probability candidates.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        post_crossover_check (bool): If True, check if child produced from crossover step passes the constraints.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Applied only when `post_crossover_check` is set to `True`.
            Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
    """

    def __init__(
        self,
        pop_size=60,
        max_iters=20,
        temp=0.3,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
        max_phenotype_iters=10,
        max_perturb_percent=0.5
    ):
        super().__init__(
            pop_size=pop_size,
            max_iters=max_iters,
            temp=temp,
            give_up_if_no_improvement=give_up_if_no_improvement,
            post_crossover_check=post_crossover_check,
            max_crossover_retries=max_crossover_retries,
        )
        self.max_phenotype_iters = max_phenotype_iters
        self.max_perturb_percent = max_perturb_percent

    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_candidate_transformations` altered appropriately
        for given `word_idx`"""
        increase = np.random.uniform() > 0.2
        num = pop_member.attributes["words_number"]
        max_w = pop_member.attributes["max_words_perturbed"]

        num = num - 1 if increase else num
        if num < 1:
            num = 1

        return PopulationMember(
            new_text,
            result=new_result,
            attributes={
                "words_number": num,
                "max_words_perturbed": max_w,
            },
        )

    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        num_words = len(pop_member.words)
        prob_weights = [0] * num_words
        i = 0
        split_prob = 1 / pop_member.attributes["words_number"]
        while i < pop_member.attributes["words_number"]:
            # select at random a number of words to perturb with equal probability
            idx = int(np.random.uniform(0, num_words))
            if prob_weights[idx] == 0:
                prob_weights[idx] = split_prob
                i += 1
        return prob_weights

    def _crossover_operation(self, pop_member1, pop_member2):
        """Actual operation that takes `pop_member1` text and `pop_member2`
        text and mixes the two to generate crossover between `pop_member1` and
        `pop_member2`.

        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        indices_to_replace = []
        words_to_replace = []

        # Select with 1/2 prob. member 1 candidates and with other half member 2 candidates
        max_words = np.min([pop_member1.attributes["max_words_perturbed"], pop_member2.attributes["max_words_perturbed"]])
        words_no = np.min([pop_member1.attributes["words_number"], pop_member2.attributes["words_number"]])

        for i in range(pop_member1.num_words):
            pop_member = pop_member1 if np.random.uniform() > 0.5 else pop_member2
            pop_member_alt = pop_member1 if pop_member == pop_member2 else pop_member2

            if i < pop_member.attributes["max_words_perturbed"]:
                idx = np.random.randint(0, pop_member.num_words)
                indices_to_replace.append(idx)
                words_to_replace.append(pop_member.words[idx])

            elif i < pop_member_alt.attributes["max_words_perturbed"]:
                idx = np.random.randint(0, pop_member_alt.num_words)
                indices_to_replace.append(idx)
                words_to_replace.append(pop_member_alt.words[idx])

        pop_member = pop_member1 if np.random.uniform() > 0.5 else pop_member2

        new_text = pop_member.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )

        return (
            new_text,
            {
                "words_number": words_no,
                "max_words_perturbed": max_words,
            },
        )

    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with a param of 1 words perturbed.
        Progressively increment no of words until a max is reached
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        max_words = np.floor(len(initial_result.attacked_text.words) * self.max_perturb_percent).astype(int)

        population = []
        for i in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={
                    "words_number": len(initial_result.attacked_text.words),
                    "max_words_perturbed": max_words,
                    "indices": []
                },
            )
            # Perturb `pop_member` in-place
            pop_member = self._perturb(pop_member, initial_result)
            population.append(pop_member)

        return population
