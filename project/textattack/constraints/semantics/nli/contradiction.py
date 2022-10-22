"""
Contradiction constraint checker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints import Constraint
from textattack.shared import utils
import nltk

sentence_transformers = utils.LazyLoader(
    "sentence_transformers", globals(), "sentence_transformers"
)


class Contradiction(Constraint):
    """
    Constraint checking whether the adversarial sample generated is a contraddiction of the
    original text. It uses Cross-Encoder for Natural Language Inference model.
    Available models can be found here: https://huggingface.co/cross-encoder
    """

    NLI_OUTPUT2IDX = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

    def __init__(
        self,
        model_name="cross-encoder/nli-deberta-v3-base",
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        self.model = sentence_transformers.CrossEncoder(model_name, device=utils.device)

    def _is_contradiction(self, starting_text, transformed_text, by_sentence=False):
        """
        Returns whether the transformed text is a contradiction of the starting text.

        Args:
            starting_text (``str``): The starting text.
            transformed_text (``str``): The transformed text.
            by_sentence (``bool``, optional): Whether to check if the transformed text is a contradiction of the starting text sentence by sentence. Defaults to False.
        """
        if by_sentence:
            # split the text into sentences since the model is trained on sentences, not long texts
            starting_sentences = nltk.sent_tokenize(starting_text.text)
            transformed_sentences = nltk.sent_tokenize(transformed_text.text)
        else:
            starting_sentences = [starting_text.text]
            transformed_sentences = [transformed_text.text]

        output = self.model.predict(list(zip(starting_sentences, transformed_sentences)))
        
        
        # check if there is at least one contradiction
        if Contradiction.NLI_OUTPUT2IDX['contradiction'] in output.argmax(axis=1):
            return True
        else:
            return False

    def _check_constraint(self, transformed_text, reference_text):
        """Return `True` if the `transformed_text` isn't contradiction of the starting text. 
        Return `False` if the `transformed_text` is contradiction of the starting text. 
        """
        if self._is_contradiction(reference_text, transformed_text):
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["model_name"] + super().extra_repr_keys()
