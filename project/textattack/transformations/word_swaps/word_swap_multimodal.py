"""
Word Swap by BERT-Masked LM.
-------------------------------
"""

import re
import copy

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils, WordEmbedding
from .word_swap import WordSwap

import nltk
from nltk.corpus import wordnet


class WordSwapMultimodal(WordSwap):
    """
    Generate potential replacements for a word using a multiple approaches combined.
    By default, a weighted sum is computed between the confidence score of replacements predicted by the masked language model,
    a score assigned to synonyms provided by WordNet and a score obtained by the frequency of the word.

    Args:
        thesaurus (str): the thesaurus used to generate synonyms (e.g. "wordnet", "babelnet"). Default is "wordnet".
        masked_language_model (Union[str|transformers.AutoModelForMaskedLM]): Either the name of pretrained masked language model from `transformers` model hub
            or the actual model. Default is `bert-base-cased`.
        tokenizer (obj): The tokenizer of the corresponding model. If you passed in name of a pretrained model for `masked_language_model`,
            you can skip this argument as the correct tokenizer can be infered from the name. However, if you're passing the actual model, you must
            provide a tokenizer.
        max_length (int): the max sequence length the masked language model is designed to work with. Default is 512.
        window_size (int): The number of surrounding words to include when making top word prediction.
            For each word to swap, we take `window_size // 2` words to the left and `window_size // 2` words to the right and pass the text within the window
            to the masked language model. Default is `float("inf")`, which is equivalent to using the whole text.
        max_candidates (int): maximum number of candidates to consider as replacements for each word. Replacements are ranked by model's confidence.
        min_confidence (float): minimum confidence threshold each replacement word must pass.
        batch_size (int): Size of batch for the MLM replacement.
    """

    def __init__(
        self,
        thesaurus="wordnet",
        masked_language_model="bert-base-uncased",
        tokenizer=None,
        max_length=512,
        window_size=float("inf"),
        max_candidates=50,
        min_confidence=5e-4,
        batch_size=16,
        multimodal_weights=(0.5, 0.4, 0.1),
        language="eng",
        **kwargs,
    ):
        nltk.download("omw-1.4")

        super().__init__(**kwargs)
        self.thesaurus = thesaurus
        self.max_length = max_length
        self.window_size = window_size
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size

        if language not in wordnet.langs():
            raise ValueError(f"Language {language} not one of {wordnet.langs()}")
        self.language = language

        if isinstance(multimodal_weights, (list, tuple)) and len(multimodal_weights) == 3 and sum(multimodal_weights) == 1:
            self.multimodal_weights = multimodal_weights
        else:
            raise ValueError("multimodal_weights must be a list of length 3 that sums to 1")

        if isinstance(masked_language_model, str):
            self._language_model = AutoModelForMaskedLM.from_pretrained(
                masked_language_model
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                masked_language_model, use_fast=True
            )
        else:
            self._language_model = masked_language_model
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self.tokenizer = tokenizer
        self._language_model.to(utils.device)
        self._language_model.eval()
        self.masked_lm_name = self._language_model.__class__.__name__
        self.embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        


    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding.to(utils.device)

    def _get_synonym_candidates(self, reference_word):
        """Get synonym candidates for the word we want to replace.

        Args:
            reference_word (str): The word we want to replace.
        """
        # initialize the array that will contain a score for each synonym found
        thesaurus_scores = torch.zeros(len(self._lm_tokenizer.vocab), device=utils.device)

        if self.thesaurus == "wordnet":
            # scores assigned to lemmas in the synset
            weights = {
                "synonym": 1.0,
                "antonym": -100.0,
                "hyponym": 0.5,
                "hypernym": 0.5
            }

            # get synonyms from wordnet
            for syn in wordnet.synsets(reference_word, lang=self.language): 
                for lemma in syn.lemmas():
                    lemma_name = lemma.name()
                    if ((lemma_name != reference_word) and ("_" not in lemma_name)):
                        # add synonym score
                        token_id = self._lm_tokenizer.convert_tokens_to_ids(lemma_name)
                        if token_id != self._lm_tokenizer.unk_token_id:
                            thesaurus_scores[token_id] = weights["synonym"]
                        else:
                            self._lm_tokenizer.add_tokens(lemma_name)
                            thesaurus_scores = torch.cat((thesaurus_scores, torch.tensor([weights["synonym"]], device=utils.device)))
                        
                        if lemma.antonyms():
                            # add antonym score
                            antonym = lemma.antonyms()[0].name()
                            token_id = self._lm_tokenizer.convert_tokens_to_ids(antonym)
                            if token_id != self._lm_tokenizer.unk_token_id:
                                thesaurus_scores[token_id] = weights["antonym"]
                            else:
                                self._lm_tokenizer.add_tokens(antonym)
                                thesaurus_scores = torch.cat((thesaurus_scores, torch.tensor([weights["antonym"]], device=utils.device)))

                # add hyponym score
                for hyponym in syn.hyponyms():
                    for lemma in hyponym.lemma_names():
                        if ((lemma != reference_word) and ("_" not in lemma)):
                            token_id = self._lm_tokenizer.convert_tokens_to_ids(lemma)
                            if token_id != self._lm_tokenizer.unk_token_id:
                                thesaurus_scores[token_id] = weights["hyponym"]
                            else:
                                self._lm_tokenizer.add_tokens(lemma)
                                thesaurus_scores = torch.cat((thesaurus_scores, torch.tensor([weights["hyponym"]]).to(utils.device)))

                # add hypernym score
                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemma_names():
                        if ((lemma != reference_word) and ("_" not in lemma)):
                            token_id = self._lm_tokenizer.convert_tokens_to_ids(lemma)
                            if token_id != self._lm_tokenizer.unk_token_id:
                                thesaurus_scores[token_id] = weights["hypernym"]
                            else:
                                self._lm_tokenizer.add_tokens(lemma)
                                thesaurus_scores = torch.cat((thesaurus_scores, torch.tensor([weights["hypernym"]]).to(utils.device)))
        else:
            raise ValueError("Only wordnet thesaurus is supported.")

        return thesaurus_scores
    
    
    def _get_word_embeddings_scores(self, reference_word, topn=50):
        """
        Get the word embedding distance scores between the reference word and the words into the vocabulary.

        Args:
            reference_word (str): The word we want to replace.
            topn (int): Used for specifying N nearest neighbours. Default is 50.
        """
        # initialize the array that will contain a score for each similar word embedding found
        word_embeddings_scores = torch.zeros(len(self._lm_tokenizer.vocab), device=utils.device)

        # check if the reference word is in the vocabulary
        try:
            word_id = self.embedding.word2index(reference_word.lower())
            nnids = self.embedding.nearest_neighbours(word_id, topn)

            for nbr_id in nnids:
                nbr_word = self.embedding.index2word(nbr_id)
                score = self.embedding.get_cos_sim(word_id, nbr_id)

                # add word embedding score
                token_id = self._lm_tokenizer.convert_tokens_to_ids(nbr_word)
                if token_id != self._lm_tokenizer.unk_token_id:
                    word_embeddings_scores[token_id] = score
                else:
                    self._lm_tokenizer.add_tokens(nbr_word)
                    word_embeddings_scores = torch.cat((word_embeddings_scores, torch.tensor([score], device=utils.device)))
           
            return word_embeddings_scores
        except KeyError:
            # if the reference word is not in the vocabulary, return an empty array
            return word_embeddings_scores

    def _is_reference_word_valid(self, reference_word):
        """Check if the reference word is valid.

        Args:
            reference_word (str): the word we want to replace
        """
        if utils.check_if_number(reference_word):
            return False
        if not reference_word.isalpha():
            return False
        if len(reference_word) <= 1:
            return False
        return True

    def _synba_replacement_words(self, current_text, indices_to_modify):
        """Get replacement words for the word we want to replace using SynBA Attack method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
        Returns:
            list: List of replacement words.
        """

        masked_texts = []
        for index in indices_to_modify:
            masked_text = current_text.replace_word_at_index(
                index, self.tokenizer.mask_token
            )
            masked_texts.append(masked_text.text)

        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        replacement_words = []
        while i < len(masked_texts):
            reference_word = current_text.words[indices_to_modify[i]]

            # check if the reference word should be replaced or not
            if not self._is_reference_word_valid(reference_word):
                replacement_words.append([])
            else:
                inputs = self._encode_text(masked_texts[i : i + self.batch_size])
                with torch.no_grad():
                    preds = self._language_model(**inputs).logits

                # retrieve indices of [MASK]
                mask_token_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

                for j, masked_idx in enumerate(mask_token_indices):
                    # make a copy of the tokenizer to avoid modifying the original one
                    self._lm_tokenizer = copy.deepcopy(self.tokenizer)

                    # get the softmax predicted for the [MASK] token
                    mask_token_logits = preds[j, masked_idx]
                    mask_token_prob = torch.softmax(mask_token_logits, dim=0)
                    mlm_scores = utils.MinMaxScaler()(mask_token_prob)
                   
                    # get thesaurus scores
                    thesaurus_scores = self._get_synonym_candidates(reference_word)
                    
                    # get word embedding scores
                    word_embeddings_scores = self._get_word_embeddings_scores(reference_word)

                    # make sures that scores arrays are of the same size
                
                    if ((mlm_scores.size() != thesaurus_scores.size()) or (mlm_scores.size() != word_embeddings_scores.size()) or (thesaurus_scores.size() != word_embeddings_scores.size())):
                        max_size = max(mlm_scores.size()[0], thesaurus_scores.size()[0], word_embeddings_scores.size()[0])
                        mlm_scores = torch.cat((mlm_scores, torch.zeros(max_size - mlm_scores.size()[0]).to(utils.device)))
                        thesaurus_scores = torch.cat((thesaurus_scores, torch.zeros(max_size - thesaurus_scores.size()[0]).to(utils.device)))
                        word_embeddings_scores = torch.cat((word_embeddings_scores, torch.zeros(max_size - word_embeddings_scores.size()[0]).to(utils.device)))
                    
                    # compute the weighted sum of the scores
                    synba_scores = torch.add(mlm_scores*self.multimodal_weights[0], thesaurus_scores*self.multimodal_weights[1]).add_(word_embeddings_scores*self.multimodal_weights[2])
                    ranked_indices = torch.argsort(synba_scores, descending=True)
                    
                    top_words = []
                    for idx in ranked_indices:
                        word = self._lm_tokenizer.convert_ids_to_tokens(idx.item())
                        score = synba_scores[idx.item()]
                        if utils.check_if_subword(
                            word,
                            self._language_model.config.model_type,
                            (masked_idx == 1),
                        ):
                            # not add subword to the candidate list
                            continue
                        if (
                            score >= self.min_confidence
                            and utils.is_one_word(word)
                            and not utils.check_if_punctuations(word)
                            and word != self._lm_tokenizer.unk_token
                        ):
                            top_words.append(word)

                        if (
                            len(top_words) >= self.max_candidates
                            or score < self.min_confidence
                        ):
                            break
                    replacement_words.append(top_words)

            i += self.batch_size

        return replacement_words



    def _get_transformations(self, current_text, indices_to_modify):
        indices_to_modify = list(indices_to_modify)
        replacement_words = self._synba_replacement_words(
            current_text, indices_to_modify
        )
        transformed_texts = []
        for i in range(len(replacement_words)):
            index_to_modify = indices_to_modify[i]
            word_at_index = current_text.words[index_to_modify]
            for word in replacement_words[i]:
                word = word.strip("Ġ")
                word = _recover_word_case(word, word_at_index)
                if (
                    word != word_at_index
                    and re.search("[a-zA-Z -']", word)
                    and len(word) > 1
                ):
                    transformed_texts.append(
                        current_text.replace_word_at_index(index_to_modify, word)
                    )
        return transformed_texts
        
    def extra_repr_keys(self):
        return [
            "thesaurus",
            "masked_lm_name",
            "max_length",
            "max_candidates",
            "min_confidence",
            "multimodal_weights"
        ]


def _recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word   
