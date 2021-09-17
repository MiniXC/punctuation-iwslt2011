"""The IWSLT Challenge Dataset, adapted to punctuation as described by Ueffing et al. (2013)"""

from enum import Enum
from typing import Union
from abc import abstractmethod
import logging
import itertools

#import paired
from xml.dom import minidom
import nltk
import datasets
import numpy as np

nltk.download("punkt")
tknzr = nltk.tokenize.TweetTokenizer()

_CITATION = """\
@inproceedings{Ueffing2013,
    title={Improved models for automatic punctuation prediction for spoken and written text},
    author={B. Ueffing and M. Bisani and P. Vozila},
    booktitle={INTERSPEECH},
    year={2013}
}
@article{Federico2011,
    author = {M. Federico and L. Bentivogli and M. Paul and S. Stüker},
    year = {2011},
    month = {01},
    pages = {},
    title = {Overview of the IWSLT 2011 Evaluation Campaign},
    journal = {Proceedings of the International Workshop on Spoken Language Translation (IWSLT), San Francisco, CA}
}
"""

_DESCRIPTION = """\
Both manual transcripts and ASR outputs from the IWSLT2011 speech translation evalutation campaign are often used for the related \
punctuation annotation task. This dataset takes care of preprocessing said transcripts and automatically inserts punctuation marks \
given in the manual transcripts in the ASR outputs using Levenshtein aligment.
"""

_VERSION = "0.0.1"

def window(a, w = 4, o = 2):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    return view.copy()

class Punctuation(Enum):
    NONE = "<none>"
    PERIOD = "<period>"
    COMMA = "<comma>"
    QUESTION = "<question>"

class LabelSubword(Enum):
    IGNORE = "<ignore>"
    NONE = "<none>"


class Task(Enum):
    TAGGING = 0
    SEQ2SEQ = 1

class TaggingTask:
    """Treat punctuation prediction as a sequence tagging problem."""

    def __init__(
        self, window_size=120, window_stride_in_percent=0.5, include_reference=False
    ):
        self.window_size = window_size
        self.window_stride_in_percent = window_stride_in_percent
        self.include_reference = include_reference

    def __eq__(self, other):
        return Task.TAGGING == other

class DecodingStrategy:
    """Strategy used to decode results."""

    def __init__(
        self, task: Union[TaggingTask]
    ):
        self.task = task

    @abstractmethod
    def decode():
        pass

class AverageDecodingStrategy:
    """Averages predictions together."""

    def decode():
        pass

class IWSLT11Config(datasets.BuilderConfig):
    """The IWSLT11 Dataset."""

    def __init__(
        self,
        task: Union[TaggingTask] = TaggingTask(),
        segmented: bool = False,
        asr_or_ref: str = "ref",
        decoder: DecodingStrategy = AverageDecodingStrategy(),
        tokenizer = None,
        label_subword: LabelSubword = LabelSubword.IGNORE,
        **kwargs
    ):
        """BuilderConfig for IWSLT2011.
        Args:
          task: the task to prepare the dataset for.
          segmented: if segmentation present in IWSLT2011 should be respected. removes segmenation by default.
          **kwargs: keyword arguments forwarded to super.
        """
        self.task = task
        self.segmented = segmented
        self.asr_or_ref = asr_or_ref
        self.decoder = decoder
        self.punctuation = [
            Punctuation.NONE,
            Punctuation.PERIOD,
            Punctuation.COMMA,
            Punctuation.QUESTION,
        ]
        if label_subword.IGNORE:
            self.punctuation.append(label_subword.IGNORE)
        self.label_subword = label_subword
        self.tokenizer = tokenizer
        super(IWSLT11Config, self).__init__(**kwargs)

    def __eq__(self, other):
        return True


class IWSLT11(datasets.GeneratorBasedBuilder):
    """The IWSLT11 Dataset, adapted for punctuation prediction."""

    BUILDER_CONFIGS = [
        IWSLT11Config(name="ref", asr_or_ref="ref"),
        IWSLT11Config(name="asr", asr_or_ref="asr"),
    ]

    def _info(self):
        if self.config.task == Task.TAGGING:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "ids": datasets.Sequence(datasets.Value("int32")),
                        "tokens": datasets.Sequence(datasets.Value("string")),
                        "labels": datasets.Sequence(
                           datasets.features.ClassLabel(
                               names=[p.name for p in self.config.punctuation]
                           )
                        ),
                    }
                ),
                supervised_keys=None,
                homepage="http://iwslt2011.org/doku.php",
                citation=_CITATION,
                version=_VERSION,
            )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        urls_to_download = {
            "train": "https://raw.githubusercontent.com/IsaacChanghau/neural_sequence_labeling/master/data/raw/LREC_converted/train.txt",
            "valid": "https://github.com/IsaacChanghau/neural_sequence_labeling/blob/master/data/raw/LREC_converted/dev.txt?raw=true",
            "test_ref": "https://github.com/IsaacChanghau/neural_sequence_labeling/raw/master/data/raw/LREC_converted/ref.txt",
            "test_asr": "https://github.com/IsaacChanghau/neural_sequence_labeling/raw/master/data/raw/LREC_converted/asr.txt",
        }
        files = dl_manager.download_and_extract(urls_to_download)

        if self.config.asr_or_ref == "asr":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": files["train"]
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": files["valid"]
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": files["test_asr"]
                    },
                ),
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": files["train"]
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": files["valid"]
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": files["test_ref"]
                    },
                ),
            ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)

        text = open(filepath).read()
        text = (
            text
            .replace(',COMMA', ',')
            .replace('.PERIOD', '.')
            .replace('?QUESTIONMARK', '?')
        )
        tokens = []
        labels = []
        for token in tknzr.tokenize(text):
            if token in [',', '.', '?']:
                if ',' in token:
                    labels[-1] = Punctuation.COMMA
                if '.' in token:
                    labels[-1] = Punctuation.PERIOD
                if '?' in token:
                    labels[-1] = Punctuation.QUESTION
            else:
                labels.append(Punctuation.NONE)
                tokens.append(token)

        tokens = np.array(tokens)
        labels = np.array(labels)
        token_len = len(tokens)
        assert len(tokens) == len(labels)

        if self.config.task == Task.TAGGING:
            def apply_window(l):
                return window(
                    l,
                    self.config.task.window_size,
                    int(self.config.task.window_size*self.config.task.window_stride_in_percent)
                )
            ids = apply_window(np.arange(len(tokens)))
            tokens = apply_window(tokens)
            labels = apply_window(labels)
            for i, (ids, tokens, labels) in enumerate(zip(ids, tokens, labels)):
                if self.config.tokenizer is None:
                    raise ValueError('tokenizer argument has to be passed to load_dataset')
                else:
                    tokenized = self.config.tokenizer([tokens.tolist()], is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
                    offsets = np.array(tokenized['offset_mapping'][0])
                    enc_labels = np.array([self.config.label_subword.name]*len(offsets), dtype=object)
                    # todo: check if performance changes if in-word is set to NONE
                    enc_labels[(offsets[:,0] == 0) & (offsets[:,1] != 0)] = [l.name for l in labels]
                    #print(enc_labels)
                    # not needed as long as the same tokenizer is used later?
                    # tokens = {k:v[0] for k,v in tokenized if k != 'offset_mapping'}
                    labels = enc_labels
                yield i, {
                            "ids": ids,
                            "tokens": tokens,
                            "labels": labels,
                        }
            logging.info(f"Loaded number of tokens = {token_len}")