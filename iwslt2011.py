"""The IWSLT Challenge Dataset, adapted to punctuation as described by Ueffing et al. (2013)"""

from __future__ import absolute_import, division, print_function

import logging
from enum import Enum
from typing import Union
from xml.dom import minidom
import re
import itertools

import datasets
from collections import deque
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from tqdm.auto import tqdm
import paired
from diskcache import Cache

nltk.download("punkt")
tknzr = TweetTokenizer()

def align(seq_1, seq_2):
    seqA = []
    seqB = []
    for i, j in paired.align(seq_1, seq_2):
        if i is not None:
            seqA.append(seq_1[i])
        else:
            seqA.append('<gap>')
        if j is not None:
            seqB.append(seq_2[j])
        else:
            seqB.append('<gap>')
    return seqA, seqB

def word_tokenize(text):
    toks = tknzr.tokenize(text)
    all_toks = []
    for tok in toks:
        if "-" in tok:
            split_tok = tok.split("-")
            all_toks.append(split_tok[0])
            all_toks.append(split_tok[1])
        else:
            all_toks.append(tok)
    return all_toks


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
If desired, acoustic pauses can be included in the text output as well.
"""

_VERSION = "0.0.1"


class Punctuation(Enum):
    NONE = "<none>"
    PERIOD = "<period>"
    COMMA = "<comma>"
    QUESTION = "<question>"
    EXCLAMATION = "<exclamation>"

punct_map = {
    "<period>": '.',
    "<comma>": ',',
    "<question>": '?',
    "<exclamation>": '!',
}

class Task(Enum):
    TAGGING = 0
    SEQ2SEQ = 1
    _STREAMING_CLASSIFICATION = 2  # for internal use only


class StreamingClassificationTask:
    """Special task in which only predicts one possible\
    punctuation token at a time, which can be useful for streaming ASR outputs.
    """

    def __init__(
        self, lookahead_range=(0, 4), window_size=128, include_reference=False
    ):
        self.lookahead_range = lookahead_range
        self.window_size = window_size
        self.include_reference = include_reference

    def __eq__(self, other):
        return Task._STREAMING_CLASSIFICATION == other
    
class TaggingTask:
    """Special task in which only predicts one possible\
    punctuation token at a time, which can be useful for streaming ASR outputs.
    """

    def __init__(
        self, lookahead_range=(0, 4), window_size=128, include_reference=False
    ):
        self.lookahead_range = lookahead_range
        self.window_size = window_size
        self.include_reference = include_reference

    def __eq__(self, other):
        return Task.TAGGING == other


class IWSLT11Config(datasets.BuilderConfig):
    """The IWSLT11 Dataset."""

    def __init__(
        self,
        include_punct: list = [
            Punctuation.NONE,
            Punctuation.PERIOD,
            Punctuation.COMMA,
            Punctuation.QUESTION,
        ],
        task: Union[Task, StreamingClassificationTask] = StreamingClassificationTask(),
        segmented: bool = False,
        asr_or_ref: str = "ref",
        teacher_forcing: bool = False,
        **kwargs
    ):
        """BuilderConfig for IWSLT2011.
        Args:
          include_punct: the punctuation to be included. by default, only commas are excluded.
          task: the task to prepare the dataset for.
          segmented: if segmentation present in IWSLT2011 should be respected. removes segmenation by default.
          **kwargs: keyword arguments forwarded to super.
        """
        self.include_punct = include_punct
        self.task = task
        self.segmented = segmented
        self.asr_or_ref = asr_or_ref
        self.teacher_forcing = teacher_forcing
        super(IWSLT11Config, self).__init__(**kwargs)


class IWSLT11(datasets.GeneratorBasedBuilder):
    """The IWSLT11 Dataset, adapted for punctuation prediction."""

    BUILDER_CONFIGS = [
        IWSLT11Config(name="ref", asr_or_ref="ref"),
        IWSLT11Config(
            name="ref-pauses",
            asr_or_ref="asr",
        ),
        IWSLT11Config(
            name="ref-pauses-tag",
            asr_or_ref="asr",
            task=TaggingTask()
        ),
        IWSLT11Config(
            name="ref-pauses-tf",
            asr_or_ref="asr",
            teacher_forcing=True,
        ),
    ]

    def __init__(self, *args, **kwargs):
        super(IWSLT11, self).__init__(*args, **kwargs)
        self.punct = [
            Punctuation.NONE,
            Punctuation.PERIOD,
            Punctuation.COMMA,
            Punctuation.QUESTION,
            Punctuation.EXCLAMATION,
        ]
        self.include_punct = self.config.include_punct
        self.exclude_punct = [p for p in self.punct if p not in self.include_punct]
        self.cache = Cache('.iswlt11_cache')
        if 'lookahead' in kwargs:
            self.lookahead = kwargs['lookahead']
        if 'pause_threshold' in kwargs:
            self.pause_threshold = kwargs['pause_threshold']
        else:
            self.pause_threshold = 0.2

    def _info(self):
        if self.config.task == Task._STREAMING_CLASSIFICATION:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "text": datasets.Value("string"),
                        "label": datasets.features.ClassLabel(
                            names=[p.value for p in self.config.include_punct]
                        ),
                        "lookahead": datasets.Value("int32"),
                    }
                ),
                supervised_keys=None,
                homepage="http://iwslt2011.org/doku.php",
                citation=_CITATION,
                version=_VERSION,
            )
        if self.config.task == Task.TAGGING:
            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "tokens": datasets.Sequence(datasets.Value("string")),
                        "label": datasets.Sequence(
                            datasets.features.ClassLabel(
                                names=[p.value for p in self.config.include_punct]
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
            "train_reference": "https://web.archive.org/web/20140203142509/http://hltshare.fbk.eu/IWSLT2011/monolingual-TED.tgz",
            "validation_reference": "https://web.archive.org/web/20170725181407/http://hltshare.fbk.eu/IWSLT2011/devsets.tgz",
            "test_reference": "https://web.archive.org/web/20170725181407/http://hltshare.fbk.eu/IWSLT2011/IWSLT11.MT.tst2011.tgz",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        downloaded_files["train_reference"] += "/monolingual-TED/IWSLT11.TALK.train.en"
        valid = downloaded_files["validation_reference"]
        downloaded_files["validation_reference"] = [
            valid + "/devsets/IWSLT11.TALK.tst2010.en-fr.en.xml",
            valid + "/devsets/IWSLT11.TALK.dev2010.en-fr.en.xml",
        ]
        downloaded_files[
            "test_reference"
        ] += "/IWSLT11.MT.tst2011/IWSLT11.TALK.tst2011.en-fr.en.xml"

        if self.config.asr_or_ref == "asr":
            urls_to_download = {
                "test_asr": "https://web.archive.org/web/20170725181407/http://hltshare.fbk.eu/IWSLT2011/IWSLT11.SLT.tst2011_2nd.tgz",
                "validation_asr_1": "https://web.archive.org/web/20170725181407/http://hltshare.fbk.eu/IWSLT2011/IWSLT11.SLT.tst2010.1best.v1.tgz",
                "validation_asr_2": "https://web.archive.org/web/20170725181407/http://hltshare.fbk.eu/IWSLT2011/IWSLT11.SLT.dev2010.1best.v1.tgz",
            }
            asr_files = dl_manager.download_and_extract(urls_to_download)
            asr_files[
                "test_asr"
            ] += "/IWSLT11.SLT.tst2011_2nd/tst2011_2nd.ASR_E.rover4021.ctm"
            asr_files[
                "validation_asr_1"
            ] += "/IWSLT11.SLT.tst2010.1best.v1/ted.tst2010.en-fr.en.ctm"
            asr_files[
                "validation_asr_2"
            ] += "/IWSLT11.SLT.dev2010.1best.v1/ted.dev2010.en-fr.en.ctm"

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": downloaded_files["validation_reference"],
                        "format": "xml",
                        "asr_files": [
                            asr_files["validation_asr_1"],
                            asr_files["validation_asr_2"],
                        ],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": downloaded_files["test_reference"],
                        "format": "xml",
                        "asr_files": [asr_files["test_asr"]],
                    },
                ),
            ]

        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": downloaded_files["train_reference"],
                        "format": "plain",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": downloaded_files["validation_reference"],
                        "format": "xml",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": downloaded_files["test_reference"],
                        "format": "xml",
                    },
                ),
            ]

    def is_word(self, word):
        return re.search("[a-zA-Z0-9\,\.\?\!]", word)

    def is_punct(self, word):
        return self.get_label(word) != Punctuation.NONE

    def get_label(self, word):
        if word.strip() == "." and Punctuation.PERIOD in self.include_punct:
            return Punctuation.PERIOD
        if word.strip() == "," and Punctuation.COMMA in self.include_punct:
            return Punctuation.COMMA
        if word.strip() == "?" and Punctuation.QUESTION in self.include_punct:
            return Punctuation.QUESTION
        if word.strip() == "!" and Punctuation.EXCLAMATION in self.include_punct:
            return Punctuation.EXCLAMATION
        return Punctuation.NONE

    def _generate_examples(self, filepath, format, asr_files=None):
        logging.info("⏳ Generating examples from = %s", filepath)
        asr_talks = None
        texts = []
        talks = {}
        if not isinstance(filepath, list):
            filepath = [filepath]
        for path in filepath:
            if format == "plain":
                texts.append([line.strip() for line in open(path).readlines()])
            if format == "xml":
                fin = open(path, "rt")
                fout = open(path + "_fixed", "wt")
                for line in fin:
                    fout.write(line.replace("&", "&amp;"))
                fout.close()
                fin.close()
                doc = minidom.parse(path + "_fixed")
                docs = doc.getElementsByTagName("doc")
                for doc in docs:
                    docid = doc.getAttribute("docid")
                    docid = int(docid)
                    if docid not in talks:
                        talks[docid] = []
                    talks[docid] = [
                        seg.childNodes[0].wholeText.strip()
                        for seg in doc.getElementsByTagName("seg")
                    ]

        texts = list(itertools.chain(*texts))

        if asr_files is not None:
            asr_talks = {}
            if not isinstance(asr_files, list):
                asr_files = [asr_files]
            for asr_file in asr_files:
                with open(asr_file, "r") as f:
                    line = f.readline()
                    prev_talkid = None
                    while len(line) != 0:
                        if len(line.split()) == 5:
                            talkid, _, start_time, duration, word = line.split()
                            talkid = int(talkid[6:])
                            if talkid not in asr_talks:
                                asr_talks[talkid] = {
                                    "words": [],
                                    "pauses": [1],
                                }
                            else:
                                asr_talks[talkid]["pauses"].append(
                                    float(start_time) - float(prev_end_time)
                                )
                            if asr_talks[talkid]["pauses"][-1] >= self.pause_threshold:
                                asr_talks[talkid]["words"].append("<pause>")
                            prev_end_time = float(start_time) + float(duration)
                            asr_talks[talkid]["words"].append(word)
                        line = f.readline()

        np.random.seed(42)
        window = self.config.task.window_size
        min_lookahead = self.config.task.lookahead_range[0]
        max_lookahead = self.config.task.lookahead_range[1]
        i = 0

        if format == "xml" and asr_talks is None:
            texts = list(itertools.chain(*[v for k,v in talks.items()]))
            format = "plain"
        if format == "plain":
            context_words = []
            context_labels = []
            for text in tqdm(texts):
                for word in word_tokenize(text):
                    word = word.lower()
                    if self.is_word(word):
                        if not self.is_punct(word):
                            context_words.append(word)
                            context_labels.append(Punctuation.NONE)
                        else:
                            if len(context_labels) > 0:
                                context_labels[-1] = self.get_label(word)
                    context_words = context_words[-(window + max_lookahead) :]
                    context_labels = context_labels[-(window + max_lookahead) :]
                    la = np.random.randint(min_lookahead, max_lookahead + 1)
                    if len(context_words) >= window:
                        if self.config.task == Task.TAGGING:
                            yield i, {
                                        "tokens": context_words[la:-max_lookahead],
                                        "label": [
                                            l.value for l in context_labels[la:-max_lookahead]
                                        ],
                                    }
                        if self.config.task == Task._STREAMING_CLASSIFICATION:
                            yield i, {
                                "text": " ".join(context_words[la:-max_lookahead])
                                + " <punct> "
                                + " ".join(context_words[-max_lookahead:][:la]),
                                "label": context_labels[-(max_lookahead + 1)].value,
                                "lookahead": la,
                            }
                    i += 1
        if format == "xml" and asr_talks is not None:
            overlap_talks = [t for t in talks.keys() if t in asr_talks.keys()]
            for talk in overlap_talks:
                context_words = []
                context_labels = []
                for text in tqdm(talks[talk]):
                    for word in word_tokenize(text):
                        word = word.lower()
                        if self.is_word(word):
                            if not self.is_punct(word):
                                context_words.append(word)
                                context_labels.append(Punctuation.NONE)
                            else:
                                context_labels[-1] = self.get_label(word)
                print("computing alignment...")
                seqHash = hash(" ".join(context_words)) + hash(" ".join(asr_talks[talk]["words"]))
                if seqHash not in self.cache:
                    self.cache[seqHash] = align(context_words, asr_talks[talk]["words"])
                seqA, seqB = self.cache[seqHash]
                lbl_i = 0
                new_words = []
                new_labels = []
                for i, word in enumerate(seqA):
                    if word == "<gap>" and seqB[i] == "<pause>":
                        new_words.append("<pause>")
                        new_labels.append(Punctuation.NONE)
                    if word != "<gap>":
                        new_words.append(word)
                        new_labels.append(context_labels[lbl_i])
                        lbl_i += 1
                new_context_words = []
                new_context_labels = []
                i = 0
                for word, label in zip(new_words, new_labels):
                    new_context_words.append(word)
                    new_context_labels.append(label)
                    new_context_words = new_context_words[-(window + max_lookahead) :]
                    new_context_labels = new_context_labels[-(window + max_lookahead) :]
                    la = np.random.randint(min_lookahead, max_lookahead + 1)
                    if len(new_context_words) >= window:
                        right_context = new_context_words[-max_lookahead:][:la]
                        left_context = new_context_words[la:-(max_lookahead)]
                        left_context_labels = new_context_labels[la:-(max_lookahead)]
                        num_right_pauses = None
                        while num_right_pauses != len(
                            [p for p in right_context if p == "<pause>"]
                        ):
                            num_right_pauses = len([p for p in right_context if p == "<pause>"])
                            left_context = new_context_words[
                                la : -(max_lookahead + num_right_pauses)
                            ]
                            left_context_labels = new_context_labels[
                                la : -(max_lookahead + num_right_pauses)
                            ]
                            if left_context[-1] == "<pause>":
                                num_right_pauses += 1
                            left_context = new_context_words[
                                la : -(max_lookahead + num_right_pauses)
                            ]
                            left_context_labels = new_context_labels[
                                la : -(max_lookahead + num_right_pauses)
                            ]
                            right_context = new_context_words[
                                -(max_lookahead + num_right_pauses) :
                            ][: (la + num_right_pauses)]
                        if self.config.teacher_forcing:
                            left_context_new = []
                            for j, word in enumerate(left_context):
                                left_context_new.append(word)
                                if j < len(left_context) - 1 and left_context_labels[j] != Punctuation.NONE:
                                    left_context_new.append(punct_map[left_context_labels[j].value])
                            left_context = ""
                            for w in left_context_new:
                                if w not in punct_map.values():
                                    left_context += " "
                                left_context += w
                            left_context = left_context.strip()
                        else:
                            left_context = " ".join(left_context)
                        if self.config.task == Task.TAGGING:
                            yield i, {
                                        "tokens": left_context.split(" "),
                                        "label": [l.value for l in left_context_labels],
                                    }
                        if self.config.task == Task._STREAMING_CLASSIFICATION:
                            yield i, {
                                        "text": left_context
                                        + " <punct> "
                                        + " ".join(right_context),
                                        "label": left_context_labels[-1].value,
                                        "lookahead": la,
                                    }
                    i += 1
