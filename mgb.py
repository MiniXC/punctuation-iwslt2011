"""The MGB Dataset, adapted for punctuation annotation"""

from __future__ import absolute_import, division, print_function

import logging
from enum import Enum
from typing import Union
from xml.dom import minidom
import re
import itertools
from multiprocessing import Pool

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
@INPROCEEDINGS{Bell2015, \
author={P. {Bell} and M. J. F. {Gales} and T. {Hain} and J. {Kilgour} and P. {Lanchantin} and X. {Liu} and A. {McParland} and S. {Renals} and O. {Saz} and M. {Wester} and P. C. {Woodland}}, \
booktitle={2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU)}, \
title={The MGB challenge: Evaluating multi-genre broadcast media recognition}, \
year={2015}, \
volume={}, \
number={}, \
pages={687-693}, \
doi={10.1109/ASRU.2015.7404863}}
"""

_DESCRIPTION = """\

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


class MGBConfig(datasets.BuilderConfig):
    """The MGB Dataset."""

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
        **kwargs,
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
        super(MGBConfig, self).__init__(**kwargs)


class MGB(datasets.GeneratorBasedBuilder):
    """The MGB Dataset, adapted for punctuation prediction."""

    BUILDER_CONFIGS = [
        MGBConfig(name="ref", asr_or_ref="ref"),
        MGBConfig(
            name="ref-tag",
            asr_or_ref="ref",
            task=TaggingTask()
        ),
        MGBConfig(
            name="ref-pauses",
            asr_or_ref="asr",
        ),
        MGBConfig(
            name="ref-pauses-tag",
            asr_or_ref="asr",
            task=TaggingTask()
        ),
        MGBConfig(
            name="ref-pauses-tf",
            asr_or_ref="asr",
            teacher_forcing=True,
        ),
    ]

    def __init__(self, *args, **kwargs):
        super(MGB, self).__init__(*args, **kwargs)
        self.punct = [
            Punctuation.NONE,
            Punctuation.PERIOD,
            Punctuation.COMMA,
            Punctuation.QUESTION,
            Punctuation.EXCLAMATION,
        ]
        self.include_punct = self.config.include_punct
        self.exclude_punct = [p for p in self.punct if p not in self.include_punct]
        self.cache = Cache('.mgb_cache')
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
            "train_reference": "https://data.cstr.ed.ac.uk/summa/data/english_punctuation/train.txt",
            "validation_reference": "https://data.cstr.ed.ac.uk/summa/data/english_punctuation/dev.txt",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        if self.config.asr_or_ref == "asr":
            urls_to_download = {
                "train_asr": "https://data.cstr.ed.ac.uk/summa/data/english_punctuation/train.ctm",
                "validation_asr": "https://data.cstr.ed.ac.uk/summa/data/english_punctuation/dev.ctm",
            }
            asr_files = dl_manager.download_and_extract(urls_to_download)

            return [
                datasets.SplitGenerator(
                   name=datasets.Split.TRAIN,
                   gen_kwargs={
                       "filepath": downloaded_files["train_reference"],
                       "format": "kaldi",
                       "asr_files": asr_files["train_asr"],
                   },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": downloaded_files["validation_reference"],
                        "format": "kaldi",
                        "asr_files": asr_files["validation_asr"],
                    },
                ),
            ]

        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": downloaded_files["train_reference"],
                        "format": "kaldi-plain",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": downloaded_files["validation_reference"],
                        "format": "kaldi-plain",
                    },
                ),
            ]

    def is_word(self, word):
        return re.search("[a-zA-Z0-9\,\.\?\!]", word)

    def is_punct(self, word):
        return self.get_label(word, all=True) != Punctuation.NONE

    def get_label(self, word, all=False):
        if (word.strip() == "<full_stop>" or word.strip() == "<dots>") and (
            Punctuation.PERIOD in self.include_punct or all
        ):
            return Punctuation.PERIOD
        if (word.strip() == "<comma>") and (
            Punctuation.COMMA in self.include_punct or all
        ):
            return Punctuation.COMMA
        if (word.strip() == "<question_mark>") and (
            Punctuation.QUESTION in self.include_punct or all
        ):
            return Punctuation.QUESTION
        if (word.strip() == "<exclamation_mark>") and (
            Punctuation.EXCLAMATION in self.include_punct or all
        ):
            return Punctuation.EXCLAMATION
        return Punctuation.NONE

    def align_talk(self, talk):
        window = self.config.task.window_size
        min_lookahead = self.config.task.lookahead_range[0]
        max_lookahead = self.config.task.lookahead_range[1]
        results = []
        context_words = []
        context_labels = []
        talks = self.talks
        asr_talks = self.asr_talks
        for text in talks[talk]:
            for word in word_tokenize(text):
                word = word.lower()
                if self.is_word(word):
                    if not self.is_punct(word):
                        context_words.append(word)
                        context_labels.append(Punctuation.NONE)
                    else:
                        try:
                            context_labels[-1] = self.get_label(word)
                        except:
                            return []
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
                    for j, word in enumerate(left_context[:-1]):
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
                    results.append(
                        (
                            i,
                            {
                                "tokens": left_context.split(" "),
                                "label": [l.value for l in left_context_labels],
                            },
                        )
                    )
                if self.config.task == Task._STREAMING_CLASSIFICATION:
                    results.append(
                        (
                            i,
                            {
                                "text": left_context
                                + " <punct> "
                                + " ".join(right_context),
                                "label": left_context_labels[-1].value,
                                "lookahead": la,
                            },
                        )
                    )
            i += 1
        return results

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
            if format == "kaldi":
                with open(path, "r") as f:
                    line = f.readline()
                    while len(line) != 0:
                        seg_id = line.split()[0].split("-")[0]
                        if seg_id not in talks:
                            talks[seg_id] = []
                        talks[seg_id].append(" ".join(line.split()[1:]))
                        line = f.readline()
            if format == "kaldi-plain":
                with open(path, "r") as f:
                    line = f.readline()
                    while len(line) != 0:
                        texts.append([" ".join(line.split()[1:])])
                        line = f.readline()

        texts = list(itertools.chain(*texts))

        if asr_files is not None:
            asr_talks = {}
            if not isinstance(asr_files, list):
                asr_files = [asr_files]
            for asr_file in asr_files:
                with open(asr_file, "r") as f:
                    line = f.readline()
                    prev_ts = None
                    segment = None
                    while len(line) != 0:
                        if len(line.split()) == 5:
                            talkid, _, start_time, duration, word = line.split()
                            if format != "kaldi":
                                talkid = int(talkid[6:])
                            else:
                                diff_segment = segment != talkid.split("-")[1]
                                talkid, segment, ts = talkid.split("-")
                                if prev_ts != ts:
                                    if prev_ts is not None:
                                        start_ts = int(prev_ts.split(':')[1])
                                    else:
                                        start_ts = 0
                                    end_ts = int(ts.split(':')[1])
                                prev_ts = ts
                            if talkid not in asr_talks:
                                asr_talks[talkid] = {
                                    "words": [],
                                    "pauses": [abs(end_ts - start_ts)/1000],
                                }
                            else:
                                asr_talks[talkid]["pauses"].append(
                                    float(start_time) - float(prev_end_time)
                                )
                            if asr_talks[talkid]["pauses"][-1] >= self.pause_threshold:
                                asr_talks[talkid]["words"].append("<pause>")
                            prev_end_time = float(start_time) + float(duration)
                            asr_talks[talkid]["words"].append(word.lower())
                        line = f.readline()

        np.random.seed(42)
        window = self.config.task.window_size
        min_lookahead = self.config.task.lookahead_range[0]
        max_lookahead = self.config.task.lookahead_range[1]
        i = 0

        if format == "plain" or format == "kaldi-plain":
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
                                "label": context_labels[
                                    -(max_lookahead + 1)
                                ].value,
                                "lookahead": la,
                            }
                    i += 1

        if format == "xml" or format == "kaldi" and asr_talks is not None:
            overlap_talks = [t for t in talks.keys() if t in asr_talks.keys()]

            def overlap_generator(talks, asr_talks):
                for t in talks.keys():
                    if t in asr_talks.keys():
                        yield t

            self.talks = talks
            self.asr_talks = asr_talks
            pool = Pool(8)
            for result in tqdm(
                pool.imap_unordered(self.align_talk, overlap_talks, chunksize=8),
                total=len(overlap_talks),
            ):
            #for result in tqdm([self.align_talk(t) for t in overlap_talks], total=len(overlap_talks)):
                for item in result:
                    yield item
            pool.close()
            pool.join()
