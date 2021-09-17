"""The IWSLT Challenge Dataset, adapted to punctuation as described by Ueffing et al. (2013)"""

from enum import Enum
from typing import Union
from abc import abstractmethod

#import paired
from xml.dom import minidom
import nltk

nltk.download("punkt")
tknzr = TweetTokenizer()

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

class Punctuation(Enum):
    NONE = "<none>"
    PERIOD = "<period>"
    COMMA = "<comma>"
    QUESTION = "<question>"

punct_map = {
    "<period>": '.',
    "<comma>": ',',
    "<question>": '?',
}

class Task(Enum):
    TAGGING = 0
    SEQ2SEQ = 1

class TaggingTask:
    """Treat punctuation prediction as a sequence tagging problem."""

    def __init__(
        self, window_size=120, window_stride_in_percent=0.5, include_reference=False
    ):
        self.lookahead_range = lookahead_range
        self.window_size = window_size
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
        include_punct: list = [
            Punctuation.NONE,
            Punctuation.PERIOD,
            Punctuation.COMMA,
            Punctuation.QUESTION,
        ],
        task: Union[TaggingTask] = TaggingTask(),
        segmented: bool = False,
        asr_or_ref: str = "ref",
        decoder: DecodingStrategy = AverageDecodingStrategy(),
        **kwargs
    ):
        """BuilderConfig for IWSLT2011.
        Args:
          include_punct: the punctuation to be included.
          task: the task to prepare the dataset for.
          segmented: if segmentation present in IWSLT2011 should be respected. removes segmenation by default.
          **kwargs: keyword arguments forwarded to super.
        """
        self.include_punct = include_punct
        self.task = task
        self.segmented = segmented
        self.asr_or_ref = asr_or_ref
        self.decoder = decoder
        super(IWSLT11Config, self).__init__(**kwargs)


class IWSLT11(datasets.GeneratorBasedBuilder):
    """The IWSLT11 Dataset, adapted for punctuation prediction."""

    BUILDER_CONFIGS = [
        IWSLT11Config(name="ref", asr_or_ref="ref"),
        IWSLT11Config(name="asr", asr_or_ref="asr"),
    ]

    def __init__(self, *args, **kwargs):
        super(IWSLT11, self).__init__(*args, **kwargs)
        self.punct = [
            Punctuation.NONE,
            Punctuation.PERIOD,
            Punctuation.COMMA,
            Punctuation.QUESTION,
        ]
        self.include_punct = self.config.include_punct
        self.exclude_punct = [p for p in self.punct if p not in self.include_punct]
        self.cache = Cache('.iswlt11_cache')

    def _info(self):
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

        print(texts)