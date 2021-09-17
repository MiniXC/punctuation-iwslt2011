""" IWSLT11 windowed punctuation metrics. """

import datasets


_CITATION = """TODO add citation
"""

_DESCRIPTION = """TODO add description
"""

_KWARGS_DESCRIPTION = """
Computes XNLI score which is just simple accuracy.
Args:
    predictions: Predicted labels.
    references: Ground truth labels.
Returns:
    'accuracy': accuracy
Examples:

    >>> predictions = [0, 1]
    >>> references = [0, 1]
    >>> xnli_metric = datasets.load_metric("xnli")
    >>> results = xnli_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0}
"""


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class IWSLT2011(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64" if self.config_name != "sts-b" else "float32"),
                    "references": datasets.Value("int64" if self.config_name != "sts-b" else "float32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        return {"accuracy": simple_accuracy(predictions, references)}
