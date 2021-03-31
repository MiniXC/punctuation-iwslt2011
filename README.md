# punctuation-iwslt2011
Huggingface datasets script for pre-processing punctuation annotation using IWSLT11 dataset. For ASR transcripts, punctuation marks are inferred using Levenshtein alignment as described by [Ueffing et al. (2013)](#1).

### Without Pause Information

```python
from datasets import load_dataset
ds = load_dataset('iwslt2011.py', 'ref')
```

### With Pause Information

```python
from datasets import load_dataset
ds = load_dataset('iwslt2011.py', 'ref-pauses')
```

## References
<a id="1">[1]</a> 
B.  Ueffing,  M.  Bisani,  and  P.  Vozila.  Improved  models  for  automatic  punctuation prediction for spoken and written text. In INTERSPEECH, 2013.