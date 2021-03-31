# punctuation-iwslt2011
Huggingface datasets script for loading pre-processing punctuation annotation using IWSLT11 dataset reference transcriptions ([IWSLT-2011](#1)).
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
International Workshop on Spoken Language Translation (IWSLT) 2011, San Francisco, CA, USA, December 8-9, 2011; ISCA Archive, http://www.isca-speech.org/archive/iwslt_11
