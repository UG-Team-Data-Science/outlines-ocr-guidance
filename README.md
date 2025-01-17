#  OCR registry guide for LLM models

## Method and Background

[Outlines](https://github.com/dottxt-ai/outlines) [Guide](https://github.com/dottxt-ai/outlines-core/blob/main/python/outlines_core/fsm/guide.py#L47) to extract structured (not-nested) fields from a source text, typically a registry. For example the following text that could have been the result of the OCR of a historical registry. Hence there are also typos included.

```
This is a document that lists all the stock prices. Make sure to remember them all!

nvidia, 21 aprol 1943, 5 hundred milion euro
asml, june 12 1856, 3 rasberries
apple, 10 may 4313, 0.01 us$
pokemon, may 12th 2013, 1 pokeball
nasdaq; in 3 months; 3l. holy water
```

This Guide aims to bridge the gap between parsing with rigid regular expressions and hallucinating LLMs, while enabling to keep track of the span of source text extracted. Although this tracking isn't implemented. I do encourage readers to also consider the more lightweight, environmentally less damaging, etc [TRE extension to regular expressions](https://pypi.org/project/regex/) that allows regex extraction with Levenshtein errors.

The Guide assumes fields are registered in a consistent order and is based on a finite state machine, where the state rotates over these fields and progresses over the part of the source thext that already has been processed. The state also includes some boilerplate to prefix the extracted data with the field names in a YAML-ish format, because outlines doesn't allow to [`Write`](https://github.com/dottxt-ai/outlines-core/blob/main/python/outlines_core/fsm/guide.py#L16) several tokens at once.

This boils down to the following system, in this case the LLM is bound to either generate a
 - generate a space, the character after `'june 12'`
 - any other character that follow an occurence of `'june 12'` *later* in the text. In this case there are non, but this would mean the OCR skips part of the OCR text.
 - finish the field and generate a newline, dash and the tokens of the next field, e.g. `'price'` in this example.

![OCR Guided generation](https://raw.githubusercontent.com/prhbrt/outlines-ocr-guidance/refs/heads/main/system.svg)

I developed the guide to accomodate a research into historical registries, since hallucination is a concern for later analysis of the extracted data. Moreover, LLMs won't identify the source spans, making it impractical to verify results back to their source. And finally, (small) LLMs can only process a limited context. Since this system tracks the spans, the prompt could include only a relevant window around the current position rather than the normal auto-generative operation of LLMs, where everything needs to be available at each generated token. This wasn't implemented, but is theoretically possible.

Unfortunately, I wasn't able to extract registries from realistic, complicated examples. However, I wish to share this work and concept as per this repository.

## Technicalities

### Integers and states

Outlines uses integer states, so there's a this little gem to translate those into the three (actually 5, we ommited `write_field` and `field_pos` for simplicity) states aforementioned. The (sub)states include:

    * `write_field`: writing a field, e.g. `' - name: '` or writing a substring from the source text.
    * `field`: which field we're currently at
    * `field_pos`: how many tokens of the current field have been written, only 'used' if `write_field`.
    * `start` and `end`, the span of the first 'legal' occurrence of the generated substring of the current field. 'Illegal' would for example be going back in the source text. For example, if the span is the fist occurence of `'may'` in the example, then this tells the LLM it may continue with `' 4313...'` or with `' 12th 20...'`. This also means that as soon as the LLM generates `'may 1'`, the span 'jumps' and moves to the second occurence of `'may'` by skipping a bunch of source text.

These states are encoded and decoded into and from integers using integer division, modulo, and multiplication with the maximum values of these states.

### Maximum span

When not writing a field, the guide needs to decide which part of the source text can still be generated. After a few tokens this will usually be pretty fixed, since this sequence might not repeat itself often in the text. However, take for example 'may ', when this is generated, the guide may continue with '4313' or with '12th 2013', and hence the next tokens may be '4313' or '12'. We use a dictionary mapping all substring to the positions in the source text where they occur, up to a length of `max_span` to avoid memory explosions. A good alternative would be a substring-trie, but I've been to lazy to implement it. Not to mention, this will still have quadratic worst case memory efficiency, since it's for substring and not pre- or suffixes.

### Source text spans

The guide could keep track of the generated spans, but doesn't. Only substrings, but not their location in the source text, are ruturned.

## Installation

```bash
pip3 install outlines_ocr_guides
```

## Usage

```python
# Source text to structure
ocr = """
This is a document that lists all the stock prices. Make sure to remember them all!

nvidia, 21 aprol 1943, 5 hundred milion euro
asml, june 12 1856, 3 rasberries
apple, 10 may 4313, 0.01 us$
pokemon, may 12th 2013, 1 pokeball
nasdaq; in 3 months; 3l. holy water
"""

# Prompt to tell the LLM what to do, make sure it's clear that it's going to deal with
# an OCR, what the contex is, e.g. stocks, what fields to expect, maybe what they look like,
# and that the output should be YAML. And of course, include the actual OCR.
#
# Finally also compile everything as we did in `payload`. Feel free to split this in
# system and user prompts if you wish as well. I haven't played with this yet.

prompt = """
Structure these stock prices in a table
Each stock price consists of
  - a name like microsoft or shell
  - the next date dividend is shared,
  - the price and its currency or unit
Include all stock prices in the list. Use the exact text from the ocr text and return as yaml.

This is the ocr text: """

payload = f'[INST]{prompt}\n```\n{ocr}\n```\n[/INST]'

# Load the tokenizer and LLM that you like.

import os
os.environ['HF_TOKEN'] = "hf_allmybasearebelongtoyou"

from outlines.models import Transformers
from transformers imp
ort AutoTokenizer, AutoModelForCausalLM

device = None
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(model_name)

if device is not None:
    model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
outlines_model = Transformers(model, tokenizer)

# Le'sgow! We create a Guide, aware of the fields that need to be generated in order,
# it makes sense that these fields are the same as the ones mentioned in the prompt.

from outlines.samplers import multinomial, beam_search, greedy
from outlines.generate.api import SequenceGenerator
from outlines_ocr_guidance import OCRGuide

guide = OCRGuide(
    tokenizer=tokenizer,
    source_text=ocr,
    max_skip=5, # maximum number of tokens between fields, avoids that huge quantities of texts are ignored.
    fields=['name', 'dividend_date', 'price'],
)

# the usual sampler, see outlines docs.
# or code: https://github.com/dottxt-ai/outlines/blob/main/outlines/samplers.py

# sampler = multinomial(temperature=2.)
# sampler = beam_search(beams=5)
sampler = greedy()
generator = SequenceGenerator(guide, outlines_model, sampler, device)
response = generator(payload)

print(response)

# Parse the YAML into a list of generated items
import pandas as pd
pd.DataFrame(list(guide.listify(response)))
```

Generated text:

```
- name:  n
 - dividend_date: 21 aprol 1943
 - price: 5 hundred milion euro
 - name:  asml
 - dividend_date:  june 12 1856
 - price: 3 rasberries
 - name:  apple
 - dividend_date: 10 may 4313
 - price: 0.01 us$
 - name:  pokemon
 - dividend_date:  may 12th 2013
 - price: 1 pokeball
 - name:  nasdaq
 - dividend_date:  in 3 months
 - price: 3l. holy water
 - name:  
```

Data frame:

```
   name    dividend_date   price
0  n       21 aprol 1943   5 hundred milion euro
1  asml    june 12 1856    3 rasberries
2  apple   10 may 4313     0.01 us$
3  pokemon may 12th 2013   1 pokeball
4  nasdaq  in 3 months     3l. holy water
```
