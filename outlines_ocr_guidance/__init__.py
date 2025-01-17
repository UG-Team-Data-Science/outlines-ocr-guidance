import re
from collections import defaultdict
from outlines.fsm.guide import Guide, Generate, Instruction


class OCRGuide(Guide):
  """Class to guide the LLM for generation of YAML-structured
  fields from a source text. This could look like:

  >>> - name: nvidia
  >>> - date: June 12th, 2034
  >>> - value: 12 blueberries
  >>> - name: ASML
  >>> - date: June 11th, 1846
  >>> - value: 7 souls

  Attributes:
    self.tokens List[Token]: tokenized `source_text`
    self.suffix_positions Dict[Tuple[Token], Set[int]]: the mapping
    self.fields List[str]: fields to generate
    self.field_tokens List[List[Token]]: tokenized variants of `self.fields` in the form `f" - {field}: "`
    self.suffix_positions Dict[Tuple[Token], Set[int]]: mapping token spans to all their positions
    self.max_spans int: max length of a generated span, in tokens
    self.max_field_len int: max length of `self.field_tokens`, used when translating states
    self.newline_token Token: token to end a field and move to the next, i.e. the newline
  """
  # TODO: investigate alternatives using a suffix trie representation.
  # This doesn't seem to be a speed bottlenck though, maybe is a
  # memory bottleneck, being (sort-off) quatratic and all.
  def _token_span_positions(self, tokenizer, source_text):
    """Internal method to initialize `self.suffix_positions`, a
    dictionary that maps every substring of size smaller than
    `max_span` to a set of source_text positions, counted in
    tokens, not characters, where this substring can be found.
    tokenizer.

    As a biproduct, also intializes `self.tokens`, the tokenized `source_text`.

    Initialized attributes:
    self.tokens List[Token]: tokenized `source_text`
    self.suffix_positions Dict[Tuple[Token], Set[int]]: the mapping
    """
    self.tokens = tokenizer(source_text.replace('\n', ' ')).input_ids[1:]
    # +1 for open-ended encoding, i.e. [start, end).
    self.max_tokens = 1 + len(self.tokens)
    self.suffix_positions = defaultdict(set)

    for a in range(len(self.tokens)):
      for b in range(a+1, min(a+self.max_span+1, len(self.tokens))):
        self.suffix_positions[tuple(self.tokens[a:b])].add(a)

  def __init__(self, **kwargs):
    """Guide to extract spans from an source text, usually an OCR, into a
    structured YAML of ordered `fields`.

    Parameters:
    source_text str: text from which fields should be extracted, in order
    fields List[str]: fields to extract from the source text
    tokenizer Tokenizer: transformers Tokenizer used by the LLM, this is used
                         to tokenize the fields and source_text, and hence
                        guide the LLM to produce the accurate tokens.
    max_span int: maximum size of a span that can be extracted, defaults to 50
    max_skip int: maximum tokens to skip between fields in one record, defaults to None
    """
    self.max_span = kwargs.pop('max_span', 50)
    self.initial_state = 1

    if len({'fields', 'source_text', 'tokenizer'} - set(kwargs)) == 0:
      self.fields = kwargs['fields']
      self.tokenizer = kwargs['tokenizer']
      self.field_tokens = [self.tokenizer(f' - {field}: ').input_ids[1:]
                           for field in self.fields]
      self.max_field_len = max(map(len, self.field_tokens)) + 2

      self._token_span_positions(self.tokenizer, kwargs['source_text'])

      self.newline_token = next(token for token in self.tokenizer('\n').input_ids if self.tokenizer.decode(token) == '\n')
      self.max_skip = kwargs.pop("max_skip", len(self.tokens))
      self.final_state = (1 + 2 * (
          (len(self.fields)-1) + len(self.fields) * (
            (self.max_field_len-1) + self.max_field_len * (
              (self.max_tokens-1) + self.max_tokens * (
                (self.max_tokens-1) + (
                    (self.max_tokens-1) * self.max_tokens
                )
              )
            )
          )
        )
      ) + 1
      # last feasible token to generate, e.g. right trip newlines and other isth.
      ignore_tokens = {1, 28705, 13}
      self.max_generate = max(
        i for i, token in enumerate(self.tokens)
        if token not in ignore_tokens
      )
      #TODO: empty list for generated spans

  def _state_to_int(self, write_field, field, field_pos, start, end, previous_end):
    """Internal function to translate a state with several variables to one
    integer, as required by outlines. Inverse of `self._int_to_state`

    Parameters:
    write_field bool: whether the field prefix is written (True), or a substring (False)
    field int: which field currently is being processed
    field_pos int: how many tokens of `f' - {field name}'` have been written, only relevant if `write_field`.
    start int: first position in `self.tokens` from where text can be generated
    end int: `self.tokens[start:end]` determines the currently generated span of tokens from the source text.
             if `self.tokens[start:end]` also occurs after `start`, then the token following those occurences
             can also be generated, if so the span moves forward to the first span that would be consistent
             with that generation.

    Returns:
    int: integer that uniquely identifies the parameters above.
    """
    assert 0 <= field < len(self.fields)
    assert  0 <= field_pos < self.max_field_len
    assert start >= 0
    assert start <= end < self.max_tokens
    assert 0 <= previous_end < self.max_tokens
    #TODO: reset generated spans to empty list if state equals 0,
    # not the cleanest way to do deal with copied guides though.
    return (
      int(not write_field) + 2 * (
        field + len(self.fields) * (
          field_pos + self.max_field_len * (
            start + self.max_tokens * (
              end + (
                  self.max_tokens * previous_end
              )
            )
          )
        )
      )
    )

  def _int_to_state(self, state):
    """Internal function to translate an integer state used by outlines to several
    substates, i.e. write_field, field, field_pos, start and end. See `self._state_to_int`
    for their meaning. Inverse of `self._state_to_int`
    Parameters:
    int: state that uniquely maps to the 5 mentioned substates
    Return:
    Tuple[bool, int, int, int, int]: the 5 mentioned substates.
    """
    assert state < self.final_state, "Can't translate final state to substates"
    write_field = not bool(state % 2)
    field = (state := state // 2) % len(self.fields)
    field_pos = (state := state // len(self.fields)) % self.max_field_len
    start = (state := state // self.max_field_len) % self.max_tokens
    end = (state := state // self.max_tokens) % self.max_tokens
    previous_end = state // self.max_tokens
    return write_field, field, field_pos, start, end, previous_end

  def get_next_instruction(self, state: int) -> Instruction:
    """ Maps a state to an instruction, which is always a `Generate`, such that a YAML output
    of all the extracted fields is generated. This translates to the following transitions:

    - start in the state where `write_field, field, field_pos, start, end == True, 0, 0, 0, 0`
    - if `write_field`:
        - Generate the token at `tokens_of(f" - {field[field]}: ")[field_pos]`
        - If field_pos doesn't overflow those tokens, increase it.
          Otherwise reset to 0 and set `write_field` to False.
    - else if start == end, nothing from `source_text` has been generated for that field yet, so:
        - Generate any token of `self.tokens[start:]`
    - else, some tokens are already generated, i.e. `self.tokens[start:end]`, so either:
        - generate a token from the source text by:
          - Extract all the `source_text` positions from `self.suffix_positions[self.tokens[start:end]]`
          - Add `end-start` to all these positions p, and generate any of the tokens at self.tokens[p]
          - Adjust `start` and `end` such that the newly generated tokens are represented, i.e. given
          `tokens_generated = (*self.tokens[start:end], newly_generated_token)`, then `start` becomes
          the first next occurence of those tokens, i.e. `min(self.suffix_positions[tokens_generated]`,
          and `end` becomes `start + len(tokens_generated)`.
        - generate a `self.new_line_token`, finishing the current field and moving to the next field.
          - `start` becomes `end` and `write_field` becomes True and `field` increases by 1 or turns
            0 again if all fields are written, to indicate a new record can be generated. `field_pos`
            should still be 0.

    """
    write_field, field, field_pos, start, end, previous_end = self._int_to_state(state)

    # Write `f" - {field}: "` token for token.
    if write_field:
      return Generate([self.field_tokens[field][field_pos]])

    # If nothing for the current field has been generated, generate anything that's in the unprocessed
    # part of `source_text`, i.e. `self.tokens[start:]`.
    if start == end:
      return Generate(list(set(self.tokens[start:])))

    # A field can be finished with a newline, or if this was the last field of
    # a record also with an end of sequence token.
    finish_field_tokens = (
        [self.newline_token, self.tokenizer.eos_token_id]
        if field == len(self.fields) - 1 else [self.newline_token])

    # If the maximum span size is exceeded what's recorded in `self.suffix_positions`, force to
    # move to the next field or stop generating all together if this was also the last field.
    # This may fix the try-catch in `get_next_state`.
    if end - start >= self.max_span - 1:
      return Generate(finish_field_tokens)


    # If something has been generated, extract all token positions in the unprocessed
    # `source_text` where this substring occurs. All tokens that exceed this occurences
    # can be generated.
    possible_succeeding_tokens = list(set(
        self.tokens[p + end - start]
        for p in self.suffix_positions[tuple(self.tokens[start:end])]
        if p >= start and p < previous_end + self.max_skip
    ))

    # Generate one of these tokens, or a newline to indicate the next field (or record) should
    # be generated.
    return Generate(list(set(finish_field_tokens + possible_succeeding_tokens)))

  def get_next_state(self, state: int, token_id: int) -> int:
    """ Gets the next state based on the generated token. See `self.get_next_instruction` for
    a summary of these state transitions.
    """
    write_field, field, field_pos, start, end, previous_end = self._int_to_state(state)
    
    # If the LLM decides to stop generating, e.g. if it finds that all
    # entries have been generated.
    if token_id == self.tokenizer.eos_token_id:
      #TODO: flush to generated spans
      return self.final_state

    # If the source_text is finished, don't continue.
    if end >= len(self.tokens):
      #TODO: flush to generated spans
      return self.final_state

    # If a newline token was generated, the current field has finished and we move
    # to the next one.
    if token_id == self.newline_token:
      #TODO: flush to generated spans
      write_field = True
      field = (field + 1) % len(self.fields)
      field_pos = 0# superfluous, also happens when field is written
      previous_end = end
      start = end
    else:
      # If `write_field`, we either write the next token of `f" - {field}: "` by increasing
      # field_pos, or when finished we set `write_field` to False to indicate the start
      # of generation from `source_text` for that field.
      if write_field:
        if field_pos + 1 >= len(self.field_tokens[field]):
          write_field = False
          field_pos=0 # superfluous, also happens at newline
        else:
          field_pos += 1
      else:
        # This branch determines the new state while still generating from `source_text`
        # We first determine all generated tokens so far.
        tokens_generated = (*self.tokens[start:end], token_id)

        # Then we determine the first position in the unprocessed `source_text` of
        # `tokens_generated`, and ajust `start` and `end` to mark that span.
        # This may fail if (apparantly), either if the generated text is longer than `max`
        start = next((
            p for p in sorted(self.suffix_positions[tokens_generated])
            if p >= start), start)
        end = start + len(tokens_generated)
    return self._state_to_int(write_field, field, field_pos,
                              start, end, previous_end)

  def is_final_state(self, state: int) -> bool:
    """Final state is -1"""
    # TODO: Could be taken into stronger consideration, e.g. allowing to generate
    # a special end-of-generation token rather than just stopping when the
    # source_text is empty.
    return state >= self.final_state

  #TODO: figure out where this is used and whether it's relevant.
  # Unsure why I created it in the first place.
  def copy(self) -> 'OCRGuide':
    """Very poorly tested copy, not sure how this is used, but kind off trust in it
    since all these fields are constant during the entire generation process except for
    the state which isn't in the class. I.e. this class should be, once initialized,
    more or less stateless w.r.t. the generation process."""
    guide = OCRGuide()
    guide.tokens = self.tokens
    guide.suffix_positions = self.suffix_positions
    guide.fields = self.fields
    guide.field_tokens = self.field_tokens
    guide.tokenizer = self.tokenizer
    guide.newline_token = self.newline_token
    guide.max_field_len = self.max_field_len
    guide.max_skip = self.max_skip
    guide.max_tokens = self.max_tokens
    guide.final_state = self.final_state
    #TODO: deep copy generated span list
    return guide

  def listify(self, generated: str):
    """Parse the generated text to a list of dictionaries
    Parameters:
    generated str: Guided-LLM generated text
    Return:
    List[Dict[str, str]]: entries that the LLM found as a list of dicts.
    """
    field_exps = {
        field: re.compile(
            f'(?:^ ?|\\n )\- {re.escape(field)}\\: +')
        for field in self.fields}
    while True:
      data = dict()
      for field, next_field in zip(
          self.fields, self.fields[1:] + self.fields[:1]):
        ex0, ex1 = field_exps[field], field_exps[next_field]
        match0 = ex0.search(generated)
        if match0 is None:
          return
        start = match0.span()[1]
        generated = generated[start:]
        match1 = ex1.search(generated)
        end = len(generated) if match1 is None else match1.span()[0]
        data[field] = generated[:end]
        generated = generated[end:]
      yield data
