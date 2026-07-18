# relationlex-16k-v1

This directory contains the lexical tokenizer JSON and the train-derived exact
whitespace boundary vocabulary used by the RelationLex experiments.

The tokenizer has 16,000 lexical IDs and defines `<s>`, `</s>`, and `<unk>`.
The boundary vocabulary is a separate aligned channel; ID 0 is the empty
boundary. The tokenizer and factorization code are released under the
repository MIT license.
