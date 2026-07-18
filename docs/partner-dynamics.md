# Partner dynamics and stacked relation blocks

The current Relation LM chooses **one anchor and up to eight partners per query position**. Partners are processed independently with the anchor and then combined by a learned softmax-weighted reduction.

## Causal anchor exclusion

The optional `exclude_anchor_history_from_partners` rule removes every position that has already been selected as an anchor at the current or an earlier query position. It does not inspect future anchor choices.

A 20.39M-parameter RelationLex checkpoint was evaluated without retraining. Relative joint-BPB changes were:

| Context | Joint-BPB change |
|---:|---:|
| 128 | +0.196% |
| 256 | +0.128% |
| 512 | +0.008% |

This suggests that causal anchor/partner role separation is a viable training experiment, especially at longer contexts.

## Multiple-partner usage

The model does not collapse to one partner. Mean active and effective partner counts were:

| Context | Active partners | Effective partners | Top-1 weight | Top-2 cumulative |
|---:|---:|---:|---:|---:|
| 128 | 6.10 | 4.48 | 44.8% | 64.5% |
| 256 | 7.05 | 5.11 | 41.6% | 60.6% |
| 512 | 7.53 | 5.81 | 36.5% | 54.7% |

Influence is reported as `softmax_weight * relation_vector_norm`. This is a contribution-magnitude proxy, not a causal attribution, because relation vectors can cancel.

## Partner sharing across anchors

Shared partner hubs are common:

- 84–86% of unique selected partner positions are owned by at least two distinct anchors.
- 97.6–98.2% of selection events use a partner that is shared by multiple anchors.
- One partner was selected by as many as 28, 33 and 43 distinct anchors at contexts 128, 256 and 512.

Sharing is therefore an important diagnostic signal. It is not forbidden by default because hub positions may carry useful global information.

## Near, middle and far regions

Zones use backward distance divided by the available causal history:

- near: `< 1/3`
- middle: `1/3 .. 2/3`
- far: `>= 2/3`

Influence distribution:

| Context | Near | Middle | Far |
|---:|---:|---:|---:|
| 128 | 79.3% | 13.6% | 7.1% |
| 256 | 88.9% | 7.7% | 3.4% |
| 512 | 93.3% | 4.4% | 2.3% |

The long-context model is strongly local despite having global retrieval. Future work should test zone-aware partner budgets or regularization rather than assuming that a larger context automatically produces long-range relations.

## Stacked relation blocks

`StackedRelationLayers` is the relation analogue of repeated attention blocks. Each block performs a fresh anchor/partner selection, relation transformation and residual update. A two-block, 1.73M-parameter standalone stack passed a strict causal-prefix test.

Partners still do not communicate directly with one another. A separate `PartnerSetMixer` would be required for true partner-to-partner interaction before weighted reduction.
