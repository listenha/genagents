# Trait Scoring

This file defines how raw item-level survey responses are transformed into analysis-level trait variables for both humans and agents. The complete survey items can be viewed in the file `SURVEY.md`.

## General rule
All survey items are scored on a 1-5 Likert scale.

For reverse-coded items:
`reversed = 6 - raw_response`

Unless otherwise stated, aggregate scores are computed as the mean of keyed items.

## 1. Big Five Personality Inventory (BFI-10)

Derived traits:
- `bfi_extraversion = mean(item_1_rev, item_6)`
- `bfi_agreeableness = mean(item_2, item_7_rev)`
- `bfi_conscientiousness = mean(item_3_rev, item_8)`
- `bfi_neuroticism = mean(item_4_rev, item_9)`
- `bfi_openness = mean(item_5_rev, item_10)`

Reverse-coded items:
- item 1: is reserved
- item 3: tends to be lazy
- item 4: is relaxed, handles stress well
- item 5: has few artistic interests
- item 7: tends to find fault with others

## 2. BES-A Cognitive Empathy Subset

This is a reduced-form subset score, not the full BES-A cognitive subscale.

Derived trait:
- `besa_cognitive_empathy_subset = mean(item_1_rev, item_2, item_3, item_4_rev, item_5_rev)`

Reverse-coded items:
- item 1: I find it hard to know when my friends are frightened
- item 4: I am not usually aware of my friends' feelings
- item 5: I have trouble figuring out when my friends are happy

## 3. REI Rational Ability Subset

This is a reduced-form subset score for rational ability.

Derived trait:
- `rei_rational_ability_subset = mean(item_1_rev, item_2_rev, item_3_rev, item_4_rev, item_5_rev)`

All five items are reverse-coded because higher agreement indicates lower rational ability.

## 4. SD3 Narcissism Subset

This is a reduced-form narcissism subset score.

Derived trait:
- `sd3_narcissism_subset = mean(item_1, item_2_rev, item_3, item_4, item_5)`

Reverse-coded items:
- item 2: I hate being the center of attention

## 5. General Decision Making Style (GDMS) Reduced Form

This survey uses a reduced-form GDMS, so styles are represented as:
- one 2-item rational score
- four single-item style indicators

Derived traits:
- `gdms_rational = mean(item_1, item_6)`
- `gdms_avoidant = item_2`
- `gdms_dependent = item_3`
- `gdms_intuitive = item_4`
- `gdms_spontaneous = item_5`

No reverse coding is applied in the current reduced form.

## Notes for analysis
- Humans and agents use the same survey items, so these derived variables are directly comparable.
- BFI-10 follows a standard short-form scoring structure.
- BES-A, REI, SD3, and GDMS variables here should be described as subset-based or reduced-form traits when writing methods/results.
- GDMS non-rational styles are single-item indicators and should be interpreted cautiously.