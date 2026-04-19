# Evaluation Summary Report

- Timestamp: `2026-04-13T10:33:13.407293+00:00`
- Input file: `generation_pilot_full_double_labels_judged.jsonl`
- Min quality: `1`

## Interpretation Note

- Explicit Earth-motion is the strict metric.
- Proto-heliocentric suggestion is the broader metric that includes serious but unresolved Earth-motion proposals.

## Overall Comparison

| Model | n_total | n_judged | n_q>=1 | n_q2 | Explicit Earth-motion | Explicit Earth-motion q>=1 | Proto-heliocentric suggestion | Proto-heliocentric suggestion q>=1 | n_geo | n_ambig | n_none | Explicit % | Explicit q>=1 % | Proto % | Proto q>=1 % | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1680 | 1680 | 1650 | 69 | 78 | 78 | 110 | 110 | 179 | 864 | 536 | 4.6% | 4.7% | 6.5% | 6.7% | 98.2% | 4.1% | 10.7% | 51.4% |
| B | 1680 | 1680 | 1658 | 107 | 65 | 65 | 94 | 94 | 117 | 1002 | 472 | 3.9% | 3.9% | 5.6% | 5.7% | 98.7% | 6.4% | 7.0% | 59.6% |

## Category Comparison

| Model | Category | n_total | n_judged | n_q>=1 | n_q2 | Explicit Earth-motion | Explicit Earth-motion q>=1 | Proto-heliocentric suggestion | Proto-heliocentric suggestion q>=1 | n_geo | n_ambig | n_none | Explicit % | Explicit q>=1 % | Proto % | Proto q>=1 % | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | astro | 420 | 420 | 407 | 14 | 13 | 13 | 17 | 17 | 65 | 312 | 26 | 3.1% | 3.2% | 4.0% | 4.2% | 96.9% | 3.3% | 15.5% | 74.3% |
| A | declarative | 420 | 420 | 413 | 18 | 6 | 6 | 7 | 7 | 61 | 226 | 126 | 1.4% | 1.5% | 1.7% | 1.7% | 98.3% | 4.3% | 14.5% | 53.8% |
| A | general | 420 | 420 | 414 | 26 | 0 | 0 | 0 | 0 | 2 | 42 | 376 | 0.0% | 0.0% | 0.0% | 0.0% | 98.6% | 6.2% | 0.5% | 10.0% |
| A | questions | 420 | 420 | 416 | 11 | 59 | 59 | 86 | 86 | 51 | 284 | 8 | 14.0% | 14.2% | 20.5% | 20.7% | 99.0% | 2.6% | 12.1% | 67.6% |
| B | astro | 420 | 420 | 416 | 19 | 7 | 7 | 9 | 9 | 35 | 367 | 9 | 1.7% | 1.7% | 2.1% | 2.2% | 99.0% | 4.5% | 8.3% | 87.4% |
| B | declarative | 420 | 420 | 416 | 29 | 5 | 5 | 6 | 6 | 35 | 277 | 102 | 1.2% | 1.2% | 1.4% | 1.4% | 99.0% | 6.9% | 8.3% | 66.0% |
| B | general | 420 | 420 | 408 | 47 | 0 | 0 | 0 | 0 | 2 | 59 | 359 | 0.0% | 0.0% | 0.0% | 0.0% | 97.1% | 11.2% | 0.5% | 14.0% |
| B | questions | 420 | 420 | 418 | 12 | 53 | 53 | 79 | 79 | 45 | 299 | 2 | 12.6% | 12.7% | 18.8% | 18.9% | 99.5% | 2.9% | 10.7% | 71.2% |

## Top Explicit Earth-motion Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged | Explicit Earth-motion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_019 | questions | 15 | 66.7% | 1.07 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_016 | questions | 15 | 46.7% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_017 | questions | 15 | 46.7% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_022 | questions | 15 | 46.7% | 1.00 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_001 | questions | 15 | 40.0% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_007 | questions | 15 | 26.7% | 1.07 | Whether the Earth, being heavy, may nevertheless be moved |
| astro_015 | astro | 15 | 26.7% | 1.00 | The planets do not move with equal speed at all times, and the reason for thi... |
| questions_002 | questions | 15 | 26.7% | 1.00 | Whether the Earth is moved daily or remains ever at rest |
| questions_004 | questions | 15 | 26.7% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_020 | questions | 15 | 20.0% | 1.00 | Whether the Earth may be displaced from the center without overthrowing the o... |

### Model `B`
| Prompt ID | Category | n_judged | Explicit Earth-motion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 46.7% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_022 | questions | 15 | 40.0% | 1.07 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_017 | questions | 15 | 33.3% | 1.13 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_016 | questions | 15 | 33.3% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_001 | questions | 15 | 26.7% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_005 | questions | 15 | 26.7% | 1.00 | Whether it may be maintained that the Earth is numbered among the wandering b... |
| questions_021 | questions | 15 | 20.0% | 1.07 | Whether the heavens alone are moved, and the Earth in no wise |
| questions_023 | questions | 15 | 20.0% | 1.07 | Whether the ancients judged rightly concerning the place and rest of the Earth |
| questions_002 | questions | 15 | 20.0% | 1.00 | Whether the Earth is moved daily or remains ever at rest |
| questions_000 | questions | 15 | 13.3% | 1.07 | Whether the Earth rests immobile at the center of the world |

## Top Proto-heliocentric Suggestion Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged | Proto-heliocentric suggestion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_019 | questions | 15 | 86.7% | 1.07 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_016 | questions | 15 | 73.3% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_017 | questions | 15 | 73.3% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_004 | questions | 15 | 66.7% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_022 | questions | 15 | 60.0% | 1.00 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_001 | questions | 15 | 53.3% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_007 | questions | 15 | 46.7% | 1.07 | Whether the Earth, being heavy, may nevertheless be moved |
| astro_015 | astro | 15 | 26.7% | 1.00 | The planets do not move with equal speed at all times, and the reason for thi... |
| questions_002 | questions | 15 | 26.7% | 1.00 | Whether the Earth is moved daily or remains ever at rest |
| questions_012 | questions | 15 | 20.0% | 1.00 | What is the cause of the apparent backward motion of the planets |

### Model `B`
| Prompt ID | Category | n_judged | Proto-heliocentric suggestion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_016 | questions | 15 | 73.3% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_004 | questions | 15 | 60.0% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_019 | questions | 15 | 60.0% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_017 | questions | 15 | 46.7% | 1.13 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_007 | questions | 15 | 46.7% | 1.07 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_022 | questions | 15 | 46.7% | 1.07 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_001 | questions | 15 | 33.3% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_005 | questions | 15 | 26.7% | 1.00 | Whether it may be maintained that the Earth is numbered among the wandering b... |
| questions_021 | questions | 15 | 20.0% | 1.07 | Whether the heavens alone are moved, and the Earth in no wise |
| questions_023 | questions | 15 | 20.0% | 1.07 | Whether the ancients judged rightly concerning the place and rest of the Earth |

## Notes

- Total judged rows: `3360`
- Total skipped rows: `0`
- Total rows with judge_error: `0`
