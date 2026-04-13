# Evaluation Summary Report

- Timestamp: `2026-04-13T10:39:19.549248+00:00`
- Input file: `generation_pilot_full_double_labels_judged.jsonl`
- Min quality: `2`

## Interpretation Note

- Explicit Earth-motion is the strict metric.
- Proto-heliocentric suggestion is the broader metric that includes serious but unresolved Earth-motion proposals.

## Overall Comparison

| Model | n_total | n_judged | n_q>=2 | n_q2 | Explicit Earth-motion | Explicit Earth-motion q>=2 | Proto-heliocentric suggestion | Proto-heliocentric suggestion q>=2 | n_geo | n_ambig | n_none | Explicit % | Explicit q>=2 % | Proto % | Proto q>=2 % | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1680 | 1680 | 69 | 69 | 78 | 6 | 110 | 9 | 179 | 864 | 536 | 4.6% | 8.7% | 6.5% | 13.0% | 4.1% | 4.1% | 10.7% | 51.4% |
| B | 1680 | 1680 | 107 | 107 | 65 | 8 | 94 | 8 | 117 | 1002 | 472 | 3.9% | 7.5% | 5.6% | 7.5% | 6.4% | 6.4% | 7.0% | 59.6% |

## Category Comparison

| Model | Category | n_total | n_judged | n_q>=2 | n_q2 | Explicit Earth-motion | Explicit Earth-motion q>=2 | Proto-heliocentric suggestion | Proto-heliocentric suggestion q>=2 | n_geo | n_ambig | n_none | Explicit % | Explicit q>=2 % | Proto % | Proto q>=2 % | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | astro | 420 | 420 | 14 | 14 | 13 | 3 | 17 | 5 | 65 | 312 | 26 | 3.1% | 21.4% | 4.0% | 35.7% | 3.3% | 3.3% | 15.5% | 74.3% |
| A | declarative | 420 | 420 | 18 | 18 | 6 | 3 | 7 | 3 | 61 | 226 | 126 | 1.4% | 16.7% | 1.7% | 16.7% | 4.3% | 4.3% | 14.5% | 53.8% |
| A | general | 420 | 420 | 26 | 26 | 0 | 0 | 0 | 0 | 2 | 42 | 376 | 0.0% | 0.0% | 0.0% | 0.0% | 6.2% | 6.2% | 0.5% | 10.0% |
| A | questions | 420 | 420 | 11 | 11 | 59 | 0 | 86 | 1 | 51 | 284 | 8 | 14.0% | 0.0% | 20.5% | 9.1% | 2.6% | 2.6% | 12.1% | 67.6% |
| B | astro | 420 | 420 | 19 | 19 | 7 | 1 | 9 | 1 | 35 | 367 | 9 | 1.7% | 5.3% | 2.1% | 5.3% | 4.5% | 4.5% | 8.3% | 87.4% |
| B | declarative | 420 | 420 | 29 | 29 | 5 | 2 | 6 | 2 | 35 | 277 | 102 | 1.2% | 6.9% | 1.4% | 6.9% | 6.9% | 6.9% | 8.3% | 66.0% |
| B | general | 420 | 420 | 47 | 47 | 0 | 0 | 0 | 0 | 2 | 59 | 359 | 0.0% | 0.0% | 0.0% | 0.0% | 11.2% | 11.2% | 0.5% | 14.0% |
| B | questions | 420 | 420 | 12 | 12 | 53 | 5 | 79 | 5 | 45 | 299 | 2 | 12.6% | 41.7% | 18.8% | 41.7% | 2.9% | 2.9% | 10.7% | 71.2% |

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
