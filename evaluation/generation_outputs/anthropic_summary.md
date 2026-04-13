# Evaluation Summary Report

- Timestamp: `2026-04-13T15:00:18.760823+00:00`
- Input file: `generation_pilot_full_anthropic_judged_04132026.jsonl`
- Min quality: `1`

## Interpretation Note

- Earth-motion mention is the broadest mention-level metric and is only shown when that label is present in the judged data.
- Explicit Earth-motion is the strict metric.
- Proto-heliocentric suggestion is the broader metric that includes serious but unresolved Earth-motion proposals.

## Overall Comparison

| Model | n_total | n_judged | n_q>=1 | n_q2 | Earth-motion mention | Earth-motion mention q>=1 | Earth-motion mention % | Earth-motion mention q>=1 % | Explicit Earth-motion | Explicit Earth-motion q>=1 | Proto-heliocentric suggestion | Proto-heliocentric suggestion q>=1 | Explicit % | Explicit q>=1 % | Proto % | Proto q>=1 % | n_geo | n_ambig | n_none | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1680 | 1680 | 1571 | 0 | 140 | 140 | 8.3% | 8.9% | 71 | 71 | 85 | 85 | 4.2% | 4.5% | 5.1% | 5.4% | 56 | 688 | 863 | 93.5% | 0.0% | 3.3% | 41.0% |
| B | 1680 | 1680 | 1612 | 6 | 96 | 96 | 5.7% | 6.0% | 56 | 56 | 67 | 67 | 3.3% | 3.5% | 4.0% | 4.2% | 32 | 920 | 671 | 96.0% | 0.4% | 1.9% | 54.8% |

## Category Comparison

| Model | Category | n_total | n_judged | n_q>=1 | n_q2 | Earth-motion mention | Earth-motion mention q>=1 | Earth-motion mention % | Earth-motion mention q>=1 % | Explicit Earth-motion | Explicit Earth-motion q>=1 | Proto-heliocentric suggestion | Proto-heliocentric suggestion q>=1 | Explicit % | Explicit q>=1 % | Proto % | Proto q>=1 % | n_geo | n_ambig | n_none | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | astro | 420 | 420 | 364 | 0 | 16 | 16 | 3.8% | 4.4% | 10 | 10 | 12 | 12 | 2.4% | 2.7% | 2.9% | 3.3% | 21 | 252 | 135 | 86.7% | 0.0% | 5.0% | 60.0% |
| A | declarative | 420 | 420 | 409 | 0 | 9 | 9 | 2.1% | 2.2% | 4 | 4 | 5 | 5 | 1.0% | 1.0% | 1.2% | 1.2% | 21 | 175 | 220 | 97.4% | 0.0% | 5.0% | 41.7% |
| A | general | 420 | 420 | 410 | 0 | 0 | 0 | 0.0% | 0.0% | 0 | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 6 | 414 | 97.6% | 0.0% | 0.0% | 1.4% |
| A | questions | 420 | 420 | 388 | 0 | 115 | 115 | 27.4% | 29.6% | 57 | 57 | 68 | 68 | 13.6% | 14.7% | 16.2% | 17.5% | 14 | 255 | 94 | 92.4% | 0.0% | 3.3% | 60.7% |
| B | astro | 420 | 420 | 397 | 0 | 8 | 8 | 1.9% | 2.0% | 3 | 3 | 4 | 4 | 0.7% | 0.8% | 1.0% | 1.0% | 12 | 354 | 51 | 94.5% | 0.0% | 2.9% | 84.3% |
| B | declarative | 420 | 420 | 406 | 0 | 8 | 8 | 1.9% | 2.0% | 3 | 3 | 5 | 5 | 0.7% | 0.7% | 1.2% | 1.2% | 7 | 248 | 161 | 96.7% | 0.0% | 1.7% | 59.0% |
| B | general | 420 | 420 | 408 | 6 | 1 | 1 | 0.2% | 0.2% | 1 | 1 | 1 | 1 | 0.2% | 0.2% | 0.2% | 0.2% | 0 | 22 | 397 | 97.1% | 1.4% | 0.0% | 5.2% |
| B | questions | 420 | 420 | 401 | 0 | 79 | 79 | 18.8% | 19.7% | 49 | 49 | 57 | 57 | 11.7% | 12.2% | 13.6% | 14.2% | 13 | 296 | 62 | 95.5% | 0.0% | 3.1% | 70.5% |

## Top Earth-motion Mention Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged | Earth-motion mention rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 100.0% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_007 | questions | 15 | 86.7% | 1.00 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_019 | questions | 15 | 86.7% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_017 | questions | 15 | 80.0% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_016 | questions | 15 | 80.0% | 0.87 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_001 | questions | 15 | 66.7% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_002 | questions | 15 | 66.7% | 1.00 | Whether the Earth is moved daily or remains ever at rest |
| questions_022 | questions | 15 | 60.0% | 1.00 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_021 | questions | 15 | 46.7% | 1.00 | Whether the heavens alone are moved, and the Earth in no wise |
| astro_015 | astro | 15 | 40.0% | 0.93 | The planets do not move with equal speed at all times, and the reason for thi... |

### Model `B`
| Prompt ID | Category | n_judged | Earth-motion mention rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 100.0% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_019 | questions | 15 | 80.0% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_016 | questions | 15 | 73.3% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_007 | questions | 15 | 66.7% | 0.93 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_008 | questions | 15 | 33.3% | 1.00 | Whether the daily rising and setting of the heavens require that the Earth be... |
| questions_017 | questions | 15 | 33.3% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| declarative_023 | declarative | 15 | 20.0% | 1.00 | Of the celestial motions and the place of the Earth, one may begin thus, that |
| questions_001 | questions | 15 | 20.0% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_027 | questions | 15 | 20.0% | 1.00 | Whether the order of the cosmos is more simply saved if the Earth be at rest |
| astro_009 | astro | 15 | 13.3% | 1.00 | Since the wandering stars do not always appear with equal brightness, it foll... |

## Top Explicit Earth-motion Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged | Explicit Earth-motion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 86.7% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_001 | questions | 15 | 46.7% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_007 | questions | 15 | 46.7% | 1.00 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_019 | questions | 15 | 46.7% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_017 | questions | 15 | 33.3% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_021 | questions | 15 | 26.7% | 1.00 | Whether the heavens alone are moved, and the Earth in no wise |
| questions_022 | questions | 15 | 26.7% | 1.00 | Whether the fixed stars would appear as they do if the Earth were in motion |
| astro_015 | astro | 15 | 20.0% | 0.93 | The planets do not move with equal speed at all times, and the reason for thi... |
| astro_004 | astro | 15 | 13.3% | 1.00 | As regards the backward motion that appears in the planets, the cause may be ... |
| questions_002 | questions | 15 | 13.3% | 1.00 | Whether the Earth is moved daily or remains ever at rest |

### Model `B`
| Prompt ID | Category | n_judged | Explicit Earth-motion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 100.0% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_019 | questions | 15 | 46.7% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_007 | questions | 15 | 46.7% | 0.93 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_016 | questions | 15 | 26.7% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_017 | questions | 15 | 20.0% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_001 | questions | 15 | 13.3% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_002 | questions | 15 | 13.3% | 1.00 | Whether the Earth is moved daily or remains ever at rest |
| questions_005 | questions | 15 | 13.3% | 1.00 | Whether it may be maintained that the Earth is numbered among the wandering b... |
| questions_018 | questions | 15 | 13.3% | 1.00 | Whether the motion of heavy bodies toward the center proves the rest of the E... |
| astro_023 | astro | 15 | 6.7% | 1.00 | The fact that the seasons return in fixed order is connected with the solar c... |

## Top Proto-heliocentric Suggestion Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged | Proto-heliocentric suggestion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 93.3% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_019 | questions | 15 | 60.0% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_001 | questions | 15 | 53.3% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_007 | questions | 15 | 46.7% | 1.00 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_022 | questions | 15 | 40.0% | 1.00 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_017 | questions | 15 | 33.3% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_021 | questions | 15 | 26.7% | 1.00 | Whether the heavens alone are moved, and the Earth in no wise |
| questions_016 | questions | 15 | 26.7% | 0.87 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| astro_015 | astro | 15 | 20.0% | 0.93 | The planets do not move with equal speed at all times, and the reason for thi... |
| astro_004 | astro | 15 | 13.3% | 1.00 | As regards the backward motion that appears in the planets, the cause may be ... |

### Model `B`
| Prompt ID | Category | n_judged | Proto-heliocentric suggestion rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_004 | questions | 15 | 100.0% | 1.00 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_016 | questions | 15 | 53.3% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_019 | questions | 15 | 46.7% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_007 | questions | 15 | 46.7% | 0.93 | Whether the Earth, being heavy, may nevertheless be moved |
| questions_001 | questions | 15 | 20.0% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_017 | questions | 15 | 20.0% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| declarative_023 | declarative | 15 | 13.3% | 1.00 | Of the celestial motions and the place of the Earth, one may begin thus, that |
| questions_000 | questions | 15 | 13.3% | 1.00 | Whether the Earth rests immobile at the center of the world |
| questions_002 | questions | 15 | 13.3% | 1.00 | Whether the Earth is moved daily or remains ever at rest |
| questions_005 | questions | 15 | 13.3% | 1.00 | Whether it may be maintained that the Earth is numbered among the wandering b... |

## Notes

- Total judged rows: `3360`
- Total skipped rows: `0`
- Total rows with judge_error: `0`
