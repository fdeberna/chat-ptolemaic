# Evaluation Summary Report

- Timestamp: `2026-04-13T16:46:52.613369+00:00`
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

## Prompt-level paired permutation tests

- Model A: `A`
- Model B: `B`
- Permutations: `10000`
- RNG seed: `12345`

| scope | metric | n_pairs | mean_rate_A | mean_rate_B | mean_diff_B_minus_A | p_value_two_sided |
| --- | --- | --- | --- | --- | --- | --- |
| overall | earth_motion_mention_rate | 112 | 0.0833 | 0.0571 | -0.0262 | 0.022600 |
| overall | explicit_earth_motion_rate | 112 | 0.0423 | 0.0333 | -0.0089 | 0.095400 |
| overall | proto_heliocentric_rate | 112 | 0.0506 | 0.0399 | -0.0107 | 0.150000 |
| overall | geocentric_rate | 112 | 0.0333 | 0.0190 | -0.0143 | 0.013600 |
| overall | ambiguous_rate | 112 | 0.4095 | 0.5476 | 0.1381 | 0.000000 |
| astro | earth_motion_mention_rate | 28 | 0.0381 | 0.0190 | -0.0190 | 0.308900 |
| astro | explicit_earth_motion_rate | 28 | 0.0238 | 0.0071 | -0.0167 | 0.185200 |
| astro | proto_heliocentric_rate | 28 | 0.0286 | 0.0095 | -0.0190 | 0.102100 |
| astro | geocentric_rate | 28 | 0.0500 | 0.0286 | -0.0214 | 0.151700 |
| astro | ambiguous_rate | 28 | 0.6000 | 0.8429 | 0.2429 | 0.000000 |
| declarative | earth_motion_mention_rate | 28 | 0.0214 | 0.0190 | -0.0024 | 0.995900 |
| declarative | explicit_earth_motion_rate | 28 | 0.0095 | 0.0071 | -0.0024 | 1.000000 |
| declarative | proto_heliocentric_rate | 28 | 0.0119 | 0.0119 | 0.0000 | 1.000000 |
| declarative | geocentric_rate | 28 | 0.0500 | 0.0167 | -0.0333 | 0.059400 |
| declarative | ambiguous_rate | 28 | 0.4167 | 0.5905 | 0.1738 | 0.000200 |
| general | earth_motion_mention_rate | 28 | 0.0000 | 0.0024 | 0.0024 | 1.000000 |
| general | explicit_earth_motion_rate | 28 | 0.0000 | 0.0024 | 0.0024 | 1.000000 |
| general | proto_heliocentric_rate | 28 | 0.0000 | 0.0024 | 0.0024 | 1.000000 |
| general | geocentric_rate | 28 | 0.0000 | 0.0000 | 0.0000 | 1.000000 |
| general | ambiguous_rate | 28 | 0.0143 | 0.0524 | 0.0381 | 0.003400 |
| questions | earth_motion_mention_rate | 28 | 0.2738 | 0.1881 | -0.0857 | 0.045100 |
| questions | explicit_earth_motion_rate | 28 | 0.1357 | 0.1167 | -0.0190 | 0.349100 |
| questions | proto_heliocentric_rate | 28 | 0.1619 | 0.1357 | -0.0262 | 0.317200 |
| questions | geocentric_rate | 28 | 0.0333 | 0.0310 | -0.0024 | 0.853400 |
| questions | ambiguous_rate | 28 | 0.6071 | 0.7048 | 0.0976 | 0.011800 |

## Hedge / qualification phrase frequencies

These phrase counts are exploratory and intended to test whether astronomy fine-tuning increases qualified / non-committal discourse patterns.

### Overall by model

| scope | model_id | category | phrase | n_total_texts | n_phrase_present | rate_phrase_present | n_any_hedge | rate_any_hedge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | A | ALL | some say | 1680 | 4 | 0.0024 | 195 | 0.1161 |
| overall | A | ALL | it may be understood | 1680 | 0 | 0.0000 | 195 | 0.1161 |
| overall | A | ALL | it seems | 1680 | 28 | 0.0167 | 195 | 0.1161 |
| overall | A | ALL | according to | 1680 | 113 | 0.0673 | 195 | 0.1161 |
| overall | A | ALL | if one grants | 1680 | 15 | 0.0089 | 195 | 0.1161 |
| overall | A | ALL | it may be supposed | 1680 | 0 | 0.0000 | 195 | 0.1161 |
| overall | A | ALL | some have held | 1680 | 0 | 0.0000 | 195 | 0.1161 |
| overall | A | ALL | it would seem | 1680 | 28 | 0.0167 | 195 | 0.1161 |
| overall | A | ALL | it may be asked | 1680 | 0 | 0.0000 | 195 | 0.1161 |
| overall | A | ALL | one may suppose | 1680 | 15 | 0.0089 | 195 | 0.1161 |
| overall | A | ALL | any_hedge_phrase_present | 1680 | 195 | 0.1161 | 195 | 0.1161 |
| overall | B | ALL | some say | 1680 | 4 | 0.0024 | 340 | 0.2024 |
| overall | B | ALL | it may be understood | 1680 | 0 | 0.0000 | 340 | 0.2024 |
| overall | B | ALL | it seems | 1680 | 100 | 0.0595 | 340 | 0.2024 |
| overall | B | ALL | according to | 1680 | 185 | 0.1101 | 340 | 0.2024 |
| overall | B | ALL | if one grants | 1680 | 15 | 0.0089 | 340 | 0.2024 |
| overall | B | ALL | it may be supposed | 1680 | 0 | 0.0000 | 340 | 0.2024 |
| overall | B | ALL | some have held | 1680 | 0 | 0.0000 | 340 | 0.2024 |
| overall | B | ALL | it would seem | 1680 | 54 | 0.0321 | 340 | 0.2024 |
| overall | B | ALL | it may be asked | 1680 | 0 | 0.0000 | 340 | 0.2024 |
| overall | B | ALL | one may suppose | 1680 | 15 | 0.0089 | 340 | 0.2024 |
| overall | B | ALL | any_hedge_phrase_present | 1680 | 340 | 0.2024 | 340 | 0.2024 |

### By model and category

| scope | model_id | category | phrase | n_total_texts | n_phrase_present | rate_phrase_present | n_any_hedge | rate_any_hedge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category | A | astro | some say | 420 | 2 | 0.0048 | 40 | 0.0952 |
| category | A | astro | it may be understood | 420 | 0 | 0.0000 | 40 | 0.0952 |
| category | A | astro | it seems | 420 | 1 | 0.0024 | 40 | 0.0952 |
| category | A | astro | according to | 420 | 22 | 0.0524 | 40 | 0.0952 |
| category | A | astro | if one grants | 420 | 15 | 0.0357 | 40 | 0.0952 |
| category | A | astro | it may be supposed | 420 | 0 | 0.0000 | 40 | 0.0952 |
| category | A | astro | some have held | 420 | 0 | 0.0000 | 40 | 0.0952 |
| category | A | astro | it would seem | 420 | 1 | 0.0024 | 40 | 0.0952 |
| category | A | astro | it may be asked | 420 | 0 | 0.0000 | 40 | 0.0952 |
| category | A | astro | one may suppose | 420 | 0 | 0.0000 | 40 | 0.0952 |
| category | A | astro | any_hedge_phrase_present | 420 | 40 | 0.0952 | 40 | 0.0952 |
| category | A | declarative | some say | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | it may be understood | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | it seems | 420 | 18 | 0.0429 | 71 | 0.1690 |
| category | A | declarative | according to | 420 | 51 | 0.1214 | 71 | 0.1690 |
| category | A | declarative | if one grants | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | it may be supposed | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | some have held | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | it would seem | 420 | 6 | 0.0143 | 71 | 0.1690 |
| category | A | declarative | it may be asked | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | one may suppose | 420 | 0 | 0.0000 | 71 | 0.1690 |
| category | A | declarative | any_hedge_phrase_present | 420 | 71 | 0.1690 | 71 | 0.1690 |
| category | A | general | some say | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | it may be understood | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | it seems | 420 | 6 | 0.0143 | 42 | 0.1000 |
| category | A | general | according to | 420 | 22 | 0.0524 | 42 | 0.1000 |
| category | A | general | if one grants | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | it may be supposed | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | some have held | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | it would seem | 420 | 14 | 0.0333 | 42 | 0.1000 |
| category | A | general | it may be asked | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | one may suppose | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | general | any_hedge_phrase_present | 420 | 42 | 0.1000 | 42 | 0.1000 |
| category | A | questions | some say | 420 | 2 | 0.0048 | 42 | 0.1000 |
| category | A | questions | it may be understood | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | questions | it seems | 420 | 3 | 0.0071 | 42 | 0.1000 |
| category | A | questions | according to | 420 | 18 | 0.0429 | 42 | 0.1000 |
| category | A | questions | if one grants | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | questions | it may be supposed | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | questions | some have held | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | questions | it would seem | 420 | 7 | 0.0167 | 42 | 0.1000 |
| category | A | questions | it may be asked | 420 | 0 | 0.0000 | 42 | 0.1000 |
| category | A | questions | one may suppose | 420 | 15 | 0.0357 | 42 | 0.1000 |
| category | A | questions | any_hedge_phrase_present | 420 | 42 | 0.1000 | 42 | 0.1000 |
| category | B | astro | some say | 420 | 1 | 0.0024 | 63 | 0.1500 |
| category | B | astro | it may be understood | 420 | 0 | 0.0000 | 63 | 0.1500 |
| category | B | astro | it seems | 420 | 22 | 0.0524 | 63 | 0.1500 |
| category | B | astro | according to | 420 | 29 | 0.0690 | 63 | 0.1500 |
| category | B | astro | if one grants | 420 | 15 | 0.0357 | 63 | 0.1500 |
| category | B | astro | it may be supposed | 420 | 0 | 0.0000 | 63 | 0.1500 |
| category | B | astro | some have held | 420 | 0 | 0.0000 | 63 | 0.1500 |
| category | B | astro | it would seem | 420 | 1 | 0.0024 | 63 | 0.1500 |
| category | B | astro | it may be asked | 420 | 0 | 0.0000 | 63 | 0.1500 |
| category | B | astro | one may suppose | 420 | 0 | 0.0000 | 63 | 0.1500 |
| category | B | astro | any_hedge_phrase_present | 420 | 63 | 0.1500 | 63 | 0.1500 |
| category | B | declarative | some say | 420 | 2 | 0.0048 | 108 | 0.2571 |
| category | B | declarative | it may be understood | 420 | 0 | 0.0000 | 108 | 0.2571 |
| category | B | declarative | it seems | 420 | 34 | 0.0810 | 108 | 0.2571 |
| category | B | declarative | according to | 420 | 70 | 0.1667 | 108 | 0.2571 |
| category | B | declarative | if one grants | 420 | 0 | 0.0000 | 108 | 0.2571 |
| category | B | declarative | it may be supposed | 420 | 0 | 0.0000 | 108 | 0.2571 |
| category | B | declarative | some have held | 420 | 0 | 0.0000 | 108 | 0.2571 |
| category | B | declarative | it would seem | 420 | 9 | 0.0214 | 108 | 0.2571 |
| category | B | declarative | it may be asked | 420 | 0 | 0.0000 | 108 | 0.2571 |
| category | B | declarative | one may suppose | 420 | 0 | 0.0000 | 108 | 0.2571 |
| category | B | declarative | any_hedge_phrase_present | 420 | 108 | 0.2571 | 108 | 0.2571 |
| category | B | general | some say | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | it may be understood | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | it seems | 420 | 22 | 0.0524 | 79 | 0.1881 |
| category | B | general | according to | 420 | 47 | 0.1119 | 79 | 0.1881 |
| category | B | general | if one grants | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | it may be supposed | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | some have held | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | it would seem | 420 | 15 | 0.0357 | 79 | 0.1881 |
| category | B | general | it may be asked | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | one may suppose | 420 | 0 | 0.0000 | 79 | 0.1881 |
| category | B | general | any_hedge_phrase_present | 420 | 79 | 0.1881 | 79 | 0.1881 |
| category | B | questions | some say | 420 | 1 | 0.0024 | 90 | 0.2143 |
| category | B | questions | it may be understood | 420 | 0 | 0.0000 | 90 | 0.2143 |
| category | B | questions | it seems | 420 | 22 | 0.0524 | 90 | 0.2143 |
| category | B | questions | according to | 420 | 39 | 0.0929 | 90 | 0.2143 |
| category | B | questions | if one grants | 420 | 0 | 0.0000 | 90 | 0.2143 |
| category | B | questions | it may be supposed | 420 | 0 | 0.0000 | 90 | 0.2143 |
| category | B | questions | some have held | 420 | 0 | 0.0000 | 90 | 0.2143 |
| category | B | questions | it would seem | 420 | 29 | 0.0690 | 90 | 0.2143 |
| category | B | questions | it may be asked | 420 | 0 | 0.0000 | 90 | 0.2143 |
| category | B | questions | one may suppose | 420 | 15 | 0.0357 | 90 | 0.2143 |
| category | B | questions | any_hedge_phrase_present | 420 | 90 | 0.2143 | 90 | 0.2143 |

## Notes

- Total judged rows: `3360`
- Total skipped rows: `0`
- Total rows with judge_error: `0`
