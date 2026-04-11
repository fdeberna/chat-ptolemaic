# Evaluation Summary Report

- Timestamp: `2026-04-11T16:10:34.595443+00:00`
- Input file: `generation_pilot_full_judged_full.jsonl`
- Min quality: `1`

## Overall Comparison

| Model | n_total | n_judged | n_q>=1 | n_q2 | n_helio | n_geo | n_ambig | n_none | helio% | helio_q% | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1680 | 1680 | 1288 | 27 | 144 | 299 | 664 | 594 | 8.6% | 11.1% | 76.7% | 1.6% | 17.8% | 39.5% |
| B | 1680 | 1680 | 1377 | 60 | 184 | 358 | 643 | 506 | 11.0% | 13.4% | 82.0% | 3.6% | 21.3% | 38.3% |

## Category Comparison

| Model | Category | n_total | n_judged | n_q>=1 | n_q2 | n_helio | n_geo | n_ambig | n_none | helio% | helio_q% | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | astro | 420 | 420 | 349 | 4 | 33 | 119 | 200 | 71 | 7.9% | 9.5% | 83.1% | 1.0% | 28.3% | 47.6% |
| A | declarative | 420 | 420 | 362 | 13 | 17 | 89 | 191 | 123 | 4.0% | 4.7% | 86.2% | 3.1% | 21.2% | 45.5% |
| A | general | 420 | 420 | 205 | 4 | 0 | 5 | 41 | 374 | 0.0% | 0.0% | 48.8% | 1.0% | 1.2% | 9.8% |
| A | questions | 420 | 420 | 372 | 6 | 94 | 86 | 232 | 26 | 22.4% | 25.0% | 88.6% | 1.4% | 20.5% | 55.2% |
| B | astro | 420 | 420 | 391 | 20 | 57 | 162 | 172 | 30 | 13.6% | 14.6% | 93.1% | 4.8% | 38.6% | 41.0% |
| B | declarative | 420 | 420 | 374 | 22 | 20 | 84 | 213 | 102 | 4.8% | 5.3% | 89.0% | 5.2% | 20.0% | 50.7% |
| B | general | 420 | 420 | 222 | 5 | 6 | 5 | 54 | 356 | 1.4% | 2.7% | 52.9% | 1.2% | 1.2% | 12.9% |
| B | questions | 420 | 420 | 390 | 13 | 101 | 107 | 204 | 18 | 24.0% | 25.9% | 92.9% | 3.1% | 25.5% | 48.6% |

## Top Heliocentric Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged_samples | heliocentric_rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_019 | questions | 15 | 100.0% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_016 | questions | 15 | 86.7% | 0.87 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_017 | questions | 15 | 66.7% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_022 | questions | 15 | 60.0% | 0.93 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_004 | questions | 15 | 53.3% | 0.93 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| questions_020 | questions | 15 | 40.0% | 1.00 | Whether the Earth may be displaced from the center without overthrowing the o... |
| questions_021 | questions | 15 | 33.3% | 1.00 | Whether the heavens alone are moved, and the Earth in no wise |
| astro_015 | astro | 15 | 33.3% | 0.87 | The planets do not move with equal speed at all times, and the reason for thi... |
| questions_001 | questions | 15 | 26.7% | 1.13 | Whether the Earth has any motion proper to itself |
| astro_004 | astro | 15 | 26.7% | 1.00 | As regards the backward motion that appears in the planets, the cause may be ... |

### Model `B`
| Prompt ID | Category | n_judged_samples | heliocentric_rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| questions_016 | questions | 15 | 93.3% | 1.00 | Whether appearances in the heavens may be preserved if the Earth be granted m... |
| questions_019 | questions | 15 | 80.0% | 1.20 | Whether one may suppose the Earth to be moved and yet save the appearances |
| questions_017 | questions | 15 | 60.0% | 1.13 | Whether it is consonant with natural philosophy that the Earth should move |
| questions_022 | questions | 15 | 60.0% | 1.00 | Whether the fixed stars would appear as they do if the Earth were in motion |
| questions_005 | questions | 15 | 46.7% | 1.00 | Whether it may be maintained that the Earth is numbered among the wandering b... |
| questions_004 | questions | 15 | 46.7% | 0.93 | Whether the Sun circles the Earth, or the Earth is borne about the Sun |
| astro_004 | astro | 15 | 40.0% | 1.20 | As regards the backward motion that appears in the planets, the cause may be ... |
| astro_027 | astro | 15 | 40.0% | 1.00 | The observed order of the planets with respect to the Sun may be explained from |
| astro_000 | astro | 15 | 33.3% | 1.00 | When the wandering stars appear at times to go backward against the firmament, |
| astro_013 | astro | 15 | 33.3% | 1.00 | One must account for the fact that Mercury and Venus do not wander through th... |

## Notes

- Total judged rows: `3360`
- Total skipped rows: `0`
- Total rows with judge_error: `0`
