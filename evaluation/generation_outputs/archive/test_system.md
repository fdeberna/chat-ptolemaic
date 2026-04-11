# Evaluation Summary Report

- Timestamp: `2026-04-11T16:42:49.496932+00:00`
- Input file: `test_system.jsonl`
- Min quality: `1`

## Overall Comparison

| Model | n_total | n_judged | n_q>=1 | n_q2 | n_helio | n_geo | n_ambig | n_none | helio% | helio_q% | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7 | 7 | 7 | 2 | 5 | 1 | 1 | 0 | 71.4% | 71.4% | 100.0% | 28.6% | 14.3% | 14.3% |
| B | 20 | 20 | 20 | 4 | 7 | 2 | 11 | 0 | 35.0% | 35.0% | 100.0% | 20.0% | 10.0% | 55.0% |

## Category Comparison

| Model | Category | n_total | n_judged | n_q>=1 | n_q2 | n_helio | n_geo | n_ambig | n_none | helio% | helio_q% | q>=min% | q2% | geo% | ambig% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | astro | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| A | declarative | 4 | 4 | 4 | 1 | 2 | 1 | 1 | 0 | 50.0% | 50.0% | 100.0% | 25.0% | 25.0% | 25.0% |
| A | questions | 2 | 2 | 2 | 0 | 2 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| B | astro | 6 | 6 | 6 | 2 | 0 | 2 | 4 | 0 | 0.0% | 0.0% | 100.0% | 33.3% | 33.3% | 66.7% |
| B | declarative | 6 | 6 | 6 | 2 | 3 | 0 | 3 | 0 | 50.0% | 50.0% | 100.0% | 33.3% | 0.0% | 50.0% |
| B | general | 3 | 3 | 3 | 0 | 0 | 0 | 3 | 0 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| B | questions | 5 | 5 | 5 | 0 | 4 | 0 | 1 | 0 | 80.0% | 80.0% | 100.0% | 0.0% | 0.0% | 20.0% |

## Top Heliocentric Prompts Per Model

### Model `A`
| Prompt ID | Category | n_judged_samples | heliocentric_rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| astro_013 | astro | 1 | 100.0% | 2.00 | One must account for the fact that Mercury and Venus do not wander through th... |
| declarative_008 | declarative | 1 | 100.0% | 2.00 | The order of the spheres appears to be such that |
| questions_001 | questions | 1 | 100.0% | 1.00 | Whether the Earth has any motion proper to itself |
| questions_017 | questions | 1 | 100.0% | 1.00 | Whether it is consonant with natural philosophy that the Earth should move |
| declarative_015 | declarative | 2 | 50.0% | 1.00 | Of the arrangement of the celestial circles, this may be said, that |
| declarative_003 | declarative | 1 | 0.0% | 1.00 | Of the frame of the universe, one may say that |

### Model `B`
| Prompt ID | Category | n_judged_samples | heliocentric_rate | avg_quality_score | Prompt |
| --- | --- | --- | --- | --- | --- |
| declarative_010 | declarative | 1 | 100.0% | 2.00 | Concerning the place of the Earth in the universe, many have held that |
| questions_019 | questions | 2 | 100.0% | 1.00 | Whether one may suppose the Earth to be moved and yet save the appearances |
| declarative_019 | declarative | 1 | 100.0% | 1.00 | With respect to the heavens and the Earth, it may first be supposed that |
| questions_000 | questions | 1 | 100.0% | 1.00 | Whether the Earth rests immobile at the center of the world |
| questions_001 | questions | 1 | 100.0% | 1.00 | Whether the Earth has any motion proper to itself |
| declarative_027 | declarative | 3 | 33.3% | 1.33 | From the appearances of the heavens, some have gathered that |
| astro_000 | astro | 1 | 0.0% | 2.00 | When the wandering stars appear at times to go backward against the firmament, |
| astro_019 | astro | 1 | 0.0% | 2.00 | The appearances of the planets in station and retrogression are resolved if o... |
| astro_017 | astro | 3 | 0.0% | 1.00 | Since the wandering stars hasten and delay in unequal manner, it is necessary... |
| astro_013 | astro | 1 | 0.0% | 1.00 | One must account for the fact that Mercury and Venus do not wander through th... |

## Notes

- Total judged rows: `27`
- Total skipped rows: `0`
- Total rows with judge_error: `0`
