# Tabelas de analise - modelo 2

Fonte: `analises-modelo2.ipynb`. Valores extraidos das saidas exibidas no notebook.

## Resumo por modelo

| indice | model | n | f1_mean | f1_median | success_rate | doubt_mean | reasoning_tok_mean | answer_tok_mean | reasoning_ratio_mean | in_tok_mean | out_tok_mean | retrieval_reasoning_overlap_mean | reasoning_answer_overlap_mean | cot_cov_mean | cot_novelty_mean | cot_chunk_max_cov_mean | cot_chunk_entropy_mean | cot_chunks_used_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | original | 50 | 0.616842 | 0.630164 | 0.840000 | 0.080000 | 161.160000 | 27.340000 | 7.749730 | 1771.420000 | 298.560000 | 0.106019 | 0.209468 | 0.527217 | 0.472783 | 0.356298 | 2.053318 | 8.000000 |
| 1 | shuffled | 50 | 0.638628 | 0.641463 | 0.860000 | 0.100000 | 170.760000 | 27.260000 | 7.860759 | 1771.420000 | 318.620000 | 0.118972 | 0.214744 | 0.562431 | 0.437569 | 0.370154 | 2.054250 | 8.000000 |


## Resumo por modelo e sucesso

| indice | model | is_success | n | f1_mean | doubt_mean | retrieval_reasoning_overlap_mean | reasoning_answer_overlap_mean | cot_cov_mean | cot_novelty_mean | cot_chunk_entropy_mean | cot_chunks_used_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | original | 0 | 8 | 0.436726 | 0.500000 | 0.090094 | 0.122463 | 0.462035 | 0.537965 | 2.060590 | 8.000000 |
| 1 | original | 1 | 42 | 0.651149 | 0.000000 | 0.109053 | 0.226041 | 0.539633 | 0.460367 | 2.051933 | 8.000000 |
| 2 | shuffled | 0 | 7 | 0.459447 | 0.428571 | 0.109126 | 0.085953 | 0.473761 | 0.526239 | 2.048204 | 8.000000 |
| 3 | shuffled | 1 | 43 | 0.667797 | 0.046512 | 0.120575 | 0.235710 | 0.576865 | 0.423135 | 2.055235 | 8.000000 |


## Correlacoes: original

| indice | f1_vs_gabarito | doubt_count | reasoning_tok_count | answer_tok_count | reasoning_ratio | retrieval_reasoning_lexical_overlap | reasoning_answer_overlap | cot_cov_overall | cot_novelty | cot_retrieval_jaccard | cot_chunk_max_cov | cot_chunk_entropy | cot_chunks_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito | 1.000000 | -0.387038 | -0.385314 | 0.010712 | -0.493003 | -0.020941 | 0.283892 | 0.123293 | -0.123293 | -0.027301 | 0.292492 | -0.235538 | NaN |
| doubt_count | -0.387038 | 1.000000 | 0.210845 | -0.289121 | 0.681394 | -0.340847 | -0.424722 | -0.459657 | 0.459657 | -0.351550 | -0.417997 | -0.024478 | NaN |
| reasoning_tok_count | -0.385314 | 0.210845 | 1.000000 | 0.340449 | 0.347267 | 0.195633 | -0.336621 | -0.264268 | 0.264268 | 0.206269 | -0.599770 | 0.176974 | NaN |
| answer_tok_count | 0.010712 | -0.289121 | 0.340449 | 1.000000 | -0.545455 | 0.284740 | 0.342570 | 0.105813 | -0.105813 | 0.296134 | -0.118095 | 0.117932 | NaN |
| reasoning_ratio | -0.493003 | 0.681394 | 0.347267 | -0.545455 | 1.000000 | -0.237664 | -0.645799 | -0.400104 | 0.400104 | -0.233483 | -0.457881 | 0.058923 | NaN |
| retrieval_reasoning_lexical_overlap | -0.020941 | -0.340847 | 0.195633 | 0.284740 | -0.237664 | 1.000000 | 0.483358 | 0.797501 | -0.797501 | 0.984994 | 0.499019 | 0.387349 | NaN |
| reasoning_answer_overlap | 0.283892 | -0.424722 | -0.336621 | 0.342570 | -0.645799 | 0.483358 | 1.000000 | 0.700424 | -0.700424 | 0.474228 | 0.738175 | 0.279330 | NaN |
| cot_cov_overall | 0.123293 | -0.459657 | -0.264268 | 0.105813 | -0.400104 | 0.797501 | 0.700424 | 1.000000 | -1.000000 | 0.806083 | 0.854195 | 0.405789 | NaN |
| cot_novelty | -0.123293 | 0.459657 | 0.264268 | -0.105813 | 0.400104 | -0.797501 | -0.700424 | -1.000000 | 1.000000 | -0.806083 | -0.854195 | -0.405789 | NaN |
| cot_retrieval_jaccard | -0.027301 | -0.351550 | 0.206269 | 0.296134 | -0.233483 | 0.984994 | 0.474228 | 0.806083 | -0.806083 | 1.000000 | 0.485214 | 0.417477 | NaN |
| cot_chunk_max_cov | 0.292492 | -0.417997 | -0.599770 | -0.118095 | -0.457881 | 0.499019 | 0.738175 | 0.854195 | -0.854195 | 0.485214 | 1.000000 | 0.229098 | NaN |
| cot_chunk_entropy | -0.235538 | -0.024478 | 0.176974 | 0.117932 | 0.058923 | 0.387349 | 0.279330 | 0.405789 | -0.405789 | 0.417477 | 0.229098 | 1.000000 | NaN |
| cot_chunks_used | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |


## Correlacoes: shuffled

| indice | f1_vs_gabarito | doubt_count | reasoning_tok_count | answer_tok_count | reasoning_ratio | retrieval_reasoning_lexical_overlap | reasoning_answer_overlap | cot_cov_overall | cot_novelty | cot_retrieval_jaccard | cot_chunk_max_cov | cot_chunk_entropy | cot_chunks_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito | 1.000000 | -0.317125 | -0.509480 | -0.014923 | -0.480939 | -0.085679 | 0.366346 | 0.161237 | -0.161237 | -0.095456 | 0.349322 | 0.029650 | NaN |
| doubt_count | -0.317125 | 1.000000 | 0.450966 | -0.178367 | 0.700692 | -0.200240 | -0.496039 | -0.384869 | 0.384869 | -0.198767 | -0.387906 | 0.026849 | NaN |
| reasoning_tok_count | -0.509480 | 0.450966 | 1.000000 | 0.311457 | 0.440636 | -0.063752 | -0.663165 | -0.499392 | 0.499392 | -0.030895 | -0.722293 | -0.196604 | NaN |
| answer_tok_count | -0.014923 | -0.178367 | 0.311457 | 1.000000 | -0.529577 | -0.186806 | 0.172957 | -0.340805 | 0.340805 | -0.154436 | -0.319166 | -0.311232 | NaN |
| reasoning_ratio | -0.480939 | 0.700692 | 0.440636 | -0.529577 | 1.000000 | 0.034508 | -0.629688 | -0.136772 | 0.136772 | 0.016005 | -0.292834 | 0.109013 | NaN |
| retrieval_reasoning_lexical_overlap | -0.085679 | -0.200240 | -0.063752 | -0.186806 | 0.034508 | 1.000000 | 0.253983 | 0.776809 | -0.776809 | 0.978743 | 0.494372 | 0.556047 | NaN |
| reasoning_answer_overlap | 0.366346 | -0.496039 | -0.663165 | 0.172957 | -0.629688 | 0.253983 | 1.000000 | 0.539258 | -0.539258 | 0.262061 | 0.668822 | 0.239768 | NaN |
| cot_cov_overall | 0.161237 | -0.384869 | -0.499392 | -0.340805 | -0.136772 | 0.776809 | 0.539258 | 1.000000 | -1.000000 | 0.782621 | 0.835094 | 0.518175 | NaN |
| cot_novelty | -0.161237 | 0.384869 | 0.499392 | 0.340805 | 0.136772 | -0.776809 | -0.539258 | -1.000000 | 1.000000 | -0.782621 | -0.835094 | -0.518175 | NaN |
| cot_retrieval_jaccard | -0.095456 | -0.198767 | -0.030895 | -0.154436 | 0.016005 | 0.978743 | 0.262061 | 0.782621 | -0.782621 | 1.000000 | 0.494048 | 0.522856 | NaN |
| cot_chunk_max_cov | 0.349322 | -0.387906 | -0.722293 | -0.319166 | -0.292834 | 0.494372 | 0.668822 | 0.835094 | -0.835094 | 0.494048 | 1.000000 | 0.332567 | NaN |
| cot_chunk_entropy | 0.029650 | 0.026849 | -0.196604 | -0.311232 | 0.109013 | 0.556047 | 0.239768 | 0.518175 | -0.518175 | 0.522856 | 0.332567 | 1.000000 | NaN |
| cot_chunks_used | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |


## Medias globais: success vs fail

| indice | is_success | cot_cov_overall | cot_novelty | cot_chunk_entropy | cot_chunks_used | retrieval_reasoning_lexical_overlap | doubt_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.467507 | 0.532493 | 2.054810 | 8.000000 | 0.098976 | 0.466667 |
| 1 | 1 | 0.558468 | 0.441532 | 2.053603 | 8.000000 | 0.114882 | 0.023529 |


## Deltas (shuffled - original)

| indice | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito_delta | 0.021786 | 0.078253 | -0.139981 | -0.007243 | 0.004681 | 0.038145 | 0.418227 |
| cot_cov_overall_delta | 0.035213 | 0.164315 | -0.436244 | -0.028692 | 0.031651 | 0.074337 | 0.618855 |
| cot_novelty_delta | -0.035213 | 0.164315 | -0.618855 | -0.074337 | -0.031651 | 0.028692 | 0.436244 |
| cot_chunk_entropy_delta | 0.000933 | 0.024533 | -0.105741 | -0.006026 | 0.002278 | 0.007756 | 0.112301 |
| retrieval_reasoning_lexical_overlap_delta | 0.012953 | 0.045895 | -0.093299 | -0.011503 | 0.012499 | 0.033577 | 0.172439 |
| reasoning_answer_overlap_delta | 0.005276 | 0.099324 | -0.256398 | -0.035982 | -0.000138 | 0.056338 | 0.257676 |
| doubt_count_delta | 0.020000 | 0.377424 | -1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| sent_support_rate_delta | 0.032769 | 0.166065 | -0.444444 | -0.041351 | 0.051632 | 0.146291 | 0.307692 |


## Correlacao dos deltas com Delta F1

| indice | f1_vs_gabarito_delta |
| --- | --- |
| f1_vs_gabarito_delta | 1.000000 |
| reasoning_answer_overlap_delta | 0.322628 |
| cot_cov_overall_delta | 0.279401 |
| sent_support_rate_delta | 0.178518 |
| retrieval_reasoning_lexical_overlap_delta | 0.121644 |
| cot_chunk_entropy_delta | -0.012366 |
| cot_novelty_delta | -0.279401 |
| doubt_count_delta | -0.295864 |


## Suporte de sentencas por modelo e sucesso

| indice | model | is_success | n | f1_mean | sent_support_rate_mean | sent_low_support_rate_mean | sent_mean_max_sim_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | original | 0 | 8 | 0.436726 | 0.149554 | 0.636905 | 0.096864 |
| 1 | original | 1 | 42 | 0.651149 | 0.174844 | 0.456350 | 0.124048 |
| 2 | shuffled | 0 | 7 | 0.459447 | 0.171516 | 0.547028 | 0.099707 |
| 3 | shuffled | 1 | 43 | 0.667797 | 0.208784 | 0.418528 | 0.131983 |


## Distribuicao de cot_chunks_used

```text
model     cot_chunks_used
original  8.0                50
shuffled  8.0                50
Name: count, dtype: int64
```
