# Tabelas da Análise - Modelo 1

Fonte: `analises-modelo1.ipynb`.

## Resumo Por Modelo

| model | n | f1_mean | f1_median | success_rate | doubt_mean | reasoning_tok_mean | answer_tok_mean | reasoning_ratio_mean | in_tok_mean | out_tok_mean | retrieval_reasoning_overlap_mean | reasoning_answer_overlap_mean | cot_cov_mean | cot_novelty_mean | cot_chunk_max_cov_mean | cot_chunk_entropy_mean | cot_chunks_used_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.4659 | 0.4718 | 0.42 | 0.46 | 319.79 | 62.81 | 13.844 | 1055.21 | 383.82 | 0.1222 | 0.1234 | 0.3619 | 0.6381 | 0.2362 | 1.5566 | 4.98 |
| shuffled | 100 | 0.4775 | 0.4987 | 0.49 | 0.39 | 310.37 | 74.54 | 10.7293 | 1055.21 | 386.17 | 0.1089 | 0.1222 | 0.3334 | 0.6666 | 0.2218 | 1.5366 | 4.95 |

## Resumo Por Modelo E Sucesso

| model | is_success | n | f1_mean | doubt_mean | retrieval_reasoning_overlap_mean | reasoning_answer_overlap_mean | cot_cov_mean | cot_novelty_mean | cot_chunk_entropy_mean | cot_chunks_used_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0 | 58 | 0.3991 | 0.6897 | 0.0932 | 0.0582 | 0.2952 | 0.7048 | 1.5453 | 4.9655 |
| original | 1 | 42 | 0.5581 | 0.1429 | 0.1623 | 0.2134 | 0.4539 | 0.5461 | 1.5721 | 5 |
| shuffled | 0 | 51 | 0.4015 | 0.5686 | 0.0827 | 0.051 | 0.282 | 0.718 | 1.5287 | 4.902 |
| shuffled | 1 | 49 | 0.5566 | 0.2041 | 0.1361 | 0.1963 | 0.3869 | 0.6131 | 1.5448 | 5 |

## Correlações: Original

| variavel | f1_vs_gabarito | doubt_count | reasoning_tok_count | answer_tok_count | reasoning_ratio | retrieval_reasoning_lexical_overlap | reasoning_answer_overlap | cot_cov_overall | cot_novelty | cot_retrieval_jaccard | cot_chunk_max_cov | cot_chunk_entropy | cot_chunks_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito | 1 | -0.4824 | 0.2791 | 0.5931 | -0.7481 | 0.3445 | 0.5941 | 0.312 | -0.312 | 0.3497 | 0.3136 | 0.1671 | 0.0471 |
| doubt_count | -0.4824 | 1 | -0.0789 | -0.2961 | 0.4736 | -0.401 | -0.4342 | -0.3939 | 0.3939 | -0.3947 | -0.3916 | -0.0159 | 0.096 |
| reasoning_tok_count | 0.2791 | -0.0789 | 1 | 0.3585 | 0.0983 | 0.4907 | 0.1929 | 0.2736 | -0.2736 | 0.5004 | 0.1055 | 0.1831 | 0.0394 |
| answer_tok_count | 0.5931 | -0.2961 | 0.3585 | 1 | -0.6598 | 0.3814 | 0.6183 | 0.2715 | -0.2715 | 0.3912 | 0.2108 | 0.1919 | -0.025 |
| reasoning_ratio | -0.7481 | 0.4736 | 0.0983 | -0.6598 | 1 | -0.1732 | -0.658 | -0.1933 | 0.1933 | -0.1714 | -0.2402 | -0.084 | 0.0627 |
| retrieval_reasoning_lexical_overlap | 0.3445 | -0.401 | 0.4907 | 0.3814 | -0.1732 | 1 | 0.6284 | 0.9242 | -0.9242 | 0.9934 | 0.8268 | 0.4007 | 0.1786 |
| reasoning_answer_overlap | 0.5941 | -0.4342 | 0.1929 | 0.6183 | -0.658 | 0.6284 | 1 | 0.6745 | -0.6745 | 0.6468 | 0.6626 | 0.3012 | 0.1237 |
| cot_cov_overall | 0.312 | -0.3939 | 0.2736 | 0.2715 | -0.1933 | 0.9242 | 0.6745 | 1 | -1 | 0.9329 | 0.9492 | 0.4257 | 0.1947 |
| cot_novelty | -0.312 | 0.3939 | -0.2736 | -0.2715 | 0.1933 | -0.9242 | -0.6745 | -1 | 1 | -0.9329 | -0.9492 | -0.4257 | -0.1947 |
| cot_retrieval_jaccard | 0.3497 | -0.3947 | 0.5004 | 0.3912 | -0.1714 | 0.9934 | 0.6468 | 0.9329 | -0.9329 | 1 | 0.8312 | 0.4114 | 0.1771 |
| cot_chunk_max_cov | 0.3136 | -0.3916 | 0.1055 | 0.2108 | -0.2402 | 0.8268 | 0.6626 | 0.9492 | -0.9492 | 0.8312 | 1 | 0.3384 | 0.201 |
| cot_chunk_entropy | 0.1671 | -0.0159 | 0.1831 | 0.1919 | -0.084 | 0.4007 | 0.3012 | 0.4257 | -0.4257 | 0.4114 | 0.3384 | 1 | 0.5347 |
| cot_chunks_used | 0.0471 | 0.096 | 0.0394 | -0.025 | 0.0627 | 0.1786 | 0.1237 | 0.1947 | -0.1947 | 0.1771 | 0.201 | 0.5347 | 1 |

## Correlações: Shuffled

| variavel | f1_vs_gabarito | doubt_count | reasoning_tok_count | answer_tok_count | reasoning_ratio | retrieval_reasoning_lexical_overlap | reasoning_answer_overlap | cot_cov_overall | cot_novelty | cot_retrieval_jaccard | cot_chunk_max_cov | cot_chunk_entropy | cot_chunks_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito | 1 | -0.2788 | 0.3876 | 0.6143 | -0.8149 | 0.318 | 0.5651 | 0.2289 | -0.2289 | 0.332 | 0.2405 | 0.0394 | 0.0961 |
| doubt_count | -0.2788 | 1 | -0.1499 | -0.2238 | 0.3768 | -0.3285 | -0.3547 | -0.313 | 0.313 | -0.3055 | -0.3538 | -0.0231 | -0.0023 |
| reasoning_tok_count | 0.3876 | -0.1499 | 1 | 0.4981 | -0.232 | 0.4771 | 0.3623 | 0.3047 | -0.3047 | 0.4847 | 0.191 | 0.2588 | 0.1261 |
| answer_tok_count | 0.6143 | -0.2238 | 0.4981 | 1 | -0.6978 | 0.1739 | 0.3954 | 0.0464 | -0.0464 | 0.1873 | 0.0026 | 0.0525 | 0.0127 |
| reasoning_ratio | -0.8149 | 0.3768 | -0.232 | -0.6978 | 1 | -0.2066 | -0.5328 | -0.1291 | 0.1291 | -0.2171 | -0.1474 | 0.1061 | 0.0452 |
| retrieval_reasoning_lexical_overlap | 0.318 | -0.3285 | 0.4771 | 0.1739 | -0.2066 | 1 | 0.7604 | 0.9332 | -0.9332 | 0.9935 | 0.8685 | 0.2909 | 0.1726 |
| reasoning_answer_overlap | 0.5651 | -0.3547 | 0.3623 | 0.3954 | -0.5328 | 0.7604 | 1 | 0.7383 | -0.7383 | 0.7673 | 0.7014 | 0.2171 | 0.1363 |
| cot_cov_overall | 0.2289 | -0.313 | 0.3047 | 0.0464 | -0.1291 | 0.9332 | 0.7383 | 1 | -1 | 0.942 | 0.9604 | 0.3139 | 0.1973 |
| cot_novelty | -0.2289 | 0.313 | -0.3047 | -0.0464 | 0.1291 | -0.9332 | -0.7383 | -1 | 1 | -0.942 | -0.9604 | -0.3139 | -0.1973 |
| cot_retrieval_jaccard | 0.332 | -0.3055 | 0.4847 | 0.1873 | -0.2171 | 0.9935 | 0.7673 | 0.942 | -0.942 | 1 | 0.8753 | 0.3083 | 0.1829 |
| cot_chunk_max_cov | 0.2405 | -0.3538 | 0.191 | 0.0026 | -0.1474 | 0.8685 | 0.7014 | 0.9604 | -0.9604 | 0.8753 | 1 | 0.2297 | 0.1948 |
| cot_chunk_entropy | 0.0394 | -0.0231 | 0.2588 | 0.0525 | 0.1061 | 0.2909 | 0.2171 | 0.3139 | -0.3139 | 0.3083 | 0.2297 | 1 | 0.7943 |
| cot_chunks_used | 0.0961 | -0.0023 | 0.1261 | 0.0127 | 0.0452 | 0.1726 | 0.1363 | 0.1973 | -0.1973 | 0.1829 | 0.1948 | 0.7943 | 1 |

## Médias Globais: Success Vs Fail

| is_success | cot_cov_overall | cot_novelty | cot_chunk_entropy | cot_chunks_used | retrieval_reasoning_lexical_overlap | doubt_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.289 | 0.711 | 1.5375 | 4.9358 | 0.0883 | 0.633 |
| 1 | 0.4178 | 0.5822 | 1.5574 | 5 | 0.1482 | 0.1758 |

## Deltas (Shuffled - Original)

| variavel | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito_delta | 0.0116 | 0.0754 | -0.1998 | -0.0021 | 0 | 0.0249 | 0.2051 |
| cot_cov_overall_delta | -0.0285 | 0.2048 | -0.4979 | -0.0565 | -0.0079 | 0.032 | 0.5202 |
| cot_novelty_delta | 0.0285 | 0.2048 | -0.5202 | -0.032 | 0.0079 | 0.0565 | 0.4979 |
| cot_chunk_entropy_delta | -0.02 | 0.113 | -0.788 | -0.0198 | -0.0012 | 0.0168 | 0.2879 |
| retrieval_reasoning_lexical_overlap_delta | -0.0133 | 0.0811 | -0.259 | -0.0408 | -0.0013 | 0.0221 | 0.1743 |
| reasoning_answer_overlap_delta | -0.0012 | 0.124 | -0.3279 | -0.038 | 0.0102 | 0.0485 | 0.3302 |
| doubt_count_delta | -0.07 | 0.8439 | -3 | -0.25 | 0 | 0 | 2 |
| sent_support_rate_delta | -0.0205 | 0.1542 | -0.5714 | -0.0979 | 0 | 0.0501 | 0.5897 |

## Correlação Dos Deltas Com Δf1

| variavel | f1_vs_gabarito_delta |
| --- | --- |
| f1_vs_gabarito_delta | 1 |
| reasoning_answer_overlap_delta | 0.4551 |
| retrieval_reasoning_lexical_overlap_delta | 0.2114 |
| cot_cov_overall_delta | 0.1777 |
| sent_support_rate_delta | 0.093 |
| cot_chunk_entropy_delta | -0.0197 |
| cot_novelty_delta | -0.1777 |
| doubt_count_delta | -0.2137 |

## Suporte De Sentenças Por Modelo E Sucesso

| model | is_success | n | f1_mean | sent_support_rate_mean | sent_low_support_rate_mean | sent_mean_max_sim_mean |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0 | 58 | 0.3991 | 0.1125 | 0.7797 | 0.0671 |
| original | 1 | 42 | 0.5581 | 0.2175 | 0.5263 | 0.1216 |
| shuffled | 0 | 51 | 0.4015 | 0.1045 | 0.7844 | 0.0628 |
| shuffled | 1 | 49 | 0.5566 | 0.169 | 0.6211 | 0.0995 |
