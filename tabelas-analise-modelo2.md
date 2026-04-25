# Tabelas da Análise - Modelo 2

Fonte: `analises-modelo2.ipynb`.

## Resumo Por Modelo

| model | n | f1_mean | f1_median | success_rate | doubt_mean | reasoning_tok_mean | answer_tok_mean | reasoning_ratio_mean | in_tok_mean | out_tok_mean | retrieval_reasoning_overlap_mean | reasoning_answer_overlap_mean | cot_cov_mean | cot_novelty_mean | cot_chunk_max_cov_mean | cot_chunk_entropy_mean | cot_chunks_used_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 100 | 0.4629 | 0.4662 | 0.42 | 0.47 | 318.46 | 60.25 | 13.7795 | 1056 | 380.05 | 0.119 | 0.13 | 0.3577 | 0.6423 | 0.2375 | 1.5566 | 4.98 |
| shuffled | 100 | 0.4696 | 0.4968 | 0.47 | 0.51 | 312.27 | 65.8 | 11.827 | 1056 | 379.45 | 0.1125 | 0.1277 | 0.3369 | 0.6631 | 0.2238 | 1.5426 | 4.96 |

## Resumo Por Modelo E Sucesso

| model | is_success | n | f1_mean | doubt_mean | retrieval_reasoning_overlap_mean | reasoning_answer_overlap_mean | cot_cov_mean | cot_novelty_mean | cot_chunk_entropy_mean | cot_chunks_used_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0 | 58 | 0.3981 | 0.7241 | 0.092 | 0.0646 | 0.2988 | 0.7012 | 1.5465 | 4.9655 |
| original | 1 | 42 | 0.5524 | 0.119 | 0.1562 | 0.2202 | 0.4391 | 0.5609 | 1.5705 | 5 |
| shuffled | 0 | 53 | 0.3961 | 0.8491 | 0.0879 | 0.0615 | 0.2804 | 0.7196 | 1.5333 | 4.9245 |
| shuffled | 1 | 47 | 0.5525 | 0.1277 | 0.1401 | 0.2024 | 0.4006 | 0.5994 | 1.553 | 5 |

## Correlações: Original

| variavel | f1_vs_gabarito | doubt_count | reasoning_tok_count | answer_tok_count | reasoning_ratio | retrieval_reasoning_lexical_overlap | reasoning_answer_overlap | cot_cov_overall | cot_novelty | cot_retrieval_jaccard | cot_chunk_max_cov | cot_chunk_entropy | cot_chunks_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito | 1 | -0.4906 | 0.2748 | 0.5824 | -0.7672 | 0.3362 | 0.6121 | 0.2968 | -0.2968 | 0.3431 | 0.282 | 0.1509 | 0.0435 |
| doubt_count | -0.4906 | 1 | -0.0777 | -0.2855 | 0.5014 | -0.4266 | -0.4144 | -0.413 | 0.413 | -0.4226 | -0.4099 | 0.0183 | 0.0941 |
| reasoning_tok_count | 0.2748 | -0.0777 | 1 | 0.4137 | 0.0608 | 0.4507 | 0.2489 | 0.2375 | -0.2375 | 0.4638 | 0.0996 | 0.1034 | 0.0378 |
| answer_tok_count | 0.5824 | -0.2855 | 0.4137 | 1 | -0.6424 | 0.4103 | 0.6248 | 0.2772 | -0.2772 | 0.4184 | 0.1893 | 0.166 | -0.0309 |
| reasoning_ratio | -0.7672 | 0.5014 | 0.0608 | -0.6424 | 1 | -0.2229 | -0.67 | -0.2482 | 0.2482 | -0.2214 | -0.2466 | -0.1416 | 0.0624 |
| retrieval_reasoning_lexical_overlap | 0.3362 | -0.4266 | 0.4507 | 0.4103 | -0.2229 | 1 | 0.645 | 0.9127 | -0.9127 | 0.993 | 0.8214 | 0.215 | 0.1743 |
| reasoning_answer_overlap | 0.6121 | -0.4144 | 0.2489 | 0.6248 | -0.67 | 0.645 | 1 | 0.6831 | -0.6831 | 0.6651 | 0.6445 | 0.2749 | 0.1264 |
| cot_cov_overall | 0.2968 | -0.413 | 0.2375 | 0.2772 | -0.2482 | 0.9127 | 0.6831 | 1 | -1 | 0.9244 | 0.9517 | 0.2873 | 0.1947 |
| cot_novelty | -0.2968 | 0.413 | -0.2375 | -0.2772 | 0.2482 | -0.9127 | -0.6831 | -1 | 1 | -0.9244 | -0.9517 | -0.2873 | -0.1947 |
| cot_retrieval_jaccard | 0.3431 | -0.4226 | 0.4638 | 0.4184 | -0.2214 | 0.993 | 0.6651 | 0.9244 | -0.9244 | 1 | 0.8305 | 0.2385 | 0.1738 |
| cot_chunk_max_cov | 0.282 | -0.4099 | 0.0996 | 0.1893 | -0.2466 | 0.8214 | 0.6445 | 0.9517 | -0.9517 | 0.8305 | 1 | 0.1976 | 0.203 |
| cot_chunk_entropy | 0.1509 | 0.0183 | 0.1034 | 0.166 | -0.1416 | 0.215 | 0.2749 | 0.2873 | -0.2873 | 0.2385 | 0.1976 | 1 | 0.481 |
| cot_chunks_used | 0.0435 | 0.0941 | 0.0378 | -0.0309 | 0.0624 | 0.1743 | 0.1264 | 0.1947 | -0.1947 | 0.1738 | 0.203 | 0.481 | 1 |

## Correlações: Shuffled

| variavel | f1_vs_gabarito | doubt_count | reasoning_tok_count | answer_tok_count | reasoning_ratio | retrieval_reasoning_lexical_overlap | reasoning_answer_overlap | cot_cov_overall | cot_novelty | cot_retrieval_jaccard | cot_chunk_max_cov | cot_chunk_entropy | cot_chunks_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito | 1 | -0.5279 | 0.3322 | 0.5128 | -0.7623 | 0.37 | 0.6353 | 0.3287 | -0.3287 | 0.3591 | 0.3537 | 0.1374 | 0.1877 |
| doubt_count | -0.5279 | 1 | -0.2406 | -0.4104 | 0.4674 | -0.419 | -0.4802 | -0.4179 | 0.4179 | -0.4234 | -0.3955 | -0.2451 | -0.2295 |
| reasoning_tok_count | 0.3322 | -0.2406 | 1 | 0.4413 | -0.0843 | 0.5225 | 0.3382 | 0.3208 | -0.3208 | 0.5301 | 0.2247 | 0.1732 | 0.2235 |
| answer_tok_count | 0.5128 | -0.4104 | 0.4413 | 1 | -0.6224 | 0.1979 | 0.3802 | 0.0819 | -0.0819 | 0.2011 | 0.0454 | 0.0807 | 0.1141 |
| reasoning_ratio | -0.7623 | 0.4674 | -0.0843 | -0.6224 | 1 | -0.1325 | -0.5692 | -0.1182 | 0.1182 | -0.1303 | -0.1311 | -0.0416 | -0.0294 |
| retrieval_reasoning_lexical_overlap | 0.37 | -0.419 | 0.5225 | 0.1979 | -0.1325 | 1 | 0.7043 | 0.9263 | -0.9263 | 0.9944 | 0.8669 | 0.2923 | 0.198 |
| reasoning_answer_overlap | 0.6353 | -0.4802 | 0.3382 | 0.3802 | -0.5692 | 0.7043 | 1 | 0.7271 | -0.7271 | 0.7112 | 0.7252 | 0.2727 | 0.1523 |
| cot_cov_overall | 0.3287 | -0.4179 | 0.3208 | 0.0819 | -0.1182 | 0.9263 | 0.7271 | 1 | -1 | 0.9376 | 0.9631 | 0.367 | 0.2176 |
| cot_novelty | -0.3287 | 0.4179 | -0.3208 | -0.0819 | 0.1182 | -0.9263 | -0.7271 | -1 | 1 | -0.9376 | -0.9631 | -0.367 | -0.2176 |
| cot_retrieval_jaccard | 0.3591 | -0.4234 | 0.5301 | 0.2011 | -0.1303 | 0.9944 | 0.7112 | 0.9376 | -0.9376 | 1 | 0.8727 | 0.3163 | 0.2074 |
| cot_chunk_max_cov | 0.3537 | -0.3955 | 0.2247 | 0.0454 | -0.1311 | 0.8669 | 0.7252 | 0.9631 | -0.9631 | 0.8727 | 1 | 0.303 | 0.2219 |
| cot_chunk_entropy | 0.1374 | -0.2451 | 0.1732 | 0.0807 | -0.0416 | 0.2923 | 0.2727 | 0.367 | -0.367 | 0.3163 | 0.303 | 1 | 0.6805 |
| cot_chunks_used | 0.1877 | -0.2295 | 0.2235 | 0.1141 | -0.0294 | 0.198 | 0.1523 | 0.2176 | -0.2176 | 0.2074 | 0.2219 | 0.6805 | 1 |

## Médias Globais: Success Vs Fail

| is_success | cot_cov_overall | cot_novelty | cot_chunk_entropy | cot_chunks_used | retrieval_reasoning_lexical_overlap | doubt_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.29 | 0.71 | 1.5402 | 4.9459 | 0.09 | 0.7838 |
| 1 | 0.4188 | 0.5812 | 1.5613 | 5 | 0.1477 | 0.1236 |

## Deltas (Shuffled - Original)

| variavel | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f1_vs_gabarito_delta | 0.0067 | 0.0768 | -0.2288 | -0.0108 | 0 | 0.0237 | 0.2117 |
| cot_cov_overall_delta | -0.0208 | 0.1764 | -0.5187 | -0.0522 | -0.0148 | 0.0243 | 0.4594 |
| cot_novelty_delta | 0.0208 | 0.1764 | -0.4594 | -0.0243 | 0.0148 | 0.0522 | 0.5187 |
| cot_chunk_entropy_delta | -0.014 | 0.0748 | -0.5002 | -0.0265 | 0 | 0.0104 | 0.1414 |
| retrieval_reasoning_lexical_overlap_delta | -0.0065 | 0.0727 | -0.233 | -0.0263 | -0.003 | 0.0095 | 0.2433 |
| reasoning_answer_overlap_delta | -0.0023 | 0.1257 | -0.3068 | -0.0415 | 0 | 0.0335 | 0.3585 |
| doubt_count_delta | 0.04 | 0.9312 | -4 | 0 | 0 | 0 | 3 |
| sent_support_rate_delta | 0.0052 | 0.1587 | -0.4286 | -0.0356 | 0 | 0.0411 | 0.4487 |

## Correlação Dos Deltas Com Δf1

| variavel | f1_vs_gabarito_delta |
| --- | --- |
| f1_vs_gabarito_delta | 1 |
| reasoning_answer_overlap_delta | 0.5839 |
| retrieval_reasoning_lexical_overlap_delta | 0.2178 |
| cot_cov_overall_delta | 0.1949 |
| sent_support_rate_delta | 0.1571 |
| cot_chunk_entropy_delta | -0.0387 |
| cot_novelty_delta | -0.1949 |
| doubt_count_delta | -0.3145 |

## Suporte De Sentenças Por Modelo E Sucesso

| model | is_success | n | f1_mean | sent_support_rate_mean | sent_low_support_rate_mean | sent_mean_max_sim_mean |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0 | 58 | 0.3981 | 0.1073 | 0.7628 | 0.0666 |
| original | 1 | 42 | 0.5524 | 0.2049 | 0.5239 | 0.1196 |
| shuffled | 0 | 53 | 0.3961 | 0.1137 | 0.7766 | 0.0639 |
| shuffled | 1 | 47 | 0.5525 | 0.1984 | 0.5862 | 0.1103 |
