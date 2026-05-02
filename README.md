&nbsp;
&nbsp;

<p align="center">
  <img width="800" src="./figures/conect2ai_logo.jpg" alt="Conect2AI">
</p>

# Raciocínio em Sistemas RAG para Assistência Veicular: Impacto da Organização do Contexto em Ambientes de Automação Inteligente

### Autores: [Thaís Medeiros](https://github.com/thaisaraujom), [Marianne Silva](https://github.com/MarianneDiniz), [Ivanovitch Silva](https://github.com/ivanovitchm)

Este repositório reúne um estudo de caso sobre RAG
(Retrieval-Augmented Generation) para responder perguntas técnicas sobre o
manual do Renault Kwid 2024. O fluxo transforma o manual em uma base textual
pesquisável, indexa os trechos em um banco vetorial e compara respostas geradas
com o mesmo contexto em duas organizações diferentes.

A ideia principal consiste em recuperar os chunks mais relevantes para cada
pergunta e observar se a ordem desses chunks influencia a resposta final e a
cadeia de raciocínio produzida pelo modelo.

## Visão Geral

O pipeline segue quatro etapas:

1. O PDF do manual é convertido para Markdown.
2. O texto é segmentado em chunks e indexado no Milvus.
3. O RAG recupera os chunks mais relevantes para cada pergunta.
4. As respostas são comparadas nos cenários original e embaralhado.

Para cada pergunta, o script gera duas respostas:

- Original: usa os chunks na ordem retornada pela busca vetorial no Milvus.
- Embaralhado: usa os mesmos chunks, mas com a ordem embaralhada.

## Estrutura do Repositório

```text
.
├── figures/
│   └── conect2ai_logo.jpg
├── img/
│   ├── cot_boxplot_model1.pdf
│   ├── cot_boxplot_model2.pdf
│   ├── cot_delta_model1.pdf
│   ├── cot_delta_model2.pdf
│   ├── cot_heatmap_model1.pdf
│   ├── cot_heatmap_model2.pdf
│   ├── cot_success_model1.pdf
│   └── cot_success_model2.pdf
├── markdown_manuals/
│   └── Renault_Kwid_Nov24.md
├── Renault_Kwid_2024.pdf
├── preprocessing.ipynb
├── perguntas_respostas_Kwid_Nov24_50.json
├── rag_hf_optimized.py
├── run_rag_cli.py
├── analises-modelo1.ipynb
├── analises-modelo2.ipynb
├── tabelas-analise-modelo1.md
├── tabelas-analise-modelo2.md
├── rag_cli_output_mlx_DeepSeek-R1-Distill-Llama-8B-4bit_Kwid_Nov24_50.json
├── rag_cli_output_mlx_DeepSeek-R1-Distill-Qwen-7B-4bit_Kwid_Nov24_50.json
├── logs_rag_cli_mlx_deepseek_kwid_nov24_50.json
├── logs_rag_cli_mlx_deepseek_llama8b_kwid_nov24_50.json
└── requirements.txt
```

## Arquivos

- `Renault_Kwid_2024.pdf`: manual original usado como base de conhecimento.
- `preprocessing.ipynb`: conversão, chunking, embeddings e ingestão no Milvus.
- `markdown_manuals/Renault_Kwid_Nov24.md`: versão em Markdown do manual.
- `perguntas_respostas_Kwid_Nov24_50.json`: perguntas e respostas de referência.
- `rag_hf_optimized.py`: funções de recuperação, geração e extração de CoT.
- `run_rag_cli.py`: execução do RAG em lote.
- `rag_cli_output_*.json`: resultados gerados pelos modelos.
- `logs_rag_cli_*.json`: registros de execução dos experimentos.
- `analises-modelo1.ipynb` e `analises-modelo2.ipynb`: análise dos resultados.
- `tabelas-analise-modelo1.md` e `tabelas-analise-modelo2.md`: tabelas exportadas das análises.
- `img/`: figuras geradas nas análises.

## Resultados

Os resultados comparam dois modelos em dois cenários: original, com os chunks
na ordem da recuperação, e embaralhado, com os mesmos chunks em ordem
embaralhada. A diferença entre os cenários é apenas a organização do contexto.

| Identificação | Modelo | Perguntas | Cenários |
| --- | --- | ---: | --- |
| Modelo 1 | `DeepSeek-R1-Distill-Llama-8B-4bit` | 50 | Original, Embaralhado |
| Modelo 2 | `DeepSeek-R1-Distill-Qwen-7B-4bit` | 50 | Original, Embaralhado |

### Desempenho

| Modelo | Cenário | F1 médio | Mediana F1 | Sucesso |
| --- | --- | ---: | ---: | ---: |
| Modelo 1 | Original | 0,66 | 0,64 | 0,96 |
| Modelo 1 | Embaralhado | 0,65 | 0,65 | 0,88 |
| Modelo 2 | Original | 0,62 | 0,63 | 0,84 |
| Modelo 2 | Embaralhado | 0,64 | 0,64 | 0,86 |

O efeito do embaralhamento não foi uniforme entre os modelos. O Modelo 1
apresentou leve queda no F1 médio e na taxa de sucesso, enquanto o Modelo 2 teve
pequeno aumento nas métricas finais.

### Métricas da CoT

| Modelo | Cenário | Cobertura | Novidade | Entropia | Chunks usados |
| --- | --- | ---: | ---: | ---: | ---: |
| Modelo 1 | Original | 0,17 | 0,83 | 2,02 | 7,96 |
| Modelo 1 | Embaralhado | 0,14 | 0,86 | 1,99 | 7,86 |
| Modelo 2 | Original | 0,53 | 0,47 | 2,05 | 8,00 |
| Modelo 2 | Embaralhado | 0,56 | 0,44 | 2,05 | 8,00 |

No Modelo 1, o cenário embaralhado reduz a cobertura da CoT e aumenta a
novidade. No Modelo 2, ocorre o movimento inverso, com maior cobertura e menor
novidade quando os mesmos trechos são reorganizados.

### Variação Média (embaralhado - original)

| Modelo | Δ F1 | Δ Cobertura CoT | Δ Novidade | Δ Entropia | Δ Support Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modelo 1 | -0,01 | -0,04 | 0,04 | -0,03 | -0,01 |
| Modelo 2 | 0,02 | 0,04 | -0,04 | 0,00 | 0,03 |

As variações reforçam que a ordem dos trechos afeta os modelos em direções
distintas. A queda de cobertura no Modelo 1 acompanha uma pequena redução no
desempenho, enquanto o aumento de cobertura no Modelo 2 coincide com melhora nas
métricas finais.

### Acertos vs. Erros

| Modelo | Tipo | Cobertura CoT | Novidade | Doubt Count |
| --- | --- | ---: | ---: | ---: |
| Modelo 1 | Erro | 0,09 | 0,91 | 0,63 |
| Modelo 1 | Acerto | 0,16 | 0,84 | 0,25 |
| Modelo 2 | Erro | 0,47 | 0,53 | 0,47 |
| Modelo 2 | Acerto | 0,56 | 0,44 | 0,02 |

Respostas corretas apresentam maior cobertura da CoT e menor novidade. Já
respostas incorretas apresentam mais marcadores de incerteza (`doubt count`).

## Como Rodar

### 1. Criar o ambiente Python

Crie o ambiente virtual, ative-o e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No VS Code, selecione esse ambiente como interpretador Python e use o kernel
associado ao `.venv` ao executar os notebooks.

### 2. Subir o Milvus com Docker

O projeto espera encontrar o Milvus em:

```text
http://localhost:19531
```

Depois de subir o container do Milvus, confira se ele está ativo:

```bash
docker ps
```

Se sua instância estiver em outra porta, ajuste `MILVUS_URI` em
`rag_hf_optimized.py` e a conexão no `preprocessing.ipynb`.

### 3. Fazer login no Hugging Face

Os modelos são baixados do Hugging Face. Se necessário, autentique:

```bash
huggingface-cli login
```

### 4. Executar o pré-processamento no VS Code

Abra `preprocessing.ipynb` no VS Code e execute as células em ordem. Esse passo:

1. Converte `Renault_Kwid_2024.pdf` para Markdown.
2. Gera `markdown_manuals/Renault_Kwid_Nov24.md`.
3. Divide o manual em chunks.
4. Gera embeddings com `sentence-transformers/all-MiniLM-L6-v2`.
5. Cria ou reutiliza a coleção `manuals_open` no Milvus.
6. Insere os chunks vetorizados na coleção.

Esse passo precisa ser feito antes do RAG se a coleção `manuals_open` ainda não
existir.

### 5. Conferir o arquivo de perguntas

O arquivo de entrada é:

```text
perguntas_respostas_Kwid_Nov24_50.json
```

Ele deve conter uma lista de objetos com o campo `pergunta`. O campo `resposta`
é usado como referência e aparece depois nos resultados como `original_answer`.

### 6. Rodar o RAG

Para o modelo Qwen 7B com backend MLX, usado nas execuções em Apple Silicon:

```bash
python run_rag_cli.py \
  --backend mlx \
  --model mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit \
  --input perguntas_respostas_Kwid_Nov24_50.json \
  --output rag_cli_output_mlx_DeepSeek-R1-Distill-Qwen-7B-4bit_Kwid_Nov24_50.json \
  --brand Renault \
  --car_model Kwid \
  --year 2024 \
  --top_k 8 \
  --max_new_tokens 1200
```

Para o modelo Llama 8B com backend MLX:

```bash
python run_rag_cli.py \
  --backend mlx \
  --model mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit \
  --input perguntas_respostas_Kwid_Nov24_50.json \
  --output rag_cli_output_mlx_DeepSeek-R1-Distill-Llama-8B-4bit_Kwid_Nov24_50.json \
  --brand Renault \
  --car_model Kwid \
  --year 2024 \
  --top_k 8 \
  --max_new_tokens 1200
```

O script salva os resultados incrementalmente. Se a execução parar no meio, rode
o mesmo comando de novo com o mesmo `--output`. Assim, as perguntas já processadas são
ignoradas.

### 7. Rodar as análises

Abra no VS Code:

```text
analises-modelo1.ipynb
analises-modelo2.ipynb
```

Eles leem os arquivos `rag_cli_output_*.json`, calculam as métricas e geram as
figuras em `img/`.

## Formato dos Resultados

Cada item dos arquivos `rag_cli_output_*.json` tem esta estrutura:

```json
{
  "question": "...",
  "original_answer": "...",
  "original": {},
  "shuffled": {}
}
```

Campos principais dentro de `original` e `shuffled`:

- `raw_output`: saída bruta do modelo.
- `clean_raw_output`: saída limpa de tokens especiais.
- `reasoning_text`: trecho identificado como cadeia de raciocínio.
- `answer_text`: resposta final extraída.
- `retrieved_chunks`: chunks usados como contexto.
- `input_token_count`: quantidade de tokens no prompt.
- `output_token_count`: quantidade de tokens gerados.
- `tokenizer_info`: metadados do tokenizer usado.

## Análises

As análises ficam em:

- `analises-modelo1.ipynb`: calcula métricas e visualizações para o Modelo 1.
- `analises-modelo2.ipynb`: calcula métricas e visualizações para o Modelo 2.

As tabelas exportadas ficam em:

- `tabelas-analise-modelo1.md`: tabelas da análise do Modelo 1.
- `tabelas-analise-modelo2.md`: tabelas da análise do Modelo 2.

As principais figuras geradas estão em:

- `img/cot_boxplot_model1.pdf`: cobertura da cadeia de raciocínio no Modelo 1.
- `img/cot_boxplot_model2.pdf`: cobertura da cadeia de raciocínio no Modelo 2.
- `img/cot_heatmap_model1.pdf`: cobertura por posição dos chunks no Modelo 1.
- `img/cot_heatmap_model2.pdf`: cobertura por posição dos chunks no Modelo 2.

Leitura visual:

- Boxplots de CoT: respostas corretas tendem a ter maior cobertura da CoT no
  contexto.
- Heatmaps de chunks: no cenário original há leve concentração na primeira
  posição; no embaralhado, a cobertura fica mais homogênea ao longo dos chunks.
