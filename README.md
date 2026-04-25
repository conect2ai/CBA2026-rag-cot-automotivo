&nbsp;
&nbsp;

<p align="center">
  <img width="800" src="./figures/conect2ai_logo.jpg" alt="Conect2AI">
</p>

# Cadeia de Raciocínio em Sistemas de Geração Baseados em Recuperação: Influência da Organização do Contexto

### Autores: [Thaís Medeiros](https://github.com/thaisaraujom), [Marianne Silva](https://github.com/MarianneDiniz), [Ivanovitch Silva](https://github.com/ivanovitchm)

Este repositório reúne um estudo de caso sobre RAG
(Retrieval-Augmented Generation) para responder perguntas técnicas sobre o
manual do Volkswagen Polo 2025. O fluxo transforma o manual em uma base textual
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
4. As respostas são comparadas nos cenários `original` e `shuffled`.

Para cada pergunta, o script gera duas respostas:

- `original`: usa os chunks na ordem retornada pela busca vetorial no Milvus.
- `shuffled`: usa os mesmos chunks, mas com a ordem embaralhada.

## Estrutura do Repositório

```text
.
├── figures/
│   └── conect2ai_logo.jpg
├── img/
│   ├── cot_boxplot_model1.pdf
│   ├── cot_boxplot_model2.pdf
│   ├── cot_heatmap_model1.pdf
│   └── cot_heatmap_model2.pdf
├── markdown_manuals/
│   └── Volkswagen_Polo_2025.md
├── Volkswagen_Polo_2025.pdf
├── preprocessing.ipynb
├── perguntas_respostas_Polo_2025.json
├── rag_hf_optimized.py
├── run_rag_cli.py
├── analises-modelo1.ipynb
├── analises-modelo2.ipynb
├── tabelas-analise-modelo1.md
├── tabelas-analise-modelo2.md
├── rag_cli_output_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_20260312_145726.json
├── rag_cli_output_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_20260311_104139.json
└── requirements.txt
```

## Arquivos

- `Volkswagen_Polo_2025.pdf`: manual original usado como base de conhecimento.
- `preprocessing.ipynb`: conversão, chunking, embeddings e ingestão no Milvus.
- `markdown_manuals/Volkswagen_Polo_2025.md`: versão em Markdown do manual.
- `perguntas_respostas_Polo_2025.json`: perguntas e respostas de referência.
- `rag_hf_optimized.py`: funções de recuperação, geração e extração de CoT.
- `run_rag_cli.py`: execução do RAG em lote.
- `rag_cli_output_*.json`: resultados gerados pelos modelos.
- `analises-modelo1.ipynb` e `analises-modelo2.ipynb`: análise dos resultados.
- `tabelas-analise-modelo1.md` e `tabelas-analise-modelo2.md`: tabelas exportadas das análises.
- `img/`: figuras geradas nas análises.

## Resultados

Os resultados comparam dois modelos em dois cenários: `original`, com os chunks
na ordem da recuperação, e `shuffled`, com os mesmos chunks em ordem
embaralhada. A diferença entre os cenários é apenas a organização do contexto.

| Identificação | Modelo | Perguntas | Cenários |
| --- | --- | ---: | --- |
| Modelo 1 | `DeepSeek-R1-Distill-Llama-8B` | 100 | `original`, `shuffled` |
| Modelo 2 | `DeepSeek-R1-Distill-Qwen-7B` | 100 | `original`, `shuffled` |

### Desempenho

| Modelo | Cenário | F1 médio | Mediana F1 | Sucesso |
| --- | --- | ---: | ---: | ---: |
| Modelo 1 | Original | 0,47 | 0,47 | 0,42 |
| Modelo 1 | Shuffled | 0,48 | 0,50 | 0,49 |
| Modelo 2 | Original | 0,46 | 0,47 | 0,42 |
| Modelo 2 | Shuffled | 0,47 | 0,50 | 0,47 |

O cenário embaralhado não produziu queda nas métricas finais. Nos dois modelos,
as métricas agregadas permaneceram próximas entre os cenários, com variações
discretas e, em alguns casos, leve aumento.

### Métricas da CoT

| Modelo | Cenário | Cobertura | Novidade | Entropia | Chunks usados |
| --- | --- | ---: | ---: | ---: | ---: |
| Modelo 1 | Original | 0,36 | 0,64 | 1,56 | 4,98 |
| Modelo 1 | Shuffled | 0,33 | 0,67 | 1,54 | 4,95 |
| Modelo 2 | Original | 0,36 | 0,64 | 1,56 | 4,98 |
| Modelo 2 | Shuffled | 0,34 | 0,66 | 1,54 | 4,96 |

Com o contexto embaralhado, a cobertura da CoT diminui e a novidade aumenta. A
quantidade média de chunks usados permanece próxima de cinco, indicando que o
raciocínio continua distribuído pelo contexto, mas com menor apoio lexical nos
trechos recuperados.

### Variação Média (`shuffled - original`)

| Modelo | Δ F1 | Δ Cobertura CoT | Δ Novidade | Δ Entropia | Δ Support Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Modelo 1 | 0,01 | -0,03 | 0,03 | -0,02 | -0,02 |
| Modelo 2 | 0,01 | -0,02 | 0,02 | -0,01 | 0,01 |

O efeito mais claro aparece na estrutura da CoT, não no desempenho final: as
respostas permanecem próximas em qualidade, mas o raciocínio apresenta menor
cobertura em relação ao contexto recuperado.

### Acertos vs. Erros

| Modelo | Tipo | Cobertura CoT | Novidade | Doubt Count |
| --- | --- | ---: | ---: | ---: |
| Modelo 1 | Erro | 0,29 | 0,71 | 0,63 |
| Modelo 1 | Acerto | 0,42 | 0,58 | 0,18 |
| Modelo 2 | Erro | 0,29 | 0,71 | 0,78 |
| Modelo 2 | Acerto | 0,42 | 0,58 | 0,12 |

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

1. Converte `Volkswagen_Polo_2025.pdf` para Markdown.
2. Gera `markdown_manuals/Volkswagen_Polo_2025.md`.
3. Divide o manual em chunks.
4. Gera embeddings com `sentence-transformers/all-MiniLM-L6-v2`.
5. Cria ou reutiliza a coleção `manuals_open` no Milvus.
6. Insere os chunks vetorizados na coleção.

Esse passo precisa ser feito antes do RAG se a coleção `manuals_open` ainda não
existir.

### 5. Conferir o arquivo de perguntas

O arquivo de entrada é:

```text
perguntas_respostas_Polo_2025.json
```

Ele deve conter uma lista de objetos com o campo `pergunta`. O campo `resposta`
é usado como referência e aparece depois nos resultados como `original_answer`.

### 6. Rodar o RAG

Para o modelo Qwen 7B:

```bash
python run_rag_cli.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --input perguntas_respostas_Polo_2025.json \
  --output rag_cli_output_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B.json \
  --brand Volkswagen \
  --car_model Polo \
  --year 2025 \
  --top_k 5
```

Para o modelo Llama 8B:

```bash
python run_rag_cli.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --input perguntas_respostas_Polo_2025.json \
  --output rag_cli_output_deepseek-ai_DeepSeek-R1-Distill-Llama-8B.json \
  --brand Volkswagen \
  --car_model Polo \
  --year 2025 \
  --top_k 5
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
  posição; no embaralhado a cobertura fica mais homogênea ao longo dos chunks.