"""
Funções auxiliares otimizadas para executar RAG com modelos HuggingFace.

Este módulo consolida a lógica do notebook original em um único script,
remove funções duplicadas ou não utilizadas e melhora a extração de
raciocínio e resposta ao interpretar as saídas geradas pelo modelo.

Principais melhorias:

* Há uma única implementação de `build_filename_from_model`, que adiciona
  um timestamp aos nomes dos arquivos gerados.
* `detect_reasoning_tag_from_text` faz uma busca sem diferenciar maiúsculas
  de minúsculas pelo primeiro par compatível de tags no estilo HTML, o que
  torna a detecção mais robusta para tags com capitalização mista, como
  `<analysis>` e `<Analysis>`.
* `generation_answer` inclui uma estratégia de fallback: se não encontrar
  um par completo de tags, procura uma tag de fechamento de raciocínio, como
  `</think>` ou `</analysis>`, na saída bruta e separa o texto nesse ponto.
  Isso ajuda a distinguir o raciocínio da resposta final mesmo quando apenas
  a tag de fechamento aparece. Se nenhuma tag for detectada, toda a saída é
  tratada como resposta.
* Funções auxiliares não usadas e definições duplicadas foram removidas.

Observação: as funções de embedding e recuperação dependem de uma instância
Milvus existente e mantêm o comportamento do notebook original.
"""

import random
import re
import json
from datetime import datetime
from typing import Tuple, Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient


# -----------------------------------------------------------------------------
# Inicialização do modelo e do tokenizer
# -----------------------------------------------------------------------------
# Nome padrão do modelo usado para geração. Ao usar as funções auxiliares
# programaticamente, esse valor pode ser sobrescrito carregando outro modelo
# com ``load_model`` e passando os objetos ``model`` e ``tokenizer`` resultantes
# para ``generation_answer``. Mantenha este valor como um padrão razoável.
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Carrega o tokenizer e o modelo padrão uma vez na importação do módulo. A
# função auxiliar ``load_model`` abaixo pode ser usada para carregar modelos
# diferentes sob demanda. O tokenizer e o modelo são carregados em meia
# precisão, e o modelo é alocado automaticamente no dispositivo adequado via
# ``device_map="auto"``.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

def load_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Carrega um modelo HuggingFace e seu tokenizer para geração.

    Parâmetros
    ----------
    model_name : str
        Nome ou caminho do modelo a carregar, compatível com
        ``transformers.AutoTokenizer.from_pretrained`` e
        ``transformers.AutoModelForCausalLM.from_pretrained``.

    Retorno
    -------
    tuple
        Par ``(tokenizer, model)`` carregado a partir do nome informado.

    Observações
    -----
    Esta função auxiliar carrega o modelo e o tokenizer em meia precisão e
    mapeia automaticamente o modelo para os dispositivos disponíveis. Ela pode
    ser usada por scripts de linha de comando para substituir o ``MODEL_NAME``
    padrão sem modificar as variáveis globais do módulo.
    """
    # Se o modelo solicitado for o padrão do módulo e tokenizer/modelo já
    # estiverem carregados, reutiliza os objetos existentes. Isso evita carregar
    # o mesmo checkpoint duas vezes (uma na importação e outra via load_model).
    global MODEL_NAME, tokenizer, model
    if model_name == MODEL_NAME and tokenizer is not None and model is not None:
        return tokenizer, model
    # Caso contrário, carrega o modelo especificado do zero.
    tk = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    return tk, mdl


# -----------------------------------------------------------------------------
# Construção do prompt
# -----------------------------------------------------------------------------
def build_prompt(question: str, chunks: List[str]) -> str:
    """Monta um prompt para o modelo de linguagem usando chunks recuperados.

    Parâmetros
    ----------
    question: str
        Pergunta do usuário.
    chunks: list of str
        Trechos de documentos recuperados para fornecer contexto.

    Retorno
    -------
    str
        Prompt formatado orientando o modelo a responder usando apenas o
        contexto fornecido.
    """
    context = "\n\n".join(f"[CHUNK {i}] {c}" for i, c in enumerate(chunks, 1))
    return (
        "Você é um assistente técnico automotivo. Use SOMENTE o contexto fornecido para responder à pergunta.\n\n"
        "Se o contexto não contiver a resposta, diga: \"O manual não fornece essa informação.\"\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Pergunta: {question}\nResposta:"
    )


# -----------------------------------------------------------------------------
# Detecção de tags de raciocínio
# -----------------------------------------------------------------------------
def detect_reasoning_tag_from_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Detecta o primeiro par de tags no estilo HTML em `text`.

    As tags são consideradas no formato `<tag>` ... `</tag>`. A busca não
    diferencia maiúsculas de minúsculas e retorna o nome da tag, além das
    strings literais de abertura e fechamento. Se nenhum par compatível for
    encontrado, retorna ``(None, None, None)``.

    Exemplos
    --------
    >>> detect_reasoning_tag_from_text("<think>reason</think>answer")
    ('think', '<think>', '</think>')
    >>> detect_reasoning_tag_from_text("<Analysis>test</analysis> ok")
    ('Analysis', '<Analysis>', '</analysis>')
    """
    # Busca sem diferenciar maiúsculas de minúsculas por um par abertura/fechamento.
    match = re.search(r"<([A-Za-z_][A-Za-z0-9_]*)>.*?</\1>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None, None, None
    tag_name = match.group(1)
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name.lower()}>"  # normaliza a tag de fechamento para minúsculas
    return tag_name, start_tag, end_tag


# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------
def find_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    """Encontra o índice inicial de ``subsequence`` dentro de ``sequence``.

    Retorna -1 se a subsequência não for encontrada. Esta função faz uma busca
    simples por janela deslizante e é usada para localizar tags tokenizadas na
    saída do modelo.
    """
    n = len(sequence)
    m = len(subsequence)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if sequence[i:i + m] == subsequence:
            return i
    return -1


def clean_generated_text(text: str) -> str:
    """Remove tokens especiais indesejados e espaços no início ou no fim."""
    if not text:
        return text
    text = text.replace("<｜end▁of▁sentence｜>", "")
    text = text.replace("<|end_of_text|>", "")
    return text.strip()


def build_filename_from_model(model_name: str, prefix: str = "rag_output") -> str:
    """Gera um nome de arquivo único para resultados de experimentos.

    O nome inclui uma versão limpa do nome do modelo e um timestamp.
    """
    model_clean = model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{model_clean}_{timestamp}.json"


def get_tokenizer_info(tokenizer, model_name: str) -> dict:
    """Coleta informações de diagnóstico sobre o tokenizer."""
    return {
        "tokenizer_name": getattr(tokenizer, "name_or_path", model_name),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_name_reference": model_name,
        "is_fast": getattr(tokenizer, "is_fast", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "pad_token": str(getattr(tokenizer, "pad_token", None)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token": str(getattr(tokenizer, "eos_token", None)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token": str(getattr(tokenizer, "bos_token", None)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "unk_token": str(getattr(tokenizer, "unk_token", None)),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_tokens": tokenizer.all_special_tokens,
        "all_special_ids": tokenizer.all_special_ids,
        "additional_special_tokens": getattr(tokenizer, "additional_special_tokens", []),
        "additional_special_tokens_ids": getattr(tokenizer, "additional_special_tokens_ids", []),
        "chat_template_exists": hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None,
        "chat_template": getattr(tokenizer, "chat_template", None),
        # Representação bruta para depuração.
        "tokenizer_raw_dump": repr(tokenizer),
        "added_tokens_decoder": {
            str(k): str(v) for k, v in getattr(tokenizer, "added_tokens_decoder", {}).items()
        },
    }


# -----------------------------------------------------------------------------
# Geração e extração de raciocínio
# -----------------------------------------------------------------------------
def generation_answer(
    prompt: str,
    *,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None
) -> Dict[str, object]:
    """Gera uma resposta com o modelo e separa raciocínio de resposta final.

    A função constrói um template de chat, executa a inferência do modelo e
    decodifica a saída. Em seguida, tenta extrair uma seção de raciocínio e a
    resposta final detectando tags na saída bruta. Quando encontra um par
    abertura/fechamento compatível, o texto entre as tags é tratado como
    raciocínio, e o texto após a tag de fechamento é tratado como resposta.

    Se nenhum par de tags for encontrado, a função procura uma tag de
    fechamento de raciocínio, como ``</think>`` ou ``</analysis>``; o texto
    anterior a essa tag é considerado raciocínio. Como último recurso, todo o
    texto gerado é considerado a resposta.

    Retorna um dicionário com a saída bruta e limpa, contagens de tokens,
    textos e tokens de raciocínio/resposta, nomes e posições das tags
    detectadas e diagnósticos do tokenizer.
    """
    # Usa o modelo e o tokenizer informados, quando existirem; caso contrário,
    # recorre aos padrões do módulo. Isso permite trocar dinamicamente o modelo
    # de geração por chamada.
    mdl = model if model is not None else globals()["model"]
    tk = tokenizer if tokenizer is not None else globals()["tokenizer"]

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tk.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Tokeniza a entrada.
    inputs = tk(formatted_prompt, return_tensors="pt", truncation=True)
    # Move os tensores para o mesmo dispositivo do modelo usado na geração.
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    # Executa a geração.
    output = mdl.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=1200,
        eos_token_id=tk.eos_token_id,
        pad_token_id=tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][prompt_len:]
    generated_ids_list = generated_ids.tolist()
    raw_output = tk.decode(generated_ids, skip_special_tokens=False).strip()
    clean_raw_output = clean_generated_text(
        tk.decode(generated_ids, skip_special_tokens=True).strip()
    )
    # Tenta detectar tags de raciocínio na saída bruta.
    reasoning_tag_name, found_start_tag, found_end_tag = detect_reasoning_tag_from_text(raw_output)
    found_start_tag_ids = []
    found_start_tag_start_idx = -1
    found_start_tag_end_idx = -1
    found_end_tag_ids = []
    found_end_tag_start_idx = -1
    found_end_tag_end_idx = -1
    # Localiza a tag de abertura.
    if found_start_tag:
        found_start_tag_ids = tk.encode(found_start_tag, add_special_tokens=False)
        idx = find_subsequence(generated_ids_list, found_start_tag_ids)
        if idx != -1:
            found_start_tag_start_idx = idx
            found_start_tag_end_idx = idx + len(found_start_tag_ids)
    # Localiza a tag de fechamento.
    if found_end_tag:
        found_end_tag_ids = tk.encode(found_end_tag, add_special_tokens=False)
        idx = find_subsequence(generated_ids_list, found_end_tag_ids)
        if idx != -1:
            found_end_tag_start_idx = idx
            found_end_tag_end_idx = idx + len(found_end_tag_ids)
    reasoning_ids: List[int] = []
    answer_ids: List[int] = generated_ids_list[:]
    # Caso 1: ambas as tags foram encontradas e estão na ordem correta.
    if (
        found_start_tag
        and found_end_tag
        and found_start_tag_end_idx != -1
        and found_end_tag_start_idx != -1
        and found_start_tag_end_idx <= found_end_tag_start_idx
    ):
        reasoning_ids = generated_ids_list[found_start_tag_end_idx:found_end_tag_start_idx]
        answer_ids = generated_ids_list[found_end_tag_end_idx:]
    # Caso 2: apenas a tag de fechamento foi encontrada.
    elif found_end_tag and found_end_tag_start_idx != -1:
        reasoning_ids = generated_ids_list[:found_end_tag_start_idx]
        answer_ids = generated_ids_list[found_end_tag_end_idx:]
    else:
        # Fallback: procura uma tag de fechamento de raciocínio no texto bruto.
        closing_tags = ["</think>", "</analysis>", "</thought>"]
        fallback_pos = -1
        fallback_tag = None
        for tag in closing_tags:
            pos = raw_output.lower().find(tag)
            if pos != -1:
                fallback_pos = pos
                fallback_tag = tag
                break
        if fallback_pos != -1:
            reasoning_text_raw = raw_output[:fallback_pos]
            answer_text_raw = raw_output[fallback_pos + len(fallback_tag):]
            reasoning_ids = tk.encode(reasoning_text_raw, add_special_tokens=False)
            answer_ids = tk.encode(answer_text_raw, add_special_tokens=False)
        else:
            # Nenhuma tag detectada; trata toda a saída como resposta.
            reasoning_ids = []
            answer_ids = generated_ids_list[:]
    # Decodifica raciocínio e resposta após tratar todos os casos.
    reasoning_text = tk.decode(reasoning_ids, skip_special_tokens=False).strip() if reasoning_ids else ""
    answer_text = tk.decode(answer_ids, skip_special_tokens=False).strip() if answer_ids else ""
    return {
        "raw_output": raw_output,
        "clean_raw_output": clean_raw_output,
        "formatted_prompt": formatted_prompt,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "output_token_count": int(generated_ids.shape[0]),
        "input_ids": inputs["input_ids"][0].tolist(),
        "generated_ids": generated_ids_list,
        "reasoning_text": clean_generated_text(reasoning_text),
        "reasoning_ids": reasoning_ids,
        "reasoning_tokens": tk.convert_ids_to_tokens(reasoning_ids) if reasoning_ids else [],
        "answer_text": clean_generated_text(answer_text),
        "answer_ids": answer_ids,
        "answer_tokens": tk.convert_ids_to_tokens(answer_ids) if answer_ids else [],
        "reasoning_tag_name": reasoning_tag_name,
        "reasoning_start_tag": found_start_tag,
        "reasoning_start_tag_ids": found_start_tag_ids,
        "reasoning_start_tag_tokens": tk.convert_ids_to_tokens(found_start_tag_ids) if found_start_tag_ids else [],
        "reasoning_start_tag_start_idx": found_start_tag_start_idx,
        "reasoning_start_tag_end_idx": found_start_tag_end_idx,
        "reasoning_end_tag": found_end_tag,
        "reasoning_end_tag_ids": found_end_tag_ids,
        "reasoning_end_tag_tokens": tk.convert_ids_to_tokens(found_end_tag_ids) if found_end_tag_ids else [],
        "reasoning_end_tag_start_idx": found_end_tag_start_idx,
        "reasoning_end_tag_end_idx": found_end_tag_end_idx,
        "tokenizer_info": get_tokenizer_info(tk, mdl.name_or_path if hasattr(mdl, "name_or_path") else MODEL_NAME),
    }


# -----------------------------------------------------------------------------
# Modelo de embeddings e recuperação
# -----------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)


def embed_text(texts: List[str]) -> List[List[float]]:
    """Converte uma lista de textos em vetores usando o modelo de embeddings."""
    return embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()


# Configura o cliente Milvus e os parâmetros da coleção.
MILVUS_URI = "http://localhost:19531"
COLLECTION = "manuals_open"
VECTOR_FIELD = "embedding"
TEXT_FIELD = "text"

milvus = MilvusClient(uri=MILVUS_URI)


def retrieve_chunks(
    question: str,
    k: int = 5,
    brand: Optional[str] = None,
    model_name: Optional[str] = None,
    year: Optional[str] = None
) -> List[Dict[str, object]]:
    """Recupera do Milvus os top-k chunks mais similares a uma pergunta."""
    qvec = embed_text([question])[0]
    filter_parts = []
    if brand:
        filter_parts.append(f'brand == "{brand}"')
    if model_name:
        filter_parts.append(f'model == "{model_name}"')
    if year:
        filter_parts.append(f'year == "{year}"')
    filter_expr = " and ".join(filter_parts) if filter_parts else None
    res = milvus.search(
        collection_name=COLLECTION,
        data=[qvec],
        anns_field=VECTOR_FIELD,
        limit=k,
        output_fields=[TEXT_FIELD, "brand", "model", "year", "source"],
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
    )
    hits = res[0]
    return [
        {
            "text": h["entity"][TEXT_FIELD],
            "score": h["distance"],
            "brand": h["entity"].get("brand"),
            "model": h["entity"].get("model"),
            "year": h["entity"].get("year"),
            "source": h["entity"].get("source"),
        }
        for h in hits
    ]


# -----------------------------------------------------------------------------
# Formatação e salvamento dos resultados
# -----------------------------------------------------------------------------
def build_result_record(
    question: str,
    prompt: str,
    generation_result: dict,
    variant: str,
    retrieved_chunks: List[Dict[str, object]],
) -> Dict[str, object]:
    """Constrói um registro estruturado a partir de uma geração."""
    return {
        "timestamp": datetime.now().isoformat(),
        "variant": variant,
        "question": question,
        "prompt": prompt,
        "formatted_prompt": generation_result["formatted_prompt"],
        "retrieved_chunks": [
            {
                "rank": i + 1,
                "score": item["score"],
                "source": item["source"],
                "brand": item["brand"],
                "model": item["model"],
                "year": item["year"],
                "text": item["text"],
            }
            for i, item in enumerate(retrieved_chunks)
        ],
        "raw_output": generation_result["raw_output"],
        "clean_raw_output": generation_result["clean_raw_output"],
        "reasoning": generation_result["reasoning_text"],
        "reasoning_ids": generation_result["reasoning_ids"],
        "reasoning_tokens": generation_result["reasoning_tokens"],
        "answer": generation_result["answer_text"],
        "answer_ids": generation_result["answer_ids"],
        "answer_tokens": generation_result["answer_tokens"],
        # Detalhes das tags de raciocínio.
        "reasoning_start_tag": generation_result["reasoning_start_tag"],
        "reasoning_start_tag_ids": generation_result["reasoning_start_tag_ids"],
        "reasoning_start_tag_tokens": generation_result["reasoning_start_tag_tokens"],
        "reasoning_start_tag_start_idx": generation_result["reasoning_start_tag_start_idx"],
        "reasoning_start_tag_end_idx": generation_result["reasoning_start_tag_end_idx"],
        "reasoning_end_tag": generation_result["reasoning_end_tag"],
        "reasoning_end_tag_ids": generation_result["reasoning_end_tag_ids"],
        "reasoning_end_tag_tokens": generation_result["reasoning_end_tag_tokens"],
        "reasoning_end_tag_start_idx": generation_result["reasoning_end_tag_start_idx"],
        "reasoning_end_tag_end_idx": generation_result["reasoning_end_tag_end_idx"],
        "tokenizer_info": generation_result["tokenizer_info"],
        "input_token_count": generation_result["input_token_count"],
        "output_token_count": generation_result["output_token_count"],
        "input_ids": generation_result["input_ids"],
        "generated_ids": generation_result["generated_ids"],
    }


def save_experiment_json(
    question: str,
    original_prompt: str,
    original_generation: dict,
    original_chunks: List[Dict[str, object]],
    shuffled_prompt: str,
    shuffled_generation: dict,
    shuffled_chunks: List[Dict[str, object]],
    filename: str,
) -> Dict[str, object]:
    """Salva em JSON os resultados dos contextos original e embaralhado."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "tokenizer_name": getattr(tokenizer, "name_or_path", MODEL_NAME),
        "question": question,
        "original": build_result_record(
            question=question,
            prompt=original_prompt,
            generation_result=original_generation,
            variant="original",
            retrieved_chunks=original_chunks,
        ),
        "shuffled": build_result_record(
            question=question,
            prompt=shuffled_prompt,
            generation_result=shuffled_generation,
            variant="shuffled",
            retrieved_chunks=shuffled_chunks,
        ),
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data


__all__ = [
    "build_prompt",
    "detect_reasoning_tag_from_text",
    "generation_answer",
    "embed_text",
    "retrieve_chunks",
    "build_result_record",
    "save_experiment_json",
    "build_filename_from_model",
]
