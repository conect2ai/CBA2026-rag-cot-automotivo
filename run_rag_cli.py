"""
Interface de linha de comando para executar geração aumentada por recuperação
(RAG) sobre um conjunto de perguntas.

Este script permite especificar um modelo HuggingFace para geração e selecionar
um arquivo JSON com uma lista de perguntas. Para cada pergunta, ele recupera
contexto relevante de uma coleção Milvus, monta o prompt, gera uma resposta
(opcionalmente extraindo a seção de raciocínio) e salva os resultados em JSON.

Exemplo de uso:

    python run_rag_cli.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --input perguntas.json \
        --output resultados.json \
        --brand Renault \
        --car_model Kwid \
        --year 2024 \
        --top_k 5 \
        --max_new_tokens 250 \
        --backend mlx

O JSON de entrada deve ser uma lista de objetos com pelo menos o campo
"pergunta". Campos adicionais são ignorados. O JSON de saída contém a pergunta
original, o raciocínio e a resposta gerados, além dos chunks de contexto
recuperados para montar o prompt.

python run_rag_cli.py --backend mlx --model mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --input perguntas_respostas_Kwid_Nov24_25.json --output rag_cli_output_mlx-community_DeepSeek-R1-Distill-Qwen-7B-4bit_Kwid_2024_25.json --brand Renault --car_model Kwid --year 2024 --max_new_tokens 250
"""

import argparse
import random
import copy
import json
import time
from pathlib import Path

from rag_hf_optimized import (
    load_model,
    load_mlx_model,
    build_prompt,
    retrieve_chunks,
    generation_answer,
    generation_answer_mlx,
    MODEL_NAME as DEFAULT_MODEL_NAME,
    build_filename_from_model,
)

DEFAULT_MLX_MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Executa RAG em um conjunto de perguntas de um arquivo JSON.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome ou caminho do modelo generativo. O padrão depende do backend.",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "mlx"],
        default="transformers",
        help="Backend de geração. Use 'mlx' para Apple Silicon com modelos MLX.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para um arquivo JSON com uma lista de perguntas. Cada item deve ter a chave 'pergunta'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Caminho para salvar o JSON de resultados. Por padrão, usa 'results.json'.",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default=None,
        help="Filtro de marca do veículo para a recuperação (opcional).",
    )
    parser.add_argument(
        "--car_model",
        type=str,
        default=None,
        help="Filtro de modelo do veículo para a recuperação (opcional).",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Filtro de ano do veículo para a recuperação (opcional).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Número de chunks de contexto a recuperar para cada pergunta. O padrão é 5.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=250,
        help="Número máximo de tokens novos por geração. O padrão é 250 para reduzir o tempo de execução local.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processa apenas as N primeiras perguntas não processadas. Útil para medir tempo antes da execução completa.",
    )
    parser.add_argument(
        "--allow_empty_context",
        action="store_true",
        help="Permite gerar resposta mesmo quando nenhum chunk é recuperado. Por padrão, a execução para para evitar alucinações.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model or (DEFAULT_MLX_MODEL_NAME if args.backend == "mlx" else DEFAULT_MODEL_NAME)

    # Carrega o modelo e o tokenizer especificados.
    print(f"Carregando o modelo '{model_name}' com backend '{args.backend}'…")
    if args.backend == "mlx":
        tokenizer, model = load_mlx_model(model_name)
        generate_fn = generation_answer_mlx
    else:
        tokenizer, model = load_model(model_name)
        generate_fn = generation_answer
    print("Modelo carregado.")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("O JSON de entrada deve ser uma lista de objetos com um campo 'pergunta'.")

    # Define o caminho de saída antes de processar as perguntas. Se o usuário
    # mantiver o padrão 'results.json', gera um nome único com o modelo e um
    # timestamp para evitar sobrescrever execuções anteriores.
    if args.output == "results.json":
        auto_name = build_filename_from_model(model_name, prefix=f"rag_cli_output_{args.backend}")
        output_path = Path(auto_name)
    else:
        output_path = Path(args.output)

    # Se o arquivo de saída já existir, carrega resultados prévios para retomar.
    results: list = []
    processed_questions: set[str] = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f_existing:
                existing = json.load(f_existing)
            if isinstance(existing, list):
                results = existing
                processed_questions = {rec.get("question") for rec in existing if isinstance(rec, dict)}
            else:
                # Se o arquivo existente não for uma lista, ignora e recomeça.
                results = []
                processed_questions = set()
        except Exception:
            # Se a leitura falhar, assume que não há resultados anteriores.
            results = []
            processed_questions = set()

    processed_this_run = 0
    total_items = len(data)
    for idx, item in enumerate(data, 1):
        if args.limit is not None and processed_this_run >= args.limit:
            break
        question = item.get("pergunta") or item.get("question")
        if not question:
            # Ignora itens sem uma pergunta válida.
            continue
        # Ignora perguntas que já foram processadas.
        if question in processed_questions:
            continue
        question_start = time.perf_counter()
        print(f"[{idx}/{total_items}] Processando pergunta: {question}", flush=True)
        # Recupera o contexto na ordem original.
        retrieved = retrieve_chunks(
            question,
            k=args.top_k,
            brand=args.brand,
            model_name=args.car_model,
            year=args.year,
        )
        print(f"  - Chunks recuperados: {len(retrieved)}", flush=True)
        if not retrieved and not args.allow_empty_context:
            raise RuntimeError(
                "Nenhum chunk foi recuperado para a pergunta. "
                f"Verifique se a coleção Milvus contém documentos com filtros "
                f"brand={args.brand!r}, model={args.car_model!r}, year={args.year!r}."
            )
        # Monta o prompt e gera a resposta usando a ordem original do contexto.
        chunk_texts = [r["text"] for r in retrieved]
        prompt_original = build_prompt(question, chunk_texts)
        print(f"  - Gerando resposta original (max_new_tokens={args.max_new_tokens})...", flush=True)
        gen_start = time.perf_counter()
        gen_original = generate_fn(
            prompt_original,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  - Original concluída em {time.perf_counter() - gen_start:.1f}s.", flush=True)

        # Gera outra resposta usando uma versão embaralhada controlada do mesmo contexto.
        rng = random.Random(42 + idx)
        retrieved_shuffled = retrieved[:]
        rng.shuffle(retrieved_shuffled)

        shuffled_texts = [r["text"] for r in retrieved_shuffled]
        prompt_shuffled = build_prompt(question, shuffled_texts)
        print(f"  - Gerando resposta embaralhada (max_new_tokens={args.max_new_tokens})...", flush=True)
        gen_start = time.perf_counter()

        gen_shuffled = generate_fn(
            prompt_shuffled,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  - Embaralhada concluída em {time.perf_counter() - gen_start:.1f}s.", flush=True)
        # Monta o registro da pergunta com as métricas completas de geração.
        # Copia os resultados para acrescentar campos sem afetar iterações futuras.
        orig_result = copy.deepcopy(gen_original)
        shuf_result = copy.deepcopy(gen_shuffled)
        # Anexa os chunks de contexto usados em cada variante.
        orig_result["retrieved_chunks"] = retrieved
        shuf_result["retrieved_chunks"] = retrieved_shuffled
        result_record = {
            "question": question,
            # Preserva uma resposta original fornecida, quando houver.
            "original_answer": item.get("resposta") or item.get("answer"),
            "original": orig_result,
            "shuffled": shuf_result,
        }
        results.append(result_record)
        processed_questions.add(question)
        # Persiste incrementalmente os resultados após cada nova pergunta.
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)
        processed_this_run += 1
        print(f"  - Pergunta concluída em {time.perf_counter() - question_start:.1f}s.", flush=True)

    # Os resultados já foram persistidos incrementalmente; imprime a mensagem final.
    print(f"Concluído. Resultados gravados em {output_path}")


if __name__ == "__main__":
    main()
