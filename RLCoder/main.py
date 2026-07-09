import os
import time
import json
import torch
import random
import argparse
import numpy as np

from generator import Generator
from bm25 import TaskSpecificBM25
from retriever import Retriever, tokenize
from datasets import load_test_dataset, load_train_and_valid_dataset, construct_dataset, CodeBlock
from context_query import build_base_query, build_query_bundle
from context_candidates import (
    add_retrieval_trace_results,
    recall_multi_path_candidates,
    log_retrieval_trace,
    write_retrieval_trace,
)
from context_gate import apply_candidate_gate, apply_retrieved_context_gate
from context_graph import add_graph_trace_results, expand_context_graph_candidates

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from utils.eval_metric import compute_metric_stmt
from utils.eval_codereval import eval_codereval
from utils.model_utils import local_model_path
from prettytable import PrettyTable
import copy

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set seed
def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()


# Retrieves code blocks based on different inference types.
def retrieve_codeblocks(args, examples, bm25, retriever, dataset_name, is_training=False, inference_type=None):
    """
    Retrieves code blocks based on different inference types.
    :param args: An argument object containing configuration parameters.
    :param examples: Examples used for retrieval.
    :param bm25: An instance of the BM25 model.
    :param retriever: An instance of the retriever.
    :param dataset_name: The name of the dataset.
    :param is_training: Whether it is in training mode.
    :return: A list of retrieved code blocks.
    """
    if inference_type is None:
        inference_type = args.inference_type
    if inference_type == "baseline":
        return None, [[] for _ in range(len(examples))]

    bm25_topk, unixcoder_topk, context_len = 5, 5, 20
    if inference_type in ["bm25", "unixcoder", "unixcoder_with_rl"]:
        if dataset_name not in bm25:
            bm25[dataset_name] = TaskSpecificBM25(examples, args)

        if inference_type == "unixcoder":
            bm25_topk = 50 
        elif inference_type == "unixcoder_with_rl":
            bm25_topk = args.sample_number * 10 
            unixcoder_topk = args.sample_number 

        base_queries = [build_base_query(example, context_len=context_len) for example in examples]
        queries = base_queries
        ucm_trace_rows = None

        enable_ucm_multi_path = (
            getattr(args, "enable_ucm", False)
            and getattr(args, "enable_multi_path_retrieval", False)
        )
        enable_ucm_context_graph = (
            getattr(args, "enable_ucm", False)
            and getattr(args, "enable_context_graph", False)
        )

        if enable_ucm_multi_path:
            draft_generations = None
            if args.enable_repocoder and inference_type == 'unixcoder_with_rl':
                draft_args = copy.deepcopy(args)
                draft_args.enable_ucm = False
                draft_args.enable_multi_path_retrieval = False
                _, retrieved_codeblocks = retrieve_codeblocks(draft_args, examples, bm25, retriever_RLCoder, dataset_name, inference_type="unixcoder")
                draft_generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)
                queries = [query + '\n' + prediction for query, prediction in zip(base_queries, draft_generations)]

            query_bundles = [
                build_query_bundle(
                    args,
                    example,
                    base_query=base_query,
                    draft_prediction=draft_generations[idx] if draft_generations else None,
                    context_len=context_len,
                )
                for idx, (example, base_query) in enumerate(zip(examples, base_queries))
            ]
            candidate_codeblocks, ucm_trace_rows = recall_multi_path_candidates(
                args,
                examples,
                bm25[dataset_name],
                query_bundles,
                base_topk=bm25_topk,
            )
        else:
            candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in examples], queries, topk=bm25_topk)

            if args.enable_repocoder and inference_type == 'unixcoder_with_rl':
                _, retrieved_codeblocks = retrieve_codeblocks(args, examples, bm25, retriever_RLCoder, dataset_name, inference_type="unixcoder")
                generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)

                queries = [query + '\n' + prediction for query, prediction in zip(queries, generations)]

        if enable_ucm_context_graph:
            candidate_codeblocks, graph_trace_rows = expand_context_graph_candidates(
                args,
                examples,
                bm25[dataset_name],
                candidate_codeblocks,
            )
            add_graph_trace_results(ucm_trace_rows, graph_trace_rows)

        if enable_ucm_multi_path:
            candidate_codeblocks = apply_candidate_gate(candidate_codeblocks, args)

        if inference_type == "bm25":
            _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
            if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
            return queries, candidate_codeblocks
        elif inference_type == "unixcoder":
            candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
            if not is_training:
                candidate_codeblocks = apply_retrieved_context_gate(candidate_codeblocks, args)
            _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
            if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
            return queries, candidate_codeblocks
        elif inference_type == "unixcoder_with_rl":
            if is_training:
                if args.disable_stop_block:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
                else:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk-1)

                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, examples)]
                _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
                if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                    log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
            else:
                if not args.disable_stop_block:
                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, examples)]

                candidate_codeblocks = retriever.retrieve(queries,  candidate_codeblocks, topk=unixcoder_topk)
                candidate_codeblocks = apply_retrieved_context_gate(candidate_codeblocks, args)
                _finalize_ucm_trace(args, dataset_name, ucm_trace_rows, candidate_codeblocks)
                if ucm_trace_rows and getattr(args, "ucm_trace_retrieval", False):
                    log_retrieval_trace(dataset_name, ucm_trace_rows, candidate_codeblocks)
        
            return queries, candidate_codeblocks

    raise ValueError("Unsupported inference type: {}".format(args.inference_type))


def _finalize_ucm_trace(args, dataset_name, trace_rows, retrieved_codeblocks):
    if not trace_rows:
        return

    add_retrieval_trace_results(trace_rows, retrieved_codeblocks)
    write_retrieval_trace(args.output_dir, dataset_name, trace_rows)


def _apply_ucm_output_suffix(args):
    if not (
        getattr(args, "enable_ucm", False)
        and getattr(args, "enable_multi_path_retrieval", False)
    ):
        return

    suffix = _ucm_output_suffix(args)
    output_dir = args.output_dir.rstrip("/\\")
    basename = os.path.basename(output_dir)
    if suffix not in basename:
        args.output_dir = f"{output_dir}_{suffix}"


def _ucm_output_suffix(args):
    parts = [
        "noid" if getattr(args, "ucm_disable_identifier_query", False) else "id",
        "noimport" if getattr(args, "ucm_disable_import_api_query", False) else "import",
        "path" if getattr(args, "ucm_enable_path_query", False) else "nopath",
        f"base{getattr(args, 'ucm_base_topk', 60)}",
        f"aux{getattr(args, 'ucm_topk_per_path', 30)}",
        f"pathk{getattr(args, 'ucm_path_topk', 5)}",
        f"pool{getattr(args, 'ucm_candidate_pool_size', 140)}",
        "legacybm25" if getattr(args, "ucm_disable_enhanced_bm25", False) else "enhbm25",
    ]
    if getattr(args, "enable_context_gate", False):
        parts.append(
            "gate_"
            f"aux{getattr(args, 'ucm_gate_max_auxiliary_blocks', 2)}_"
            f"path{getattr(args, 'ucm_gate_allow_path_only', 0)}_"
            f"stop{getattr(args, 'ucm_gate_stop_rank_threshold', 2)}"
        )
    else:
        parts.append("nogate")
    if getattr(args, "enable_context_graph", False):
        graph_parts = [
            f"graph_s{getattr(args, 'ucm_graph_max_seed', 20)}",
            f"n{getattr(args, 'ucm_graph_max_neighbors_per_seed', 2)}",
            f"e{getattr(args, 'ucm_graph_max_expanded', 40)}",
            "scored",
        ]
        if getattr(args, "ucm_graph_enable_identifier_edges", False):
            graph_parts.append("gid")
        if getattr(args, "ucm_graph_enable_import_edges", False):
            graph_parts.append("gimport")
        parts.append("_".join(graph_parts))
    return "_".join(parts)


class CustomDataset(Dataset):
    def __init__(self, max_query_length, max_candidate_length, tokenizer, queries, candidates, labels):
        self.max_query_length = max_query_length
        self.max_candidate_length = max_candidate_length
        self.tokenizer = tokenizer
        self.queries = queries
        self.candidates = candidates
        self.labels = labels

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_tokens_id = tokenize(self.queries[idx], self.tokenizer, self.max_query_length, True)
        candidate_tokens_id = [tokenize(str(x), self.tokenizer, self.max_candidate_length, False) for x in self.candidates[idx]]
        return torch.tensor(query_tokens_id, dtype=torch.long), torch.tensor(candidate_tokens_id, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def run(args):
    cceval_python_examples = load_test_dataset(args, "cceval", "python")
    cceval_java_examples = load_test_dataset(args, "cceval", "java")
    # codereval_python_examples = load_test_dataset(args, "codereval", "python")
    # codereval_java_examples = load_test_dataset(args, "codereval", "java")
    repoeval_line_examples = load_test_dataset(args, "repoeval", "line_level")
    repoeval_api_examples = load_test_dataset(args, "repoeval", "api_level")
    # repoeval_func_examples = load_test_dataset(args, "repoeval", "func_level")

    training_raw_data, eval_raw_data = load_train_and_valid_dataset()
    eval_all_examples = construct_dataset(eval_raw_data, 100 if args.debug else 1000)

    all_eval_examples = {
        "github_eval": eval_all_examples,
        "cceval_python": cceval_python_examples,
        "cceval_java": cceval_java_examples,
        # "codereval_python": codereval_python_examples,
        # "codereval_java": codereval_java_examples,
        "repoeval_line": repoeval_line_examples,
        "repoeval_api": repoeval_api_examples,
        # "repoeval_func": repoeval_func_examples,
    }


    global generator
    generator = Generator(args)
    retriever = Retriever(args)


    if args.enable_repocoder:
        args_RLCoder = copy.deepcopy(args)
        args_RLCoder.retriever_model_path = args.rlcoder_model_path
        global retriever_RLCoder
        retriever_RLCoder = Retriever(args_RLCoder)
    

    if not args.enable_forward_generation:
        args.forward_generation_times = 1
    else:
        if args.forward_generation_times is None:
            args.forward_generation_times = 4

    bm25 = {}
    
    if args.eval:
        table = PrettyTable()
        table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        codereval_table = PrettyTable()
        codereval_table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
        
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            print("Evaluating on {} dataset".format(name))
            
            temp_examples = copy.deepcopy(examples)
            temp_generations = []
                
            for _ in range(args.forward_generation_times):
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                # for i in range(len(retrieved_codeblocks)):
                #     for j in range(len(retrieved_codeblocks[i])):
                #         print('#', retrieved_codeblocks[i][j].file_path)
                #         print(retrieved_codeblocks[i][j].code_content)
                losses = generator.evaluate(examples, retrieved_codeblocks)


                results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if not os.path.exists(f"{args.output_dir}/{name}"):
                    os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
                with open(f"{args.output_dir}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")


                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, temp_generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in temp_generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(temp_generations,targets)])/len(temp_generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                # elif name == "repoeval_func":
                #     results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/func_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
            

            if 'codereval' in name:
                codereval_table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(table)
            print(codereval_table)
        
    else:
        print("data_per_epoch:{}, batch_size:{}, sample_number:{}, epoch:{}, inner_epoch:{}, lr:{}".format(args.data_per_epoch, args.batch_size,args.sample_number,args.epoch,args.inner_epoch,args.lr))
        optimizer = AdamW(retriever.model.parameters(), lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.2, num_training_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch)
    
        evaluate_table = {}
        for name, examples in all_eval_examples.items():
            evaluate_table[name] = PrettyTable()
            if 'codereval' in name:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
            else:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        training_table = PrettyTable()
        training_table.field_names = ["Epoch", "Dataset", "Total Samples", "Rewards", "Training Loss", "Time (sec)"]


        retriever.model.eval()
        for name, examples in all_eval_examples.items():
            # examples = examples[:10]
            
            start_time = time.time()
            temp_examples = copy.deepcopy(examples)
            temp_generations = []

                
            for _ in range(args.forward_generation_times):
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                losses = generator.evaluate(examples, retrieved_codeblocks)


                results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if os.path.exists(f"{args.output_dir}/result_init/{name}") is False:
                    os.makedirs(f"{args.output_dir}/result_init/{name}", exist_ok=True)
                with open(f"{args.output_dir}/result_init/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")

                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations, targets)])/len(generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                

            if 'codereval' in name:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(evaluate_table[name])


        for epoch in range(args.epoch):
            print("=" * 40 + "Epoch:{}".format(epoch) + "=" * 40)
            retriever.model.eval()
            start_time = time.time()
            results = {}
            results["Epoch"] = epoch


            training_examples = construct_dataset(training_raw_data, 100 if args.debug else args.data_per_epoch)
            # training_examples = construct_dataset(training_raw_data, 100)
            queries, retrieved_codeblocks = retrieve_codeblocks(args, training_examples, bm25, retriever, "github_training_{}".format(epoch), True)
            training_examples_dup = [x for x in training_examples for _ in range(args.sample_number)]
            training_codeblocks_dup = [[x] for y in retrieved_codeblocks for x in y]
            assert len(training_examples_dup) == len(training_codeblocks_dup)


            losses = generator.evaluate(training_examples_dup, training_codeblocks_dup)
            labels = torch.tensor([x for x in losses]).view(-1, args.sample_number).argmin(-1)
            results["Total Samples"] = len(queries)
            results["Rewards"] = labels.float().mean().item()

            retriever.model.train()
            total_loss = 0
            dataset = CustomDataset(args.retriever_query_context_length, args.retriever_candidate_context_length, retriever.tokenizer, queries, retrieved_codeblocks, labels.tolist())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            for inner_epoch in range(args.inner_epoch):
                for batch in dataloader:
                    source_ids, doc_ids, labels = [x.cuda() for x in batch]
                    queries_embeddings = retriever(source_ids)
                    doc_texts_embeddings = retriever(doc_ids.view(-1, doc_ids.shape[-1])).view(source_ids.shape[0], args.sample_number, -1)
                    logits = torch.einsum("ab,acb->ac", queries_embeddings, doc_texts_embeddings)*20
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(retriever.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    total_loss += loss.item()

                if args.enable_sft:
                    retriever.model.eval()
                    for name, examples in all_eval_examples.items():
                        # examples = examples[:10]
                        
                        start_time = time.time()
                        temp_examples = copy.deepcopy(examples)
                        temp_generations = []

                            
                        for _ in range(args.forward_generation_times):
                            _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                            losses = generator.evaluate(examples, retrieved_codeblocks)

                            results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                            if args.enable_generation:
                                generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                                if not temp_generations:
                                    temp_generations = generations
                                else:
                                    temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                                for i in range(len(temp_examples)):
                                    temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                                    
                        if args.enable_generation:
                            if os.path.exists(f"{args.output_dir}/result_{inner_epoch}/{name}") is False:
                                os.makedirs(f"{args.output_dir}/result_{inner_epoch}/{name}", exist_ok=True)
                            with open(f"{args.output_dir}/result_{inner_epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                                for example, generation in zip(examples, temp_generations):
                                    f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                            if name == "cceval_python":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "cceval_java":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                            elif name == "github_eval":
                                targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                                results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                            elif name == "codereval_python":
                                results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                            elif name == "codereval_java":
                                results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                            elif name == "repoeval_line":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "repoeval_api":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                        if 'codereval' in name:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                        else:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                        print(evaluate_table[name])
                    
                    retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")
                    retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")

            results["Training Loss"] = total_loss/len(dataloader)/args.inner_epoch
            results["Time (sec)"] = round(time.time() - start_time, 1)
            training_table.add_row([results["Epoch"], "github_training_{}".format(epoch), results["Total Samples"], results["Rewards"], results["Training Loss"], results["Time (sec)"]])
            print(training_table)

            
            retriever.model.eval()
            for name, examples in all_eval_examples.items():
                # examples = examples[:10]
                
                start_time = time.time()
                temp_examples = copy.deepcopy(examples)
                temp_generations = []
                    
                for _ in range(args.forward_generation_times):
                    _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                    losses = generator.evaluate(examples, retrieved_codeblocks)

                    results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                    if args.enable_generation:
                        generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                        if not temp_generations:
                            temp_generations = generations
                        else:
                            temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                        for i in range(len(temp_examples)):
                            temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                            
                if args.enable_generation:
                    if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
                        os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
                    with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                        for example, generation in zip(examples, temp_generations):
                            f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                    if name == "cceval_python":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "cceval_java":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                    elif name == "github_eval":
                        targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                        results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                    elif name == "codereval_python":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                    elif name == "codereval_java":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                    elif name == "repoeval_line":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "repoeval_api":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                if 'codereval' in name:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                else:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                print(evaluate_table[name])

            retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_model_path", default=local_model_path("deepseek-coder-6.7b-base"), type=str, help="Generator model path")
    parser.add_argument("--generator_batch_size_per_gpu", default=32, type=int, help="Generator batch size per GPU")
    parser.add_argument("--generator_max_crossfile_length", default=512, type=int, help="Maximum cross-file length for the generator")
    parser.add_argument("--generator_max_context_length", default=1024, type=int, help="Maximum context length for the generator")
    parser.add_argument("--generator_max_generation_length", default=64, type=int, help="Maximum generation length for the generator")
    parser.add_argument("--disable_generator", action="store_true", help="Disable the generator")

    parser.add_argument("--retriever_model_path", default=local_model_path("unixcoder-base"), type=str, help="Retriever model path")
    parser.add_argument("--retriever_batch_size_per_gpu", default=64, type=int, help="Retriever batch size per GPU")
    parser.add_argument("--disable_retriever", action="store_true", help="Disable the retriever")
    parser.add_argument("--retriever_query_context_length", default=256, type=int, help="Retriever query context length")
    parser.add_argument("--retriever_candidate_context_length", default=512, type=int, help="Retriever candidate context length")

    parser.add_argument("--inference_type", default="baseline", type=str, help="Inference type")
    parser.add_argument("--output_dir", default="results/baseline", type=str, help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation")
    parser.add_argument("--enable_tqdm", action="store_true", help="Enable progress bar")
    parser.add_argument("--enable_generation", action="store_true", help="Enable generation")
    parser.add_argument("--debug", action="store_true", help="Debug mode, use a small dataset")

    parser.add_argument("--num_workers", default=14, type=int, help="Number of CPU cores")
    parser.add_argument("--weighted_keywords", action="store_true", help="Weight keywords when calculating loss during training")
    parser.add_argument("--enable_fixed_block", action="store_true", help="Use fixed length blocks when building candidates")
    parser.add_argument("--enable_sft", action="store_true", help="Train using supervised learning methods")
    parser.add_argument("--disable_stop_block", action="store_true", help="Disable the stop block")

    parser.add_argument("--enable_repocoder", action="store_true", help="Use the repocoder method during generation")
    parser.add_argument("--rlcoder_model_path", default=local_model_path("unixcoder-base"), type=str, help="Stage 1 model for repocoder")

    parser.add_argument("--enable_ucm", action="store_true", help="Enable UCM extensions")
    parser.add_argument("--enable_multi_path_retrieval", action="store_true", help="Enable UCM multi-path BM25 candidate recall")
    parser.add_argument("--ucm_base_topk", default=60, type=int, help="BM25 candidates for the base query; 0 keeps the original RLCoder topK")
    parser.add_argument("--ucm_topk_per_path", default=30, type=int, help="Number of BM25 candidates per auxiliary UCM query view")
    parser.add_argument("--ucm_path_topk", default=5, type=int, help="Number of BM25 candidates for the optional path query view")
    parser.add_argument("--ucm_candidate_pool_size", default=140, type=int, help="Maximum merged UCM candidate pool size")
    parser.add_argument("--ucm_disable_identifier_query", action="store_true", help="Disable the UCM identifier query view")
    parser.add_argument("--ucm_disable_import_api_query", action="store_true", help="Disable the UCM import/API query view")
    parser.add_argument("--ucm_enable_path_query", action="store_true", help="Enable the path-based UCM query view")
    parser.add_argument("--ucm_query_identifier_limit", default=64, type=int, help="Maximum identifier tokens in the UCM identifier query")
    parser.add_argument("--ucm_query_import_limit", default=32, type=int, help="Maximum import/API lines or tokens in the UCM import query")
    parser.add_argument("--ucm_disable_enhanced_bm25", action="store_true", help="Use the original BM25 tokenization and code-only index for UCM")
    parser.add_argument("--ucm_trace_retrieval", action="store_true", help="Print UCM retrieval trace; jsonl trace files are always written for UCM retrieval")
    parser.add_argument("--enable_context_gate", action="store_true", help="Enable lightweight rule-based UCM context gate")
    parser.add_argument("--ucm_gate_max_auxiliary_blocks", default=2, type=int, help="Maximum auxiliary-only UCM candidates kept before reranking")
    parser.add_argument("--ucm_gate_allow_path_only", default=0, type=int, help="Maximum path-only UCM candidates kept before reranking")
    parser.add_argument("--ucm_gate_stop_rank_threshold", default=2, type=int, help="Trim final context after an early stop block at or before this rank")
    parser.add_argument("--enable_context_graph", action="store_true", help="Enable lightweight UCM context graph expansion before RLRetriever reranking")
    parser.add_argument("--ucm_graph_max_seed", default=20, type=int, help="Maximum retrieved/merged seed candidates used for graph expansion")
    parser.add_argument("--ucm_graph_max_neighbors_per_seed", default=2, type=int, help="Maximum graph neighbors added for each seed candidate")
    parser.add_argument("--ucm_graph_max_expanded", default=40, type=int, help="Maximum total graph-expanded candidates per example")
    parser.add_argument("--ucm_graph_enable_identifier_edges", action="store_true", help="Enable identifier-overlap graph edges")
    parser.add_argument("--ucm_graph_enable_import_edges", action="store_true", help="Enable import/path graph edges")
    parser.add_argument("--ucm_graph_identifier_max_df", default=20, type=int, help="Maximum per-task document frequency for identifier graph edges")
    parser.add_argument("--ucm_graph_seed_rank_decay", default=0.05, type=float, help="Decay applied to graph expansion candidates from lower-ranked seed blocks")
    parser.add_argument("--ucm_graph_distance_decay", default=0.75, type=float, help="Distance decay for same-file graph neighbors")
    parser.add_argument("--ucm_graph_same_file_weight", default=1.0, type=float, help="Base score weight for same-file graph edges")
    parser.add_argument("--ucm_graph_identifier_weight", default=1.2, type=float, help="Base score weight for identifier-overlap graph edges")
    parser.add_argument("--ucm_graph_import_weight", default=1.4, type=float, help="Base score weight for import/path graph edges")
    parser.add_argument("--ucm_graph_query_overlap_bonus", default=2, type=int, help="Extra identifier graph score for tokens also present in the query context")

    parser.add_argument("--do_codereval", action="store_true", help="Execute codereval evaluation in docker")
    parser.add_argument("--enable_forward_generation", action="store_true", help="Use progressive generation methods during inference")
    parser.add_argument("--forward_generation_times", default=4, type=int, help="Number of times for progressive generation")

    parser.add_argument("--epoch", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--inner_epoch", default=1, type=int, help="Number of inner training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--sample_number", default=10, type=int, help="Number of samples")
    parser.add_argument("--data_per_epoch", default=2000, type=int, help="Amount of data per epoch")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")


    print("Number of GPUs:", torch.cuda.device_count())

    args = parser.parse_args()
    _apply_ucm_output_suffix(args)
    print("Output dir:", args.output_dir)
    args.generator_batch_size = args.generator_batch_size_per_gpu * torch.cuda.device_count()
    args.retriever_batch_size = args.retriever_batch_size_per_gpu * torch.cuda.device_count()

    run(args)
