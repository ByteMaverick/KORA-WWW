"""Small introduction:
This script runs knowledge-graph question-answering experiments by driving an LLM (OpenAI / Qwen)
to answer questions from datasets (CWQ, WebQSP, SimpleQuestions, GrailQA, QALD). It supports
tool calls to fetch adjacent relations/entities from Freebase or Wikidata, multi-threaded
execution, partial saving of results, and configurable modes via environment variables.

Usage (examples):
  SAMPLE=true DATASET=CWQ MODEL_NAME=gpt-4o python main.py
  OPENAI_API_KEY=... DATASET=WebQSP NUM_EXAMPLES=3 USE_TOOLS=true python main.py

See run.sh for local vLLM serving examples and the repository README for more details.
"""

from functools import partial
import json
import sys
import os
from openai import OpenAI
import datetime
import concurrent.futures
import threading
from tqdm import tqdm
import traceback
import copy
import time as time_module

# Import all functions from the unified helper file
from few_shot_llm_tools_unified_helper import (
    get_adjacent_relations_and_entities_freebase,
    get_adjacent_relations_and_entities_wikidata,
    exact_match,
    tool_call_to_dict,
    extract_answers_after_final_answers,
    extract_answers_inside_curly_braces,
)

import few_shot_llm_tools_unified_helper

# Import system messages from utils_ai
from prompt import get_system_prompt, TOOLS_FREEBASE, TOOLS_WIKIDATA

thread_local = threading.local()


# Global statistics tracking
class UsageStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.total_calls = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cached_tokens = 0
            self.total_reasoning_tokens = 0

    def add_usage(self, usage_dict):
        if usage_dict:
            with self.lock:
                self.total_calls += 1
                self.total_input_tokens += usage_dict.get("input_tokens", 0)
                self.total_output_tokens += usage_dict.get("output_tokens", 0)
                self.total_cached_tokens += usage_dict.get("cached_tokens", 0)
                self.total_reasoning_tokens += usage_dict.get("reasoning_tokens", 0)

    def get_stats(self):
        with self.lock:
            return {
                "total_calls": self.total_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
            }


# Global instance
usage_stats = UsageStats()


def get_imports_for_dataset(dataset, format):
    """Get appropriate functions and tools based on dataset"""
    # Common functions for all datasets
    base_imports = {
        "extract_answers_after_final_answers": extract_answers_after_final_answers,
        "extract_answers_inside_curly_braces": extract_answers_inside_curly_braces,
        "exact_match": exact_match,
        "tool_call_to_dict": tool_call_to_dict,
    }

    # Choose Freebase vs Wikidata and deepcopy tool specs to avoid mutating globals
    if dataset in ["CWQ", "WebQSP", "SimpleQuestions", "GrailQA"]:
        base_imports.update(
            {
                "get_adjacent_fn": get_adjacent_relations_and_entities_freebase,
                "tools": copy.deepcopy(TOOLS_FREEBASE),
            }
        )
    else:  # QALD-9, QALD-10
        base_imports.update(
            {
                "get_adjacent_fn": get_adjacent_relations_and_entities_wikidata,
                "tools": copy.deepcopy(TOOLS_WIKIDATA),
            }
        )

    # If specific format requires modifying the tool schema, operate on the copied tools
    if format == "markdown-short":
        pass
    elif format == "markdown":
        del base_imports["tools"][0]["function"]["parameters"]["properties"][
            "properties_to_filter_for"
        ]
        del base_imports["tools"][0]["function"]["parameters"]["required"][2]
        base_imports["get_adjacent_fn"] = (
            partial(get_adjacent_relations_and_entities_freebase, return_format=format)
            if dataset in ["CWQ", "WebQSP", "SimpleQuestions", "GrailQA"]
            else partial(
                get_adjacent_relations_and_entities_wikidata, return_format=format
            )
        )
    elif format == "json":
        del base_imports["tools"][0]["function"]["parameters"]["properties"][
            "properties_to_filter_for"
        ]
        del base_imports["tools"][0]["function"]["parameters"]["required"][2]
        base_imports["get_adjacent_fn"] = (
            partial(get_adjacent_relations_and_entities_freebase, return_format=format)
            if dataset in ["CWQ", "WebQSP", "SimpleQuestions", "GrailQA"]
            else partial(
                get_adjacent_relations_and_entities_wikidata, return_format=format
            )
        )

    return base_imports


def get_system_message(dataset, num_examples, imports, io_mode=False):
    if io_mode:
        return get_system_prompt(dataset, num_examples=-1)
    return get_system_prompt(dataset, num_examples=num_examples)


def extract_gold_answers(dataset, q_item, idx):
    """Extract gold answers based on dataset format"""
    gold_strings = set()

    if dataset == "CWQ":
        answers_processed = q_item.get("answers_processed", [])
        for obj in answers_processed:
            label_val = obj.get("x_label", {}).get("value", "")
            if label_val:
                gold_strings.add(label_val)
            else:
                x_val = obj.get("x", {}).get("value", "")
                if x_val:
                    gold_strings.add(x_val)

    elif dataset == "WebQSP":
        parses = q_item.get("Parses", [])
        for parse in parses:
            answers_processed = parse.get("AnswersProcessed", [])
            for ans_processed in answers_processed:
                x_info = ans_processed.get("x", {})
                x_type = x_info.get("type", "")
                if x_type == "uri":
                    label_val = (
                        ans_processed.get("x_label", {}).get("value", "").strip()
                    )
                    if label_val:
                        gold_strings.add(label_val)
                elif x_type == "typed-literal":
                    typed_val = x_info.get("value", "").strip()
                    if typed_val:
                        gold_strings.add(typed_val)
                elif x_type == "literal":
                    literal_val = x_info.get("value", "").strip()
                    if literal_val:
                        gold_strings.add(literal_val)

    elif dataset == "SimpleQuestions":
        obj_name = q_item.get("ObjectName")
        if obj_name and obj_name != "-":
            gold_strings.add(obj_name)

    elif dataset == "GrailQA":
        answers = q_item.get("answer", [])
        for ans in answers:
            answer_type = ans.get("answer_type", "")
            if answer_type == "Entity":
                entity_name = ans.get("entity_name", "")
                if entity_name:
                    gold_strings.add(entity_name)
            elif answer_type == "Value":
                value = ans.get("answer_argument", "")
                if value:
                    gold_strings.add(value)

    elif dataset in ["QALD-9", "QALD-10"]:
        answers_processed = q_item.get("answers_processed", [])
        for ans_processed in answers_processed:
            if isinstance(ans_processed, str):
                gold_strings.add(ans_processed)
            elif isinstance(ans_processed, bool):
                gold_strings.add(str(ans_processed))

    return list(gold_strings)


def extract_question_text(dataset, q_item):
    """Extract question text based on dataset format"""
    if dataset in ["CWQ", "GrailQA"]:
        return q_item.get("question", "")
    elif dataset == "WebQSP":
        return q_item.get("ProcessedQuestion", "")
    elif dataset == "SimpleQuestions":
        return q_item.get("QuestionText", "")
    elif dataset in ["QALD-9", "QALD-10"]:
        for q_obj in q_item.get("question", []):
            if q_obj.get("language") == "en":
                return q_obj.get("string")
    return None


def get_question_id(dataset, q_item, idx):
    """Extract question ID based on dataset format"""
    if dataset == "CWQ":
        return q_item.get("ID")
    elif dataset == "WebQSP":
        return q_item.get("QuestionId", f"Unnamed-{idx}")
    elif dataset == "SimpleQuestions":
        return q_item.get("question_id", idx)
    elif dataset == "GrailQA":
        return q_item.get("qid", idx)
    elif dataset in ["QALD-9", "QALD-10"]:
        return q_item.get("id")
    return idx


def process_single_question(
    dataset,
    q_item,
    idx,
    total_questions,
    client,
    TOOLS,
    model_name,
    thinking,
    num_examples,
    use_tools,
    imports,
    io_mode=False,
):
    """
    Process a single question and return its results (threaded version).
    """
    start_time = time_module.time()

    # Get question ID and gold answers
    question_id = get_question_id(dataset, q_item, idx)
    gold_strings = extract_gold_answers(dataset, q_item, idx)

    # Skip if no gold answers
    if not gold_strings:
        if dataset in [
            "CWQ",
            "WebQSP",
            "SimpleQuestions",
            "GrailQA",
            "QALD-9",
            "QALD-10",
        ]:
            print(f"Skipping question {question_id} - no gold answers")
            return None
        gold_strings = []

    print(
        f"\n=== {dataset} question (id={question_id}) => # {idx}/{total_questions} ==="
    )
    if io_mode:
        print(f"    [IO Mode - No examples, No tools]")
    else:
        print(f"    [Examples: {num_examples}, Tools: {use_tools}]")

    # Print thinking parameter only for Qwen models
    if "qwen" in model_name.lower():
        print(f"    [Thinking: {thinking}]")

    try:
        # Extract question text
        question_text = extract_question_text(dataset, q_item)
        if not question_text:
            print(
                f"Warning: No question found for question_id={question_id}. Skipping."
            )
            return None

        # Get the appropriate system message
        system_message = get_system_message(dataset, num_examples, imports, io_mode)

        messages = [
            system_message,
            {"role": "user", "content": f"Question: {question_text}\n\n"},
        ]

        final_answer = ""
        conversation_history = []

        for _ in range(50):
            # Prepare completion arguments
            completion_args = {
                "model": model_name,
                "messages": messages,
                "timeout": 3600,
            }
            if not thinking:
                completion_args["max_tokens"] = 20480

            # Only add tools if use_tools is True (which is False in IO mode)
            if use_tools:
                completion_args["tools"] = TOOLS

            # Model-specific parameters for Qwen models
            if "qwen2.5" in model_name.lower():
                # Use default parameters for Qwen2.5
                pass
            elif "qwen3" in model_name.lower():
                if thinking == False:
                    completion_args.update(
                        {
                            "temperature": 0.7,
                            "top_p": 0.8,
                            "presence_penalty": 1.5,
                            "extra_body": {
                                "top_k": 20,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                        }
                    )
                else:
                    completion_args.update(
                        {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "extra_body": {
                                "top_k": 20,
                                "min_p": 0,
                                "chat_template_kwargs": {"enable_thinking": True},
                            },
                        }
                    )

            # Create the completion
            completion = client.chat.completions.create(**completion_args)
            assistant_msg = completion.choices[0].message

            # Track usage statistics
            if getattr(completion, "usage", None):
                prompt_tokens_details = getattr(
                    completion.usage, "prompt_tokens_details", None
                )
                completion_tokens_details = getattr(
                    completion.usage, "completion_tokens_details", None
                )
                cached_tokens = (
                    prompt_tokens_details.get("cached_tokens", 0)
                    if isinstance(prompt_tokens_details, dict)
                    else 0
                )
                reasoning_tokens = (
                    completion_tokens_details.get("reasoning_tokens", 0)
                    if isinstance(completion_tokens_details, dict)
                    else 0
                )
                usage_dict = {
                    "input_tokens": getattr(completion.usage, "prompt_tokens", 0)
                    or getattr(completion.usage, "input_tokens", 0),
                    "output_tokens": getattr(completion.usage, "completion_tokens", 0)
                    or getattr(completion.usage, "output_tokens", 0),
                    "total_tokens": getattr(completion.usage, "total_tokens", 0),
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                }
                usage_stats.add_usage(usage_dict)

            # Only extract reasoning_content for Qwen3 models
            reasoning_content = None
            if "qwen3" in model_name.lower():
                reasoning_content = getattr(assistant_msg, "reasoning_content", None)

            # Handle tool calls only if tools are enabled
            if use_tools and assistant_msg.tool_calls:
                tool_calls_as_dicts = [
                    tool_call_to_dict(tc) for tc in assistant_msg.tool_calls
                ]
            else:
                tool_calls_as_dicts = []

            conversation_entry = {
                "role": assistant_msg.role,
                "content": assistant_msg.content,
                "tool_calls": tool_calls_as_dicts,
            }

            # Add reasoning content to conversation history only for Qwen3 models
            if "qwen3" in model_name.lower() and reasoning_content:
                conversation_entry["reasoning_content"] = reasoning_content

            conversation_history.append(conversation_entry)

            messages.append(assistant_msg)

            if not tool_calls_as_dicts:
                final_answer = assistant_msg.content
                break

            # Process tool calls only if we have them
            if tool_calls_as_dicts and use_tools:
                # Process tool calls in parallel
                def process_tool_call(tool_call):
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    args = json.loads(raw_args)

                    # Add SPARQL endpoint for Freebase datasets
                    if dataset in ["CWQ", "WebQSP", "SimpleQuestions", "GrailQA"]:
                        args["sparql_endpoint"] = "http://localhost:8890/sparql"

                    if fn_name == "get_adjacent_relations_and_entities":
                        result = imports["get_adjacent_fn"](**args)

                        # Check if result only contains property and propertyLabel columns
                        if (
                            isinstance(result, str)
                            and "property" in result
                            and "propertyLabel" in result
                        ):
                            lines = result.strip().split("\n")
                            if lines and "|" in lines[0]:
                                headers = [h.strip() for h in lines[0].split("|")]
                                if len(headers) == 2 and "value" not in [
                                    h.lower() for h in headers
                                ]:
                                    # Construct a message instructing the model to filter
                                    filter_instruction = (
                                        f"The query returned too many results. Here are the available properties:\n\n"
                                        f"{result}\n\n"
                                        f"In order to reduce the number of results shown, please identify which properties are most relevant to the question and make another call to get_adjacent_relations_and_entities with:\n"
                                        # f"- The same question: {args.get('question', '')}\n"
                                        f"- The same entity: {args.get('entity', '')}\n"
                                        f"- The same direction: {args.get('direction', '')}\n"
                                        f"- Add properties_to_filter_for parameter with a list of relevant property IDs (use the property column values, not the labels)\n\n"
                                        f"For example, if 'location.location.contains' is relevant, include it in the list."
                                    )
                                    return {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": filter_instruction,
                                    }
                    else:
                        result = f"Error: unknown function {fn_name}"

                    if isinstance(result, (dict, list)):
                        import json as _json

                        try:
                            result = _json.dumps(result, ensure_ascii=False)
                        except Exception:
                            result = str(result)
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }

                # Process tool calls with ThreadPoolExecutor
                tool_messages = []
                tool_calls = assistant_msg.tool_calls
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(tool_calls) if tool_calls else 1
                ) as tool_executor:
                    future_to_tool_call = {
                        tool_executor.submit(process_tool_call, tc): tc
                        for tc in tool_calls
                    }

                    results_in_order = [None] * len(tool_calls)
                    original_tool_call_to_idx = {
                        tc.id: i for i, tc in enumerate(tool_calls)
                    }

                    for future in concurrent.futures.as_completed(future_to_tool_call):
                        tool_call_obj = future_to_tool_call[future]
                        try:
                            tool_msg = future.result()
                            idx_tool = original_tool_call_to_idx[tool_call_obj.id]
                            results_in_order[idx_tool] = tool_msg
                        except Exception as exc:
                            print(
                                f"Tool call {tool_call_obj.function.name} generated an exception: {exc}"
                            )
                            error_msg = {
                                "role": "tool",
                                "tool_call_id": tool_call_obj.id,
                                "content": f"Error executing tool {tool_call_obj.function.name}: {exc}",
                            }
                            idx_tool = original_tool_call_to_idx[tool_call_obj.id]
                            results_in_order[idx_tool] = error_msg

                    for tool_msg in results_in_order:
                        if tool_msg:
                            tool_messages.append(tool_msg)
                            conversation_history.append(tool_msg)
                            messages.append(tool_msg)

        # Calculate exact match
        em_result = False
        if final_answer:
            print(f"{final_answer}")
            final_answer_inside_curly_braces = extract_answers_inside_curly_braces(
                final_answer
            )
            print(
                f"Extracted answers inside curly braces: {final_answer_inside_curly_braces}"
            )
            print(f"Gold answers: {gold_strings}")

            if final_answer_inside_curly_braces:
                em_result = any(
                    exact_match(pred_answer, gold_answer)
                    for pred_answer in final_answer_inside_curly_braces
                    for gold_answer in gold_strings
                )
                print(f"Exact match result based on extracted answers: {em_result}")
            else:
                em_result = any(exact_match(final_answer, g) for g in gold_strings)

        # Build settings dict
        settings = {
            "io_mode": io_mode,
            "num_examples": num_examples if not io_mode else False,
            "use_tools": use_tools if not io_mode else False,
            "seq_length": SEQ_LENGTH,
        }

        # Add thinking parameter only for Qwen models
        if "qwen" in model_name.lower():
            settings["thinking"] = thinking

        end_time = time_module.time()
        question_time = end_time - start_time

        print(few_shot_llm_tools_unified_helper.KG_QUERY_COUNT_SOG)
        # Return results for this question
        return {
            "question_id": question_id,
            "question_text": question_text,
            "conversation_history": conversation_history,
            "model_answer": final_answer,  # Use generic field name
            "gold_answers": gold_strings,
            "exact_match": em_result,
            "settings": settings,
            "processing_time": question_time,
        }

    except Exception as e:
        print(f"Error processing question {question_id}: {e}")
        traceback.print_exc()
        return {
            "question_id": question_id,
            "question_text": question_text if "question_text" in locals() else None,
            "model_answer": final_answer if "final_answer" in locals() else None,
            "error": str(e),
        }


# Environment variable configurations
DATASET = os.getenv("DATASET", "CWQ")
SAMPLE = os.getenv("SAMPLE", "False").lower() == "true"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
IO_MODE = os.getenv("IO_MODE", "False").lower() == "true"

# Override settings if IO mode is enabled
if IO_MODE:
    num_examples = -1
    USE_TOOLS = False
else:
    num_examples = int(os.getenv("NUM_EXAMPLES", "0"))
    USE_TOOLS = os.getenv("USE_TOOLS", "True").lower() == "true"

SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "1"))
START_FROM_INDEX = int(os.getenv("START_FROM_INDEX", "0"))
SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", "131072"))
FORMAT = os.getenv("FORMAT", "markdown-short").lower()  # "json" or "jsonl"
assert FORMAT in [
    "json",
    "markdown",
    "markdown-short",
], "FORMAT must be one of 'json', 'markdown', or 'markdown-short'"


def main():
    # Reset usage stats before processing
    usage_stats.reset()

    # Import appropriate modules for dataset
    imports = get_imports_for_dataset(DATASET, FORMAT)
    TOOLS = imports["tools"]

    # Get model name
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

    # Determine if we're using Qwen (vLLM) or GPT (OpenAI API)
    is_qwen = "qwen" in MODEL_NAME.lower()

    # Initialize client based on model type
    if is_qwen:
        # vLLM configuration for Qwen models
        VLLM_API_BASE = os.getenv("OPENAI_API_BASE_URL", "http://localhost:9988/v1")
        VLLM_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")

        client = OpenAI(base_url=VLLM_API_BASE, api_key=VLLM_API_KEY, timeout=1200)

        # Qwen-specific settings
        THINKING = os.getenv("THINKING", "False").lower() == "true"
        MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "20"))
    else:
        # OpenAI API configuration for GPT models
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for GPT models"
            )

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=1200)

        # GPT-specific settings
        THINKING = False  # Not used for GPT
        MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "100"))

    # Print configuration
    print("=" * 60)
    print("CONFIGURATION:")
    print(f"  Dataset: {DATASET}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Model Type: {'Qwen (vLLM)' if is_qwen else 'GPT (OpenAI API)'}")
    if IO_MODE:
        print(f"  IO Mode: ENABLED (No examples, No tools)")
    else:
        print(f"  Use Examples: {num_examples}")
        print(f"  Use Tools: {USE_TOOLS}")
    if is_qwen:
        print(f"  Thinking: {THINKING}")
    print(f"  Sample Mode: {SAMPLE}")
    print(f"  Debug Mode: {DEBUG}")
    print(f"  Max Concurrency: {MAX_CONCURRENCY}")
    print(f"  Save Interval: Every {SAVE_INTERVAL} questions")
    print(f"  Start From Index: {START_FROM_INDEX}")
    print("=" * 60)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", f"{DATASET}.json")

    with open(data_path, "r", encoding="utf-8") as f:
        loaded_test_data = json.load(f)

    # Extract questions based on dataset format
    if DATASET in ["CWQ", "GrailQA"]:
        questions_all = loaded_test_data
    elif DATASET in ["WebQSP", "SimpleQuestions"]:
        questions_all = loaded_test_data.get("Questions", [])
    elif DATASET in ["QALD-9", "QALD-10"]:
        questions_all = loaded_test_data.get("questions", [])

    # Apply filtering based on DEBUG and SAMPLE modes
    if DEBUG:
        questions_all = questions_all[:10]
        print(f"DEBUG MODE: Processing first 10 questions only")

    if SAMPLE:
        questions_filtered = [
            questions_all[i] for i in range(0, len(questions_all), 10)
        ]
        # + [questions_all[i] for i in range(1, len(questions_all), 10)]
        print(
            f"SAMPLE mode: picking every 10th question => {len(questions_filtered)} total out of {len(questions_all)}."
        )

        if START_FROM_INDEX > 0:
            if START_FROM_INDEX >= len(questions_filtered):
                print(
                    f"ERROR: START_FROM_INDEX ({START_FROM_INDEX}) is >= number of sampled questions ({len(questions_filtered)})"
                )
                sys.exit(1)
            questions_filtered = questions_filtered[START_FROM_INDEX:]
            print(
                f"Starting from index {START_FROM_INDEX} among sampled questions => {len(questions_filtered)} questions to process"
            )

        questions_all = questions_filtered
    else:
        if START_FROM_INDEX > 0:
            if START_FROM_INDEX >= len(questions_all):
                print(
                    f"ERROR: START_FROM_INDEX ({START_FROM_INDEX}) is >= total number of questions ({len(questions_all)})"
                )
                sys.exit(1)
            questions_all = questions_all[START_FROM_INDEX:]
            print(
                f"Starting from index {START_FROM_INDEX} => {len(questions_all)} questions to process"
            )

    questions = questions_all[::-1]
    total_questions = len(questions)

    print(f"Starting processing of {total_questions} questions...")

    # Prepare output path with configuration in filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Update mode string based on IO mode
    examples_str = f"{num_examples}_examples"
    tools_str = "with_tools" if USE_TOOLS else "no_tools"
    format_str = f"{FORMAT}_format"
    mode_str = f"{examples_str}_{tools_str}_{format_str}"

    # Create results directory structure
    results_dir = f"results/{DATASET}"
    os.makedirs(results_dir, exist_ok=True)

    # Update output path logic (no SPECIFIC_QUESTION_IDS branching)
    if START_FROM_INDEX > 0:
        output_path = f"{results_dir}/{len(questions)}_from_{START_FROM_INDEX}_{MODEL_NAME.replace('/', '-')}_{mode_str}_answers_{timestamp}.json"
    else:
        output_path = f"{results_dir}/{len(questions)}_{MODEL_NAME.replace('/', '-')}_{mode_str}_answers_{timestamp}.json"

    print(f"Partial results will be saved to: {output_path}")

    # Store results in order with futures
    all_results = [None] * total_questions
    import time

    overall_start = time.time()

    # Multi-threaded execution for questions (always use executor loop)
    # This will handle single-question cases as well.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        # Submit all tasks and map futures to their positions
        future_to_idx = {}
        for idx, q_item in enumerate(questions):
            future = executor.submit(
                process_single_question,
                DATASET,
                q_item,
                idx + 1,
                total_questions,
                client,
                TOOLS if USE_TOOLS else None,
                MODEL_NAME,
                THINKING,
                num_examples,
                USE_TOOLS,
                imports,
                IO_MODE,
            )
            future_to_idx[future] = idx

        # Track completed questions for batch saving
        completed_count = 0

        # Process results as they complete, showing progress
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=total_questions,
            desc="Processing questions",
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                all_results[idx] = result
            except Exception as exc:
                print(f"Question {idx+1} generated an exception: {exc}")
                traceback.print_exc()
                continue

            completed_count += 1

            # Save partial results every SAVE_INTERVAL questions or on the last question
            if (
                completed_count % SAVE_INTERVAL == 0
                or completed_count == total_questions
            ):
                partial_results = [res for res in all_results if res is not None]
                with open(output_path, "w", encoding="utf-8") as fout:
                    json.dump(partial_results, fout, indent=2, ensure_ascii=False)
                print(
                    f"Saved {len(partial_results)}/{total_questions} results to {output_path}"
                )

    overall_end = time.time()
    overall_duration = overall_end - overall_start
    print(f"Overall processing time: {overall_duration:.2f} seconds")

    # Remove None values from results
    all_results = [res for res in all_results if res is not None]

    # Calculate metrics
    num_exact_matches = 0
    num_questions = len(all_results)

    for result in tqdm(all_results, desc="Calculating metrics"):
        if not result.get("error"):
            gold_answers = result.get("gold_answers", [])
            final_answer = result.get("model_answer", "")

            if gold_answers == []:
                continue

            em_result = False
            if final_answer:
                final_answer_inside_curly_braces = extract_answers_inside_curly_braces(
                    final_answer
                )

                if final_answer_inside_curly_braces:
                    em_result = any(
                        exact_match(pred_answer, gold_answer)
                        for pred_answer in final_answer_inside_curly_braces
                        for gold_answer in gold_answers
                    )
                else:
                    em_result = any(exact_match(final_answer, g) for g in gold_answers)

            if em_result:
                num_exact_matches += 1

            result["exact_match"] = em_result

    # Save final results (simplified naming without per-question branches)
    if START_FROM_INDEX > 0:
        final_output_path = f"{results_dir}/{len(questions)}_from_{START_FROM_INDEX}_{MODEL_NAME.replace('/', '-')}_{mode_str}_answers_{timestamp}_final.json"
    else:
        final_output_path = f"{results_dir}/{len(questions)}_{MODEL_NAME.replace('/', '-')}_{mode_str}_answers_{timestamp}_final.json"

    with open(final_output_path, "w", encoding="utf-8") as fout:
        json.dump(all_results, fout, indent=2, ensure_ascii=False)

    # Calculate timing statistics
    question_times = [
        res.get("processing_time", 0)
        for res in all_results
        if res.get("processing_time")
    ]
    total_time = sum(question_times)
    avg_time_per_question = total_time / len(question_times) if question_times else 0

    # Get usage statistics
    stats = usage_stats.get_stats()

    # Save efficiency statistics to file
    stats_output = {
        "dataset": DATASET,
        "method": "SoG",
        "model": MODEL_NAME,
        "num_questions": len(all_results),
        "total_llm_calls": stats["total_calls"],
        "total_input_tokens": stats["total_input_tokens"],
        "total_output_tokens": stats["total_output_tokens"],
        "total_tokens": stats["total_tokens"],
        "total_cached_tokens": stats["total_cached_tokens"],
        "total_reasoning_tokens": stats["total_reasoning_tokens"],
        "total_time_seconds": overall_duration,
        "avg_time_per_question_seconds": avg_time_per_question,
        "sample_mode": SAMPLE,
        "total_kg_queries": few_shot_llm_tools_unified_helper.KG_QUERY_COUNT_SOG,
    }

    stats_filename = f"{results_dir}/SoG_{DATASET}_efficiency_stats.json"
    with open(stats_filename, "w") as f:
        json.dump(stats_output, f, indent=2)

    # Print final metrics
    if num_questions > 0:
        final_em = num_exact_matches / num_questions
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        if START_FROM_INDEX > 0:
            print(f"  Started from index: {START_FROM_INDEX}")
        if IO_MODE:
            print(f"  Configuration: IO Mode (no examples, no tools)")
        else:
            print(f"  Configuration: {examples_str}, {tools_str}")
        print(f"  Exact Match: {num_exact_matches}/{num_questions} = {final_em:.3f}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("EFFICIENCY STATISTICS:")
        print(f"  Total LLM Calls: {stats['total_calls']}")
        print(f"  Total Input Tokens: {stats['total_input_tokens']}")
        print(f"  Total Output Tokens: {stats['total_output_tokens']}")
        print(f"  Total Tokens: {stats['total_tokens']}")
        print(f"  Total Cached Tokens: {stats['total_cached_tokens']}")
        print(f"  Total Time: {overall_duration:.2f}s")
        print(f"  Total Questions Processed: {len(all_results)}")
        print(f"  Avg Time per Question: {avg_time_per_question:.2f}s")
        print(
            f"  Total KG Queries Made: {few_shot_llm_tools_unified_helper.KG_QUERY_COUNT_SOG}"
        )
        print(f"  Statistics saved to: {stats_filename}")
        print("=" * 60)

    print(f"\nAll {DATASET} questions processed. Results saved to {final_output_path}")


if __name__ == "__main__":
    main()
