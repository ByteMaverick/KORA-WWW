vllm serve Qwen/Qwen3-30B-A3B-Insutrct-2507 -tp 4 \
    --gpu_memory_utilization 0.95 --max_model_len 131072 \
    --enable-auto-tool-choice --tool-call-parser hermes --enable-reasoning --reasoning-parser deepseek_r1 \
    --disable-log-requests --port 9988 --disable-log-requests --host 0.0.0.0


vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 -tp 4 \
    --gpu_memory_utilization 0.95 --max_model_len 131072 \
    --enable-auto-tool-choice --tool-call-parser hermes --enable-reasoning --reasoning-parser deepseek_r1 \
    --disable-log-requests --port 9988 --disable-log-requests --host 0.0.0.0

# Main Experiments
for dataset in CWQ WebQSP SimpleQuestions GrailQA; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 IO_MODE=true NUM_EXAMPLES=-1 USE_TOOLS=true THINKING=true DEBUG=false MAX_CONCURRENCY=100 SAMPLE=true OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done

for dataset in CWQ WebQSP SimpleQuestions GrailQA; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 NUM_EXAMPLES=0 USE_TOOLS=true THINKING=true DEBUG=false MAX_CONCURRENCY=100 SAMPLE=true OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done

for dataset in CWQ WebQSP SimpleQuestions GrailQA; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 NUM_EXAMPLES=1 USE_TOOLS=true THINKING=true DEBUG=false MAX_CONCURRENCY=40 SAMPLE=true OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done

for dataset in CWQ WebQSP SimpleQuestions GrailQA; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 NUM_EXAMPLES=3 USE_TOOLS=true THINKING=true DEBUG=false MAX_CONCURRENCY=40 SAMPLE=true OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done

for dataset in CWQ WebQSP SimpleQuestions GrailQA; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 NUM_EXAMPLES=5 USE_TOOLS=true THINKING=true DEBUG=false MAX_CONCURRENCY=40 SAMPLE=true OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done

# Ablation Experiments
for dataset in SimpleQuestions GrailQA CWQ WebQSP; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 NUM_EXAMPLES=5 USE_TOOLS=true THINKING=false DEBUG=false MAX_CONCURRENCY=100 SAMPLE=true FORMAT=json OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done

for dataset in SimpleQuestions GrailQA CWQ WebQSP; do
    DATASET=$dataset MODEL_NAME=Qwen/Qwen3-30B-A3B-Thinking-2507 NUM_EXAMPLES=5 USE_TOOLS=true THINKING=false DEBUG=false MAX_CONCURRENCY=100 SAMPLE=true FORMAT=markdown OPENAI_API_BASE_URL="http://localhost:9988/v1" python main.py
done
