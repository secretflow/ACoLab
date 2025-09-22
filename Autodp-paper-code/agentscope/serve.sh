vllm serve /mnt/data3/nianke_multi_agent/model/QwQ-32B \
--port 8000 \
--reasoning-parser deepseek_r1 \
--max_model_len 4096 \
--enable-auto-tool-choice \
--tool-call-parser hermes
