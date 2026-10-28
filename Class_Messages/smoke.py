from llama_cpp import Llama

llm = Llama(model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
            n_ctx=2048, n_gpu_layers=-1, seed=0, verbose=False)

schema = {
    "type": "object",
    "properties": {"tasks": {"type": "array", "items": {"type": "string"}}},
    "required": ["tasks"],
}

out = llm.create_chat_completion(
    messages=[{"role": "user",
               "content": "List the tasks in this message as JSON: "
                          "'Riya, please share the venue list by Tuesday 11am.'"}],
    response_format={"type": "json_object", "schema": schema},
    temperature=0.0, max_tokens=128,
)
print(out["choices"][0]["message"]["content"])
print(out["usage"])