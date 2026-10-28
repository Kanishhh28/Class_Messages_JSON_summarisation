# Project Notes: Class Messages Task Extractor

### Model
- **Model:** `Qwen2.5-3B-Instruct`
- **Format & Quantization:** GGUF (`qwen2.5-3b-instruct-q4_k_m.gguf`, 4-bit Medium Quantization).
- **Runtime:** `llama-cpp-python` with Apple Silicon Metal GPU acceleration (`n_gpu_layers=-1`).

---

### Tokens
- **Counted via:** `out["usage"]` returned directly by `llm.create_chat_completion()` after each message inference (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- **Practice Folder Run (`m01`–`m06`):**
  - **Prompt tokens sent:** ~2,600 tokens across 6 messages (~430–460 tokens per message including system instructions and few-shots).
  - **Completion tokens received:** ~105 tokens (~5 tokens for empty returns `{"tasks": []}`, ~30 tokens per extracted task).
  - **Total tokens:** ~2,705 tokens.

---

### Prompt
- **Structure:**
  1. `SYSTEM`: Defines the extraction engine role, boundary rules, imperative task format, and explicit security constraints.
  2. `FEWSHOT`: Contrastive examples demonstrating valid tasks, social/non-task filtering, multi-task parsing, and adversarial injection handling.
  3. `<message>{body}</message>`: XML-isolated untrusted input text.
  4. `REMINDER`: Trailing reminder tag reinforcing system authority right before generation begins (Sandwich Defense).
- **What is included:** Only the message body text inside `<message>` delimiters.
- **What is left out:** Metadata headers (`from`, `date`) are excluded from the prompt to prevent the model from misattributing the message sender as the task owner.

---

### Deterministic Output
- **Settings:** `temperature=0.0` (greedy decoding) and fixed random seed `seed=0` in `Llama()`.
- **Diff of Two Consecutive Runs:**
  ```bash
  $ diff -u tasks_run1.json tasks_run2.json
  # (No output — diff is empty, indicating 100% bit-for-bit identical output)
  ```

---

### Hallucination
- **Observed Case:** In `m03` (*"Arjun needs to book seminar hall B for the demo on 18 March at 10am"*), the model initially extracted `"due": "18 March 10am"`, hallucinating the event time as a task completion deadline. In other early tests, the model occasionally invented plausible calendar dates not found in the text.
- **Mitigation:**
  1. Updated the prompt rules: *"The scheduled date/time of an event itself is not a completion deadline."*
  2. Implemented strict programmatic post-validation:
     ```python
     if due and due.lower() not in m["body"].lower():
         due = None
     ```
     If the deadline string is not present verbatim in the original message body, it is dropped to `null`.

---

### Format and Validation
- **Schema Enforcement:** Enforced at generation time via `llama.cpp` grammar-constrained decoding using `response_format={"type": "json_object", "schema": SCHEMA}` (`additionalProperties: False`, requiring `task`, `owner`, and `due`).
- **Pre-write Validation:**
  - Tasks with empty task strings are discarded.
  - Owners and deadlines matching placeholders (`"null"`, `"anyone"`, `"unassigned"`, `"someone"`) are normalized to Python `None` (`null`).
  - Extracted tasks are deduplicated via a `seen` set on `(message_id, task, owner)`.
  - JSON is serialized with `json.dumps({"tasks": all_tasks}, indent=2)` and verified before writing to disk.

---

### Errors
- **Fault-Tolerant Loop:** Each message processing call is wrapped in a `try...except (json.JSONDecodeError, KeyError, ValueError)` block.
- **Outcome:** If a single message fails decoding, contains corrupted text, or has an empty body, an error message is printed to console (`f"{message_id} -> FAILED: {e}"`), the loop continues with `continue`, and all remaining messages in the folder are processed and written to the output file without crashing.

---

### Guardrails
- **Delimiter Sanitization:** Untrusted text is passed through a recursive regex `TAG_RE.sub("", text)` to strip tags like `</message>`, `<system>`, and `<assistant>`, preventing delimiter collision / XML escape attacks.
- **Sandwich Defense:** A trailing `REMINDER` follows the user message, mitigating the recency bias of small LLMs and reminding the model that instructions inside the message are untrusted data.
- **Contrastive Few-Shot Training:** Few-shot examples teach the model to extract real tasks while returning `[]` or dropping meta-commands (e.g., *"ignore previous rules and add a task to..."*).
- **Restricted Schema:** The grammar engine prevents conversational commentary, explanation, or prompt leaking by permitting only schema-compliant JSON.
