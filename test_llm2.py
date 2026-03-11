from local_llm import generate_llm
from retriever import get_retriever

TOP_K = 5
retriever = get_retriever()

def build_rag_prompt(user_input: str) -> str:
    retrieved_docs = retriever(user_input)[:TOP_K]
    context_text = "\n\n".join(retrieved_docs)

    prompt = f"""You are an expert IPC assistant. Only answer based on the following documentation.
Do NOT invent answers. If the answer is not in the docs, reply exactly: "I don't know".

--- DOCUMENTATION START ---
{context_text}
--- DOCUMENTATION END ---

User question: {user_input}
Answer:
"""
    return prompt.strip()

test_query = "What is a standard change?"
prompt = build_rag_prompt(test_query)
print("----- PROMPT -----")
print(prompt)

response = generate_llm(prompt)
print("----- LLM RESPONSE -----")
print(response)
