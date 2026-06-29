from llama_cpp import Llama

llm = Llama(
    model_path="C:/AI_Project/Mon3ter/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False
)

resp = llm.create_chat_completion(
    messages=[{"role": "user", "content": "你好，请用一句话自我介绍"}],
    max_tokens=64,
    temperature=0.7
)
print("模型回复：", resp["choices"][0]["message"]["content"])
print("✅ 大模型加载验证通过")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("C:/AI_Project/Mon3ter/models/embedding/bge-m3")
vector = model.encode("这是一条测试文本")

print(f"向量维度：{len(vector)}")
print("✅ 嵌入模型加载验证通过")