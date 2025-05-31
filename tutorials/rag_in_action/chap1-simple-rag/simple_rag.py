import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.deepseek import DeepSeek

def simple_rag(embed_model):
    documents = SimpleDirectoryReader(input_files=["data/黑悟空/设定.txt"]).load_data() 
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist("data/黑悟空/index")
    # query_engine = index.as_query_engine(llm=DeepSeek(api_key=os.getenv("DEEPSEEK_API_KEY")))
    query_engine = index.as_query_engine()
    print(query_engine.query("黑神话悟空中有哪些战斗工具?"))


if __name__ == "__main__":
    assert load_dotenv()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    