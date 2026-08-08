import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

class DomiEngine:
    def __init__(self, data_path="./data/company_policies.txt", db_dir="./domi_memory"):
        self.data_path = data_path
        self.db_dir = db_dir
        
        # 1. Local Vector Embeddings (Free, 100% Offline)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Local AI Brain via Ollama
        self.llm = ChatOllama(model="llama3", temperature=0.1)
        
        # 3. Initialize Memory (ChromaDB)
        self.vector_store = self._initialize_memory()
        
        # 4. Build RAG Chain
        self.chain = self._build_rag_chain()

    def _initialize_memory(self):
        """Loads data into ChromaDB if not already indexed."""
        if not os.path.exists(self.db_dir):
            loader = TextLoader(self.data_path)
            documents = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
            chunks = splitter.split_documents(documents)
            
            return Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings, 
                persist_directory=self.db_dir
            )
        else:
            return Chroma(
                persist_directory=self.db_dir, 
                embedding_function=self.embeddings
            )

    def _build_rag_chain(self):
        system_prompt = (
            "You are Domi, an enterprise AI assistant developed by Dopmin. "
            "Answer the user's question accurately using ONLY the provided context below. "
            "Keep your answers professional, concise, and structured with bullet points where appropriate. "
            "If the answer cannot be found in the context, respond with: "
            "'I do not have access to that information in my knowledge base.'\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, qa_chain)

    def query(self, question: str) -> str:
        """Sends a question to Domi and returns the response string."""
        response = self.chain.invoke({"input": question})
        return response["answer"]