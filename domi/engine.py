import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class DomiEngine:
    def __init__(self, data_path="./data/company_policies.txt", db_dir="./domi_memory", model_name="llama3"):
        self.data_path = data_path
        self.db_dir = db_dir
        self.model_name = model_name

        # 1. Local Vector Embeddings (Free, 100% Offline)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. Local AI Brain via Ollama
        self.llm = self._create_llm()

        # 3. Initialize Memory (ChromaDB)
        self.vector_store = self._initialize_memory()
        # Limit the database to only return the 2 most relevant chunks.
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        self.chat_history = []

        # 4. Build prompt template for retrieval-augmented prompting
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are Domi, an enterprise AI assistant developed by Dopmin. "
                    "Answer the user's question accurately using ONLY the provided context below. "
                    "Keep your answers professional, concise, and structured with bullet points where appropriate. "
                    "If the answer cannot be found in the context, respond with: "
                    "'I do not have access to that information in my knowledge base.'\n\n"
                    "Context:\n{context}"
                ),
            ),
            ("human", "{input}"),
        ])

    def _create_llm(self):
        try:
            return ChatOllama(model=self.model_name, temperature=0, streaming=True)
        except Exception as exc:
            if self.model_name != "llama3":
                return ChatOllama(model="llama3", temperature=0, streaming=True)
            raise RuntimeError(f"Unable to initialize Ollama model '{self.model_name}'.") from exc

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
                persist_directory=self.db_dir,
            )
        else:
            return Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
            )

    def _build_context(self, question: str) -> str:
        retrieved_docs = self.retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        if self.chat_history:
            history_text = "\n\n".join(
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in self.chat_history[-2:]
            )
            return f"Recent conversation:\n{history_text}\n\nRelevant context:\n{context_text}"

        return context_text

    def _append_history(self, question: str, answer: str) -> None:
        self.chat_history.append({"user": question, "assistant": answer})
        self.chat_history = self.chat_history[-2:]

    def stream_query(self, question: str):
        """Streams the answer token-by-token from the local Ollama model."""
        context = self._build_context(question)
        messages = self.prompt.format_messages(input=question, context=context)
        full_response = []

        for chunk in self.llm.stream(messages):
            content = chunk.content
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            elif not isinstance(content, str):
                content = str(content)

            if content:
                full_response.append(content)
                yield content

        self._append_history(question, "".join(full_response))

    def query(self, question: str) -> str:
        """Sends a question to Domi and returns the response string."""
        return "".join(self.stream_query(question))