import os
from typing import Dict, List, Optional, Union, Any, Callable, Iterator

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    BaseMessage
)
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    HumanMessagePromptTemplate
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ClaudeLangChain:
    """
    A LangChain-based client for interacting with the Anthropic Claude API.
    
    This class provides enhanced methods using LangChain's abstractions to interact with Claude,
    including streaming, vectorized context, and conversation history management.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "claude-3-7-sonnet-20250219",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        context_path: Optional[str] = None,
    ):
        """
        Initialize the Claude LangChain client.
        
        Args:
            api_key: The API key for authentication. If None, will try to read from ANTHROPIC_API_KEY env var.
            model_name: The Claude model to use.
            temperature: The default temperature for generation.
            max_tokens: The default maximum number of tokens to generate.
            context_path: The path to the context file.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either through the constructor or ANTHROPIC_API_KEY environment variable")
        
        # Set up the Claude model via LangChain
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=self.api_key
        )

        # Create embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Initialize vector store as None
        self.vector_store = None

        # Set vector store based on context
        if context_path:
            self.set_context(context_path)

        # Default conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Default prompt template for chat
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Update the system prompt in the chat template.
        
        Args:
            system_prompt: The system prompt to use.
        """
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

    def set_context(self, file_path: str) -> None:
        """
        Add or remove documents from context.
        
        Args:
            file_path: The path to the file to add to context.
        """
        # Load and process the document
        documents = self.load_document(file_path)
        split_docs = self.split_documents(documents)

        # Create vector store (assuming documents are already processed)
        self.vector_store = Chroma.from_documents(
            documents=split_docs,  # Your documents here
            embedding=self.embeddings,
            persist_directory="chromadb/",
        )

    def load_document(self, file_path):
        """
        Load a document file into LangChain Document objects.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        # Determine file type and use appropriate loader
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            # Fallback to unstructured for other file types
            loader = UnstructuredFileLoader(file_path)
        
        # Load the document
        return loader.load()

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Split documents into smaller chunks for better processing.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def chat(
        self, 
        message: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
        history: Optional[List[BaseMessage]] = None
    ) -> str:
        """
        Send a message to Claude and get a response.
        
        Args:
            message: The user's message.
            system_prompt: Optional system prompt to override the default.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.
            history: Optional conversation history as LangChain messages.
            
        Returns:
            Claude's response as a string.
        """
        # Set up model with potential overrides
        model = self.llm
        if temperature is not None or max_tokens is not None:
            model_kwargs = {}
            if temperature is not None:
                model_kwargs["temperature"] = temperature
            if max_tokens is not None:
                model_kwargs["max_tokens"] = max_tokens
                
            model = ChatAnthropic(
                model=self.llm.model,
                anthropic_api_key=self.api_key,
                **model_kwargs
            )
        
        # Use custom system prompt if provided
        prompt = self.prompt
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ])
        
        # Prepare the messages or use provided history
        messages = history or []
        
        # Execute the chain
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"question": message, "chat_history": messages})
        
        # Update history with this exchange if using internal memory
        if not history:
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(response)
            
        return response
    
    def stream_chat(
        self, 
        message: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
        history: Optional[List[BaseMessage]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> Iterator[str]:
        """
        Stream a response from Claude.
        
        Args:
            message: The user's message.
            system_prompt: Optional system prompt to override the default.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.
            history: Optional conversation history as LangChain messages.
            callback: Optional callback function to process chunks.
            
        Yields:
            Chunks of Claude's response as they arrive.
        """
        # Set up model with potential overrides
        model = self.llm
        if temperature is not None or max_tokens is not None:
            model_kwargs = {}
            if temperature is not None:
                model_kwargs["temperature"] = temperature
            if max_tokens is not None:
                model_kwargs["max_tokens"] = max_tokens
                
            model = ChatAnthropic(
                model=self.llm.model,
                anthropic_api_key=self.api_key,
                **model_kwargs,
                streaming=True
            )
        else:
            # Ensure streaming is enabled
            model = ChatAnthropic(
                model=self.llm.model,
                anthropic_api_key=self.api_key,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
                streaming=True
            )
        
        # Use custom system prompt if provided
        prompt = self.prompt
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ])
        
        # Prepare the messages or use provided history
        messages = history or []
        
        # Set up a streaming chain
        chain = prompt | model | StrOutputParser()
        
        # Full response for memory
        full_response = ""
        
        # Stream the response
        for chunk in chain.stream({"question": message, "chat_history": messages}):
            if callback:
                callback(chunk)
            full_response += chunk
            yield chunk
        
        # Update history with this exchange if using internal memory
        if not history:
            self.memory.chat_memory.add_user_message(message)
            self.memory.chat_memory.add_ai_message(full_response)

    def create_retrieval_chain(
        self,
        vector_store: VectorStore,
        system_prompt: Optional[str] = None,
        chain_type: str = "stuff",
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> ConversationalRetrievalChain:
        """
        Create a retrieval-augmented generation chain with Claude and a vector store.
        
        Args:
            vector_store: The vector store to use for retrieving context.
            system_prompt: Optional system prompt for context.
            chain_type: Type of retrieval chain to use (e.g., "stuff", "map_reduce").
            search_kwargs: Optional arguments for the vector store search.
            
        Returns:
            A ConversationalRetrievalChain that can be invoked.
        """
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )
        
        # Create memory for the chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create standard system prompt with instructions for using context
        context_system_prompt = system_prompt or """
        You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        """
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            chain_type=chain_type,
            condense_question_prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content="Given the chat history and a new question, formulate a standalone question that captures all necessary context."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("New question: {question}")
            ])
        )
        
        return chain
    
    def query_with_context(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
        return_source_docs: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Query Claude with context from a vector store.
        
        Args:
            question: The user's question.
            system_prompt: Optional system prompt.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            search_kwargs: Optional arguments for vector search.
            return_source_docs: Whether to return source documents along with answer.
            
        Returns:
            Either the answer string or a dict with answer and source documents.
            
        Raises:
            ValueError: If no context has been set via set_context().
        """
        if self.vector_store is None:
            raise ValueError("No context has been set. Please call set_context() with a file path first.")
            
        # Set up model with potential overrides
        model = self.llm
        if temperature is not None or max_tokens is not None:
            model_kwargs = {}
            if temperature is not None:
                model_kwargs["temperature"] = temperature
            if max_tokens is not None:
                model_kwargs["max_tokens"] = max_tokens
                
            model = ChatAnthropic(
                model=self.llm.model,
                anthropic_api_key=self.api_key,
                **model_kwargs
            )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )
        
        # Create context-aware prompt
        context_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt or "You are a helpful assistant. Use the retrieved documents to answer the question."),
            MessagesPlaceholder(variable_name="context"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        # Get relevant documents
        docs = retriever.invoke(question)
        
        # Format documents as messages
        context_messages = []
        for doc in docs:
            context_messages.append(SystemMessage(content=f"Document: {doc.page_content}"))
        
        # Get response
        chain = context_prompt | model | StrOutputParser()
        response = chain.invoke({
            "context": context_messages,
            "question": question
        })
        
        if return_source_docs:
            return {
                "answer": response,
                "source_documents": docs
            }
        return response

    def get_conversation_history(self) -> List[BaseMessage]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation messages.
        """
        return self.memory.chat_memory.messages
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.memory.chat_memory.clear()
    
    def set_conversation_history(self, history: BaseChatMessageHistory) -> None:
        """
        Set a custom chat history object.
        
        Args:
            history: A LangChain chat history object.
        """
        self.memory.chat_memory = history


# Example usage
if __name__ == "__main__":
    # Create a Claude client
    claude = ClaudeLangChain(context_path="context/")
    
    # Simple chat
    response = claude.chat(
        message="What are three interesting facts about octopuses?",
        system_prompt="You are a marine biologist specializing in cephalopods."
    )
    print(f"Response: {response}")
    
    # Conversation with history
    history = [
        HumanMessage(content="Hello, I'd like to know about deep sea creatures."),
        AIMessage(content="Hello! I'd be happy to tell you about deep sea creatures. Is there a specific aspect or creature you're interested in?"),
    ]
    
    response = claude.chat(
        message="Tell me about giant squid.",
        history=history,
        temperature=0.9
    )
    print(f"Response with history: {response}")
    
    # Example of streaming (uncomment to use)
    """
    print("Streaming response:")
    for chunk in claude.stream_chat(
        message="Write a short poem about the ocean.",
        system_prompt="You are a poet who specializes in nature poetry."
    ):
        print(chunk, end='', flush=True)
    print("\nStreaming complete.")
    """
    
    # Example with vector store (requires additional setup)
    
    
    
    # Query with context
    result = claude.query_with_context(
        question="What is the capital of France?",
        return_source_docs=True
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['source_documents']}")