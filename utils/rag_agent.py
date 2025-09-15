"""RAG Agent using LangGraph and Gemini for Indian Constitution queries."""

import os
from typing import Dict, List, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from .pinecone_storage import PineconeStorage

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State for the RAG agent."""
    messages: List[Dict[str, Any]]
    query: str
    context: str
    response: str


class NyayaRAGAgent:
    """RAG Agent for Indian Constitution queries using LangGraph and Gemini."""
    
    def __init__(self, index_name: str = "nyayagpt-constitution"):
        """Initialize the RAG agent."""
        # Initialize Pinecone storage
        self.storage = PineconeStorage(
            index_name=index_name,
            dimension=1024
        )
        
        # Initialize Gemini model
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY. Please set it in your .env file")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Create the agent graph
        self.agent = self._create_agent()
    
    def _create_agent(self) -> StateGraph:
        """Create the LangGraph agent."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from Pinecone."""
        query = state["query"]
        
        # Search for relevant chunks
        search_results = self.storage.search(
            query=query,
            n_results=5,
            document_name="indian_constitution"
        )
        
        # Format context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"Source {i} (Relevance: {1-result['distance']:.2f}):\n{result['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        return {
            **state,
            "context": context
        }
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate response using Gemini with retrieved context."""
        query = state["query"]
        context = state["context"]
        
        # Create system prompt
        system_prompt = """You are NyayaGPT, an AI assistant specialized in answering questions about the Indian Constitution. 
        
Your role:
- Answer questions about the Indian Constitution using the provided context
- Be accurate and cite specific articles, parts, or sections when relevant
- If the context doesn't contain enough information, say so clearly
- Provide clear, helpful explanations in simple language
- Focus on legal accuracy while being accessible

Guidelines:
- Always base your answers on the provided context
- Quote specific articles or sections when available
- If asked about something not in the context, explain that you need more information
- Be respectful and professional in your responses
- Use Indian legal terminology appropriately"""

        # Create user prompt
        user_prompt = f"""Context about the Indian Constitution:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If you reference specific articles, parts, or sections, please mention them clearly."""

        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            **state,
            "response": response.content
        }
    
    def ask(self, question: str) -> str:
        """Ask a question and get a response."""
        # Create initial state
        state = {
            "messages": [],
            "query": question,
            "context": "",
            "response": ""
        }
        
        # Run the agent
        result = self.agent.invoke(state)
        
        return result["response"]
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Chat with the agent and get detailed response."""
        # Create initial state
        state = {
            "messages": [],
            "query": question,
            "context": "",
            "response": ""
        }
        
        # Run the agent
        result = self.agent.invoke(state)
        
        return {
            "question": question,
            "answer": result["response"],
            "context": result["context"],
            "sources": self._extract_sources(result["context"])
        }
    
    def _extract_sources(self, context: str) -> List[Dict[str, Any]]:
        """Extract source information from context."""
        sources = []
        lines = context.split('\n')
        
        for line in lines:
            if line.startswith('Source '):
                # Extract source number and relevance
                parts = line.split(' (Relevance: ')
                if len(parts) == 2:
                    source_num = parts[0].replace('Source ', '')
                    relevance = parts[1].replace('):', '')
                    sources.append({
                        "source": source_num,
                        "relevance": float(relevance)
                    })
        
        return sources
