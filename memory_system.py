from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
from llm_controller import LLMController
from retrievers import SimpleEmbeddingRetriever, ChromaRetriever
import json
import logging

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4",
                 evo_threshold: int = 3,
                 api_key: Optional[str] = None,
                 llm_controller = None):  
        """Initialize the memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            llm_controller: Optional custom LLM controller for testing
        """
        self.memories = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.chroma_retriever = ChromaRetriever()
        self.llm_controller = llm_controller or LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Evolution system prompt
        self._evolution_system_prompt = '''
        You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
        Analyze the new memory note and its nearest neighbors to determine if and how it should evolve.

        New memory:
        Content: {content}
        Context: {context}
        Keywords: {keywords}

        Nearest neighbors:
        {nearest_neighbors_memories}

        Based on this information, determine:
        1. Should this memory evolve? Consider its relationships with other memories
        2. What type of evolution should occur?
        3. What specific changes should be made?

        Return your decision in JSON format:
        {{
            "should_evolve": true/false,
            "evolution_type": ["update", "merge"],
            "reasoning": "Explanation for the decision",
            "affected_memories": ["memory_ids"],
            "evolution_details": {{
                "new_context": "Updated context",
                "new_keywords": ["keyword1", "keyword2"],
                "new_relationships": ["rel1", "rel2"]
            }}
        }}
        '''
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def create(self, content: str, **kwargs) -> str:
        """Create a new memory note.
        
        Args:
            content: The content of the memory
            **kwargs: Additional metadata (tags, category, etc.)
            
        Returns:
            str: ID of the created memory
        """
        # Create memory note
        note = MemoryNote(content=content, **kwargs)
        self.memories[note.id] = note
        
        # Add to retrievers
        metadata = {
            "context": note.context,
            "keywords": note.keywords,
            "tags": note.tags,
            "category": note.category,
            "timestamp": note.timestamp
        }
        self.chroma_retriever.add_document(document=content, metadata=metadata, doc_id=note.id)
        self.retriever.add_document(content)
        
        # First increment the counter
        self.evo_cnt += 1
        
        # Process evolution when threshold is reached
        if self.evo_cnt >= self.evo_threshold:
            evolved = self._process_memory_evolution(note)
            if evolved:
                self.evo_cnt = 0  # Reset after successful evolution
            # Keep current count for failed evolution, allowing further accumulation
        
        return note.id

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        metadata = {
            "context": note.context,
            "keywords": note.keywords,
            "tags": note.tags,
            "category": note.category,
            "timestamp": note.timestamp
        }
        self.chroma_retriever.delete_document(memory_id)
        self.chroma_retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.chroma_retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.chroma_retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.
        
        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        # Get results from ChromaDB
        chroma_results = self.chroma_retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': chroma_results['distances'][0][i]
                })
                
        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)
        
        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)
                    
        return memories[:k]
        
    def _process_memory_evolution(self, note: MemoryNote) -> bool:
        """Process potential memory evolution for a new note.
        
        Args:
            note: The new memory note to evaluate for evolution
            
        Returns:
            bool: Whether evolution occurred
        """
        # Get nearest neighbors
        neighbors = self.search(note.content, k=3)
        if not neighbors:
            return False
            
        # Format neighbors for LLM
        neighbors_text = "\n".join([
            f"Memory {i+1}:\n"
            f"Content: {mem['content']}\n"
            f"Context: {mem['context']}\n"
            f"Keywords: {mem['keywords']}\n"
            for i, mem in enumerate(neighbors)
        ])
        
        # Query LLM for evolution decision
        prompt = self._evolution_system_prompt.format(
            content=note.content,
            context=note.context,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbors_text
        )
        
        try:
            # Use mock_response directly
            if hasattr(self.llm_controller, 'mock_response'):
                response = self.llm_controller.mock_response
            else:
                response = self.llm_controller.get_completion(prompt)
                
            result = json.loads(response)
            
            if not result.get("should_evolve", False):
                return False
                
            # Process evolution based on type
            evolution_occurred = False
            if "merge" in result.get("evolution_type", []):
                # Merge memories
                affected_ids = result.get("affected_memories", [])
                evolution_details = result.get("evolution_details", {})
                
                for mem_id in affected_ids:
                    if mem_id in self.memories:
                        # Update the original memory with merged content
                        memory = self.memories[mem_id]
                        memory.context = evolution_details.get("new_context", memory.context)
                        memory.keywords = evolution_details.get("new_keywords", memory.keywords)
                        
                        # Add new relationships
                        new_relationships = evolution_details.get("new_relationships", [])
                        if not hasattr(memory, 'relationships'):
                            memory.relationships = []
                        memory.relationships.extend(new_relationships)
                        
                        # Remove the merged memory
                        self.delete(note.id)
                        evolution_occurred = True
                        
            elif "update" in result.get("evolution_type", []):
                # Update the memory
                evolution_details = result.get("evolution_details", {})
                note.context = evolution_details.get("new_context", note.context)
                note.keywords = evolution_details.get("new_keywords", note.keywords)
                
                # Add new relationships
                new_relationships = evolution_details.get("new_relationships", [])
                if not hasattr(note, 'relationships'):
                    note.relationships = []
                note.relationships.extend(new_relationships)
                
                evolution_occurred = True
                
            return evolution_occurred
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Error in memory evolution: {str(e)}")
            return False
