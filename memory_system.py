from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
from llm_controller import LLMController
from retrievers import SimpleEmbeddingRetriever, ChromaRetriever
import json

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
    
    _evolution_system_prompt = """You are a memory evolution system. Your task is to analyze a new memory and its relationship to existing memories.
                    Given the content and context of a new memory, along with information about related existing memories, determine if the new memory should trigger any evolution.

                    New Memory:
                    Content: {content}
                    Context: {context}
                    Keywords: {keywords}

                    Related Existing Memories:
                    {nearest_neighbors_memories}

                    Analyze the relationship between the new memory and existing memories. Consider:
                    1. Semantic overlap and relationships
                    2. Potential for knowledge synthesis
                    3. Updates or corrections to existing information
                    4. Patterns or trends across memories

                    Provide your analysis in the following JSON format:
                    {{{{
                        "should_evolve": true/false,
                        "evolution_type": "update/merge/link",
                        "reasoning": "Explanation of why evolution is needed",
                        "affected_memories": ["list of memory IDs that should be evolved"],
                        "evolution_details": {{{{
                            "new_context": "Updated context if needed",
                            "new_keywords": ["updated", "keyword", "list"],
                            "new_relationships": ["memory_id1", "memory_id2"]
                        }}}}
                    }}}}
                    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None):
        """Initialize the memory system with its core components.
        
        Args:
            model_name (str): Name of the model for the SimpleEmbeddingRetriever
            llm_backend (str): Backend for the LLMController
            llm_model (str): Model name for the LLMController
            evo_threshold (int): Threshold for memory evolution
            api_key (Optional[str]): API key for the LLMController
        """
        self.memories = {}  # id -> MemoryNote
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.chroma_retriever = ChromaRetriever()
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self._evo_cnt = 0 
        self._evo_threshold = evo_threshold

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

    def create(self, content: str, timestamp: str = None, **kwargs) -> str:
        """Create a new memory note with analyzed metadata.
        
        This method:
        1. Analyzes the content using LLM to extract metadata
        2. Creates a new MemoryNote with the content and metadata
        3. Stores the note in memory and vector stores
        4. Triggers memory evolution processing
        
        Args:
            content (str): The text content for the new memory
            timestamp (Optional[str]): Creation timestamp
            **kwargs: Additional metadata to override analyzed values
            
        Returns:
            str: ID of the created memory note
        """
        # First analyze the content
        analysis = self.analyze_content(content)
        
        # Create note with analyzed metadata, but allow kwargs to override
        note_kwargs = {
            'content': content,
            'keywords': analysis["keywords"],
            'context': analysis["context"],
            'timestamp': timestamp
        }
        
        # Only add tags from analysis if not provided in kwargs
        if 'tags' not in kwargs:
            note_kwargs['tags'] = analysis["tags"]
            
        # Add any additional kwargs
        note_kwargs.update(kwargs)
        
        # Create the note
        note = MemoryNote(**note_kwargs)
        self.memories[note.id] = note
        
        # Add to vector stores
        metadata = {
            "keywords": note.keywords,
            "context": note.context,
            "category": note.category,
            "tags": note.tags,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed
        }
        self.chroma_retriever.add_document(content, metadata, note.id)
        self.retriever.add_document(content)
        
        # Process memory evolution
        self._process_memory_evolution(note)
        
        return note.id

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, content: str = None, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id (str): ID of the memory to update
            content (Optional[str]): New content
            **kwargs: Additional metadata to update
            
        Returns:
            bool: True if memory was updated, False if not found
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        if content:
            note.content = content
            # Re-analyze content if needed
            analysis = self.analyze_content(content)
            note.keywords = analysis["keywords"]
            note.context = analysis["context"]
            note.tags = analysis["tags"]
        
        # Update other metadata
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        metadata = {
            "keywords": note.keywords,
            "context": note.context,
            "category": note.category,
            "tags": note.tags,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed
        }
        self.chroma_retriever.delete_document(memory_id)
        self.chroma_retriever.add_document(note.content, metadata, memory_id)
        
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
        """Process potential evolution for a memory note.
        
        This method:
        1. Finds semantically related memories
        2. Analyzes relationships and potential evolution
        3. Updates memory metadata based on evolution decision
        
        The evolution can include:
        - Updating tags and keywords
        - Merging related memories
        - Creating new relationships
        
        Args:
            note (MemoryNote): The memory note to process
            
        Returns:
            bool: True if evolution occurred, False otherwise
        """
        # Get nearest neighbors
        related = self._search_raw(note.content, k=5)
        
        # Format nearest neighbors for prompt
        neighbors_text = ""
        for r in related:
            doc_id = r.get('id')
            if doc_id:
                memory = self.memories.get(doc_id)
                if memory:
                    neighbors_text += f"Content: {memory.content}\n"
                    neighbors_text += f"Context: {memory.context}\n"
                    neighbors_text += f"Keywords: {', '.join(memory.keywords)}\n\n"
        
        # Generate prompt for evolution analysis
        prompt = self._evolution_system_prompt.format(
            content=note.content,
            context=note.context,
            keywords=", ".join(note.keywords),
            nearest_neighbors_memories=neighbors_text
        )
        
        try:
            # Get evolution decision from LLM
            evolution_result = json.loads(self.llm_controller.get_completion(prompt))
            
            # Apply evolution actions if needed
            if evolution_result["should_evolve"]:
                # Update tags if evolution type is "update"
                if "update" in evolution_result["evolution_type"]:
                    note.tags.extend(evolution_result["evolution_details"]["new_keywords"])
                    note.tags = list(set(note.tags))  # Remove duplicates
                
                # Merge with related memories if evolution type is "merge"
                if "merge" in evolution_result["evolution_type"]:
                    for memory_id in evolution_result["affected_memories"]:
                        if memory_id in self.memories:
                            memory = self.memories[memory_id]
                            memory.context = evolution_result["evolution_details"]["new_context"]
                            memory.tags = evolution_result["evolution_details"]["new_keywords"]
                
            return evolution_result["should_evolve"]
            
        except Exception as e:
            print(f"Error in memory evolution: {e}")
            return False
