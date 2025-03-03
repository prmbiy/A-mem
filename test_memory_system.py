import unittest
from memory_system import AgenticMemorySystem, MemoryNote
from llm_controller import BaseLLMController
from datetime import datetime

class TestAgenticMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend="openai",  # Use Ollama for testing to avoid API costs
            llm_model="gpt-4o-mini"
        )
        
    def test_create_memory(self):
        """Test creating a new memory."""
        content = "Test memory content"
        memory_id = self.memory_system.create(content)
        
        # Verify memory was created
        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        
    def test_read_memory(self):
        """Test reading a memory."""
        # Create a test memory
        content = "Test memory for reading"
        memory_id = self.memory_system.create(content)
        
        # Read the memory
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        
        # Try reading non-existent memory
        non_existent = self.memory_system.read("non_existent_id")
        self.assertIsNone(non_existent)
        
    def test_update_memory(self):
        """Test updating a memory."""
        # Create a test memory
        content = "Original content"
        memory_id = self.memory_system.create(content)
        
        # Update the memory
        new_content = "Updated content"
        success = self.memory_system.update(memory_id, content=new_content)
        self.assertTrue(success)
        
        # Verify update
        memory = self.memory_system.read(memory_id)
        self.assertEqual(memory.content, new_content)
        
        # Try updating non-existent memory
        success = self.memory_system.update("non_existent_id", content="test")
        self.assertFalse(success)
        
    def test_delete_memory(self):
        """Test deleting a memory."""
        # Create a test memory
        content = "Memory to delete"
        memory_id = self.memory_system.create(content)
        
        # Delete the memory
        success = self.memory_system.delete(memory_id)
        self.assertTrue(success)
        
        # Verify deletion
        memory = self.memory_system.read(memory_id)
        self.assertIsNone(memory)
        
        # Try deleting non-existent memory
        success = self.memory_system.delete("non_existent_id")
        self.assertFalse(success)
        
    def test_search_memories(self):
        """Test searching memories."""
        # Create test memories
        contents = [
            "Python programming language",
            "JavaScript web development",
            "Python data science",
            "Ruby on Rails framework",
            "Python machine learning"
        ]
        
        for content in contents:
            self.memory_system.create(content)
            
        # Search for Python-related memories
        results = self.memory_system.search("Python")
        self.assertGreater(len(results), 0)
        
    def test_memory_metadata(self):
        """Test memory metadata handling."""
        # Create a memory with metadata
        content = "Test memory with metadata"
        tags = ["test", "metadata"]
        category = "Testing"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        memory_id = self.memory_system.create(
            content=content,
            tags=tags,
            category=category,
            timestamp=timestamp
        )
        
        # Verify metadata
        memory = self.memory_system.read(memory_id)
        self.assertEqual(memory.tags, tags)
        self.assertEqual(memory.category, category)
        self.assertEqual(memory.timestamp, timestamp)
        
    def test_memory_evolution(self):
        """Test memory evolution system."""
        # Create related memories
        contents = [
            "Deep learning neural networks",
            "Neural network architectures",
            "Training deep neural networks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.create(content)
            memory_ids.append(memory_id)
            
        # Verify that memories have been properly evolved
        for memory_id in memory_ids:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory.tags)
            self.assertIsNotNone(memory.context)
            
if __name__ == '__main__':
    unittest.main()
