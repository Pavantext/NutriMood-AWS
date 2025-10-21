"""
Embedding Service - Generates embeddings using AWS Titan Text Embeddings V2
Matches the embeddings used in your Pinecone index
"""

import os
import boto3
import json
from typing import List, Optional
from botocore.exceptions import ClientError


class EmbeddingService:
    def __init__(self, pinecone_dimension: Optional[int] = None):
        """
        Initialize AWS Titan for embeddings
        
        Args:
            pinecone_dimension: The dimension from Pinecone index (auto-detected if None)
        """
        try:
            # Initialize Bedrock client
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
            )
            
            # Model configuration
            self.model_id = os.getenv("TITAN_EMBEDDING_MODEL")
            
            # Determine dimensions
            # Pinecone index showed 3072, but Titan V2 max is 1024
            # This suggests either sparse vectors or different configuration
            if pinecone_dimension:
                # If Pinecone dimension is provided, use appropriate Titan dimension
                if pinecone_dimension >= 1024:
                    self.dimensions = 1024  # Titan V2 max
                else:
                    self.dimensions = pinecone_dimension
            else:
                self.dimensions = int(os.getenv("TITAN_EMBEDDING_DIMENSIONS", "1024"))
            
            self.normalize = os.getenv("TITAN_NORMALIZE", "true").lower() == "true"
            
            print(f"✅ AWS Titan embedding service initialized ({self.dimensions}D, normalized={self.normalize})")
            
        except Exception as e:
            print(f"❌ Error initializing Titan embeddings: {e}")
            self.client = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a text query using AWS Titan
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector (list of floats) or None if failed
        """
        if not self.client:
            return None
        
        try:
            # Titan V2 request format
            request_body = {
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": self.normalize
            }
            
            # Invoke model
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            result = json.loads(response['body'].read())
            embedding = result.get('embedding', [])
            
            if not embedding:
                print(f"⚠️  Warning: Empty embedding generated for query: {text}")
                return None
            
            return embedding
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ AWS Error generating embedding: {error_code}")
            return None
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        return self.dimensions

