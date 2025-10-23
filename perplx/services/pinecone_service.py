"""
Pinecone Service - Handles vector search for food recommendations
"""

import os
from typing import List, Dict, Tuple, Optional
from pinecone import Pinecone


class PineconeService:
    def __init__(self):
        """Initialize Pinecone client"""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "niloufer-test")
        
        if not self.api_key:
            print("⚠️  Warning: PINECONE_API_KEY not found in environment")
            self.client = None
            self.index = None
            return
        
        try:
            # Initialize Pinecone
            self.client = Pinecone(api_key=self.api_key)
            
            # Connect to index
            self.index = self.client.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            print(f"✅ Connected to Pinecone index '{self.index_name}' with {total_vectors} vectors")
            
        except Exception as e:
            print(f"❌ Error connecting to Pinecone: {e}")
            self.client = None
            self.index = None
    
    def search_foods(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Search for food items using vector similarity
        
        Args:
            query_embedding: The embedding vector for the user's query
            top_k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of (food_metadata, similarity_score) tuples
        """
        if not self.index:
            print("⚠️  Pinecone not initialized, returning empty results")
            return []
        
        try:
            # Build filter dict for Pinecone
            pinecone_filter = self._build_pinecone_filter(filters) if filters else None
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Extract matches
            matches = []
            # Handle Pinecone QueryResponse object properly
            matches_list = results.matches if hasattr(results, 'matches') else []
            
            for match in matches_list:
                metadata = match.metadata if hasattr(match, 'metadata') else {}
                score = match.score if hasattr(match, 'score') else 0.0
                
                # Convert Pinecone metadata to food item format
                food_item = self._convert_metadata_to_food(metadata)
                matches.append((food_item, score))
            
            return matches
            
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            return []
    
    def _build_pinecone_filter(self, filters: Dict) -> Dict:
        """
        Build Pinecone metadata filter from user filters
        
        Args:
            filters: User-provided filters
        
        Returns:
            Pinecone-compatible filter dict
        """
        pinecone_filter = {}
        
        # Category filter
        if 'category' in filters:
            pinecone_filter['category_name'] = {"$eq": filters['category']}
        
        # Calorie filters
        if 'max_calories' in filters:
            pinecone_filter['calories'] = {"$lte": filters['max_calories']}
        
        if 'min_calories' in filters:
            if 'calories' in pinecone_filter:
                pinecone_filter['calories']['$gte'] = filters['min_calories']
            else:
                pinecone_filter['calories'] = {"$gte": filters['min_calories']}
        
        # Dietary filters (using boolean metadata)
        if 'dietary' in filters:
            dietary = filters['dietary'].lower()
            if dietary == 'vegetarian':
                pinecone_filter['is_vegetarian'] = {"$eq": True}
            elif dietary == 'vegan':
                pinecone_filter['is_vegan'] = {"$eq": True}
            elif dietary == 'gluten-free':
                pinecone_filter['is_gluten_free'] = {"$eq": True}
            elif dietary == 'high-protein':
                pinecone_filter['is_high_protein'] = {"$eq": True}
        
        # Low calorie filter
        if filters.get('low_calorie'):
            pinecone_filter['is_low_calorie'] = {"$eq": True}
        
        # Popular items
        if filters.get('popular'):
            pinecone_filter['is_popular'] = {"$eq": True}
        
        return pinecone_filter
    
    def _convert_metadata_to_food(self, metadata: Dict) -> Dict:
        """
        Convert Pinecone metadata to food item format
        
        Args:
            metadata: Pinecone metadata
        
        Returns:
            Food item dictionary
        """
        return {
            'Id': metadata.get('id', ''),
            'ProductName': metadata.get('product_name', ''),
            'Description': metadata.get('description', ''),
            'KioskCategoryName': metadata.get('category_name', ''),
            'SubCategoryName': metadata.get('sub_category', ''),
            'calories': metadata.get('calories', 0),
            'Price': metadata.get('price', 0),
            'Image': metadata.get('image_url', ''),
            'GST': metadata.get('gst', 5),
            'IsPopular': metadata.get('is_popular', False),
            'macronutrients': self._build_macronutrients_str(metadata),
            'ingredients': metadata.get('ingredients_list', '[]'),
            'dietary': metadata.get('dietary_list', '[]')
        }
    
    def _build_macronutrients_str(self, metadata: Dict) -> str:
        """Build macronutrients JSON string from metadata"""
        macros = {
            "protein": metadata.get('protein', '0g'),
            "carbohydrates": metadata.get('carbohydrates', '0g'),
            "fat": metadata.get('fat', '0g'),
            "fiber": metadata.get('fiber', '0g')
        }
        import json
        return json.dumps(macros)
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        if not self.index:
            return {"error": "Pinecone not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "index_name": self.index_name,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_food_by_id(self, food_id: str) -> Optional[Dict]:
        """
        Get a specific food item by ID from Pinecone
        
        Args:
            food_id: Food item ID
        
        Returns:
            Food item dictionary or None
        """
        if not self.index:
            return None
        
        try:
            result = self.index.fetch(ids=[food_id])
            
            # Handle Pinecone FetchResponse object properly
            vectors = result.vectors if hasattr(result, 'vectors') else {}
            
            if food_id in vectors:
                vector_data = vectors[food_id]
                metadata = vector_data.metadata if hasattr(vector_data, 'metadata') else {}
                return self._convert_metadata_to_food(metadata)
            
            return None
            
        except Exception as e:
            print(f"❌ Error fetching food by ID: {e}")
            return None

