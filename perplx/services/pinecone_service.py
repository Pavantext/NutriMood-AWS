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
        self.bom_index_name = os.getenv("PINECONE_BOM_INDEX_NAME", "niloufer-bom")  # Separate index for user-uploaded BOM items
        
        if not self.api_key:
            print("⚠️  Warning: PINECONE_API_KEY not found in environment")
            self.client = None
            self.index = None
            self.bom_index = None
            return
        
        try:
            # Initialize Pinecone
            self.client = Pinecone(api_key=self.api_key)
            
            # Connect to main index (for original food items)
            self.index = self.client.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            print(f"✅ Connected to Pinecone index '{self.index_name}' with {total_vectors} vectors")
            
            # Try to connect to BOM index (for user-uploaded items)
            try:
                self.bom_index = self.client.Index(self.bom_index_name)
                bom_stats = self.bom_index.describe_index_stats()
                bom_vectors = bom_stats.get('total_vector_count', 0)
                print(f"✅ Connected to BOM Pinecone index '{self.bom_index_name}' with {bom_vectors} vectors")
            except Exception as e:
                print(f"⚠️  Warning: Could not connect to BOM index '{self.bom_index_name}': {e}")
                print(f"💡 BOM index will be created automatically on first upsert if it doesn't exist")
                self.bom_index = None
            
        except Exception as e:
            print(f"❌ Error connecting to Pinecone: {e}")
            self.client = None
            self.index = None
            self.bom_index = None
    
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
    
    def get_index_stats(self, use_bom_index: bool = False) -> Dict:
        """Get Pinecone index statistics"""
        target_index = self.bom_index if use_bom_index else self.index
        index_name = self.bom_index_name if use_bom_index else self.index_name
        
        if not target_index:
            return {"error": f"Pinecone index '{index_name}' not initialized"}
        
        try:
            stats = target_index.describe_index_stats()
            return {
                "index_name": index_name,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_bom_index_stats(self) -> Dict:
        """Get BOM index statistics"""
        return self.get_index_stats(use_bom_index=True)
    
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
    
    def upsert_food_item(
        self,
        food_id: str,
        embedding: List[float],
        food_data: Dict,
        use_bom_index: bool = False
    ) -> bool:
        """
        Upsert (insert or update) a food item in Pinecone
        
        Args:
            food_id: Unique identifier for the food item
            embedding: Vector embedding for the food item
            food_data: Dictionary containing food item metadata
            use_bom_index: If True, use BOM index (for user-uploaded items), otherwise use main index
        
        Returns:
            True if successful, False otherwise
        """
        # Select the appropriate index
        target_index = self.bom_index if use_bom_index else self.index
        
        if not target_index:
            if use_bom_index:
                # Try to create/connect to BOM index if it doesn't exist
                if self.client:
                    try:
                        # Try to get or create the BOM index
                        self.bom_index = self.client.Index(self.bom_index_name)
                        target_index = self.bom_index
                        print(f"✅ Connected to BOM index '{self.bom_index_name}'")
                    except Exception as e:
                        print(f"⚠️  BOM index '{self.bom_index_name}' not found. Please create it in Pinecone console.")
                        print(f"   Error: {e}")
                        return False
                else:
                    print("⚠️  Pinecone client not initialized, cannot upsert food item")
                    return False
            else:
                print("⚠️  Pinecone not initialized, cannot upsert food item")
                return False
        
        try:
            # Build metadata from food_data
            metadata = self._build_pinecone_metadata(food_data)
            
            # Upsert to Pinecone
            target_index.upsert(
                vectors=[{
                    "id": food_id,
                    "values": embedding,
                    "metadata": metadata
                }]
            )
            
            index_type = "BOM" if use_bom_index else "main"
            print(f"✅ Successfully upserted food item to {index_type} index: {food_data.get('product_name', food_id)}")
            return True
            
        except Exception as e:
            print(f"❌ Error upserting food item to Pinecone: {e}")
            return False
    
    def _build_pinecone_metadata(self, food_data: Dict) -> Dict:
        """
        Build Pinecone metadata dictionary from food data
        
        Args:
            food_data: Food item data dictionary
        
        Returns:
            Pinecone-compatible metadata dictionary
        """
        import json
        
        # Extract ingredients and dietary info
        ingredients_list = food_data.get('ingredients', [])
        if isinstance(ingredients_list, str):
            try:
                ingredients_list = json.loads(ingredients_list)
            except:
                ingredients_list = []
        
        dietary_list = food_data.get('dietary', [])
        if isinstance(dietary_list, str):
            try:
                dietary_list = json.loads(dietary_list)
            except:
                dietary_list = []
        
        # Extract macronutrients
        macros = food_data.get('macronutrients', {})
        if isinstance(macros, str):
            try:
                macros = json.loads(macros)
            except:
                macros = {}
        
        # Build metadata
        metadata = {
            'id': food_data.get('id', food_data.get('Id', '')),
            'product_name': food_data.get('product_name', food_data.get('ProductName', '')),
            'description': food_data.get('description', food_data.get('Description', '')),
            'category_name': food_data.get('category_name', food_data.get('KioskCategoryName', '')),
            'sub_category': food_data.get('sub_category', food_data.get('SubCategoryName', '')),
            'calories': int(food_data.get('calories', 0)),
            'price': float(food_data.get('price', food_data.get('Price', 0))),
            'image_url': food_data.get('image_url', food_data.get('Image', '')),
            'gst': float(food_data.get('gst', food_data.get('GST', 5))),
            'is_popular': bool(food_data.get('is_popular', food_data.get('IsPopular', False))),
            'ingredients_list': json.dumps(ingredients_list) if ingredients_list else '[]',
            'dietary_list': json.dumps(dietary_list) if dietary_list else '[]',
        }
        
        # Add macronutrients
        metadata['protein'] = str(macros.get('protein', '0g'))
        metadata['carbohydrates'] = str(macros.get('carbohydrates', '0g'))
        metadata['fat'] = str(macros.get('fat', '0g'))
        metadata['fiber'] = str(macros.get('fiber', '0g'))
        
        # Add dietary flags
        dietary_lower = [d.lower() for d in dietary_list] if isinstance(dietary_list, list) else []
        metadata['is_vegetarian'] = 'vegetarian' in dietary_lower or 'veg' in dietary_lower
        metadata['is_vegan'] = 'vegan' in dietary_lower
        metadata['is_gluten_free'] = 'gluten-free' in dietary_lower
        
        # Check for high protein (safely parse protein value)
        is_high_protein = 'high-protein' in dietary_lower
        if not is_high_protein:
            try:
                protein_str = str(macros.get('protein', '0g'))
                protein_value = float(protein_str.replace('g', '').replace('G', '').strip())
                is_high_protein = protein_value > 20
            except (ValueError, AttributeError):
                pass
        metadata['is_high_protein'] = is_high_protein
        
        metadata['is_low_calorie'] = int(metadata.get('calories', 0)) < 300
        
        return metadata

