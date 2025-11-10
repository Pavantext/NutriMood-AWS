"""
Food Service - Handles food data management, matching, and recommendation logic
Uses Pinecone for semantic vector search
"""

import json
from typing import List, Dict, Tuple, Optional
import re
from pathlib import Path
from services.pinecone_service import PineconeService
from services.embedding_service import EmbeddingService


class FoodService:
    def __init__(self):
        """Initialize food service with Pinecone and embedding support"""
        self.food_items = []
        self.food_index = {}  # id -> food item mapping
        
        # Initialize Pinecone service first
        self.pinecone_service = PineconeService()
        
        # Get Pinecone dimension if available
        pinecone_dim = None
        if self.pinecone_service.index:
            stats = self.pinecone_service.get_index_stats()
            pinecone_dim = stats.get('dimension')
            if pinecone_dim:
                print(f"ℹ️  Pinecone index dimension: {pinecone_dim}")
        
        # Initialize embedding service with Pinecone dimension
        self.embedding_service = EmbeddingService(pinecone_dimension=pinecone_dim)
        
        # Check if vector search is available
        self.use_vector_search = (
            self.pinecone_service.index is not None and 
            self.embedding_service.client is not None
        )
        
        if self.use_vector_search:
            print("✅ Vector search enabled (Pinecone + AWS Titan)")
        else:
            print("⚠️  Vector search not available, using keyword matching fallback")
        
    def load_food_data(self, file_path: str):
        """
        Load food data from JSON file
        
        Args:
            file_path: Path to the food data JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.food_items = json.load(f)
            
            # Create index for quick lookups (handle both 'Id' and 'id' formats)
            self.food_index = {}
            for item in self.food_items:
                food_id = item.get('Id') or item.get('id')
                if food_id and str(food_id).strip():
                    self.food_index[str(food_id).strip()] = item
            
            print(f"✅ Loaded {len(self.food_items)} food items")
            
        except FileNotFoundError:
            print(f"⚠️  Warning: Food data file not found at {file_path}")
            self.food_items = []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing food data JSON: {e}")
            self.food_items = []
    
    def find_matching_foods(
        self,
        query: str,
        conversation_history: List[Dict],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Find food items matching the query using vector search (Pinecone) or keyword matching
        
        Args:
            query: User's search query
            conversation_history: Previous conversation messages
            top_k: Number of results to return
            filters: Optional filters (category, max_calories, etc.)
        
        Returns:
            List of (food_item, relevance_score) tuples
        """
        # Build enhanced query from conversation context
        enhanced_query = self._build_contextual_query(query, conversation_history)
        
        # Use vector search if available
        if self.use_vector_search:
            return self._find_with_vector_search(enhanced_query, top_k, filters, original_query=query)
        else:
            # Fallback to keyword matching
            return self._find_with_keyword_matching(enhanced_query, top_k, filters)
    
    def _build_contextual_query(self, query: str, conversation_history: List[Dict]) -> str:
        """
        Build enhanced query from conversation context
        
        Args:
            query: Current user query
            conversation_history: Previous messages
        
        Returns:
            Enhanced query string
        """
        # For simple queries, just use the query as-is
        query_lower = query.lower().strip()
        
        # If it's just a greeting, return as-is
        if query_lower in ['hi', 'hello', 'hey']:
            return query
        
        # If asking for specials/popular, enhance query to find Niloufer specials
        if any(word in query_lower for word in ['special', 'popular', 'signature', 'famous', 'must try', 'must-try']):
            # Enhance query to find Niloufer special items
            return "Niloufer special signature items " + query
        
        # Add context from recent conversation (last 2 messages)
        context_parts = [query]
        
        for msg in conversation_history[-2:]:
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                if content and content.lower() not in ['hi', 'hello', 'hey']:
                    context_parts.append(content)
        
        # Combine with spaces
        return ' '.join(context_parts)
    
    def _find_with_vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict],
        original_query: str = None
    ) -> List[Tuple[Dict, float]]:
        """
        Find foods using Pinecone vector search
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
        
        Returns:
            List of (food_item, score) tuples
        """
        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_embedding(query)
        
        if not query_embedding:
            print("⚠️  Failed to generate embedding, falling back to keyword matching")
            return self._find_with_keyword_matching(query, top_k, filters)
        
        # Enhance filters for special/popular queries
        enhanced_filters = filters.copy() if filters else {}
        
        # If query mentions special/popular, ensure signature items are included
        # Use original query for detection, not enhanced query
        detection_query = original_query.lower() if original_query else query.lower()
        if any(word in detection_query for word in ['special', 'popular', 'signature', 'famous', 'niloufer special']):
            # First try with popular filter
            popular_matches = self.pinecone_service.search_foods(
                query_embedding=query_embedding,
                top_k=top_k,
                filters={**enhanced_filters, 'popular': True}
            )
            
            # If we got results, ensure signature items are included
            if popular_matches:
                # Add signature items if not already present
                signature_items = self._get_signature_items()
                existing_ids = {food.get('Id') for food, _ in popular_matches}
                
                for sig_item in signature_items:
                    sig_id = sig_item.get('Id')
                    if sig_id not in existing_ids:
                        popular_matches.append((sig_item, 1.0))  # High score for signature items
                
                # Ensure we don't exceed top_k but prioritize signature items
                # Sort by score (signature items have score 1.0, so they'll be at the top)
                popular_matches.sort(key=lambda x: x[1], reverse=True)
                
                # Take top_k results, but ensure signature items are included
                final_matches = popular_matches[:top_k]
                
                # Double-check that all signature items are in the final results
                final_ids = {food.get('Id') for food, _ in final_matches}
                for sig_item in signature_items:
                    sig_id = sig_item.get('Id')
                    if sig_id not in final_ids:
                        # Replace the lowest-scoring item with this signature item
                        if len(final_matches) >= top_k:
                            final_matches[-1] = (sig_item, 1.0)
                        else:
                            final_matches.append((sig_item, 1.0))
                
                return final_matches
            
            # Otherwise, search without popular filter but query enhanced
        
        # Search Pinecone
        matches = self.pinecone_service.search_foods(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=enhanced_filters if enhanced_filters else None
        )
        
        # Post-process: Boost items with "Niloufer" in name for special queries
        if 'special' in detection_query or 'niloufer' in detection_query:
            boosted_matches = []
            for food, score in matches:
                name = food.get('ProductName', '').lower()
                if 'niloufer' in name or 'special' in name:
                    # Boost score for Niloufer special items
                    boosted_matches.append((food, score + 0.1))
                else:
                    boosted_matches.append((food, score))
            
            # Re-sort by boosted scores
            boosted_matches.sort(key=lambda x: x[1], reverse=True)
            return boosted_matches
        
        return matches
    
    def _get_signature_items(self) -> List[Dict]:
        """
        Get Niloufer signature items (hardcoded for reliability)
        These are the items that define Niloufer's identity
        """
        signature_names = [
            "Niloufer Special Tea",
            "Niloufer Special Coffee", 
            "Maska Bun",
            "Khara Bun"
        ]
        
        signature_items = []
        
        for name in signature_names:
            found = False
            # Try to find by name in local index first
            for food in self.food_items:
                if food.get('ProductName', '').lower() == name.lower():
                    signature_items.append(food)
                    found = True
                    break
            
            # If not found locally and Pinecone is available, try Pinecone
            if not found:
                if self.use_vector_search:
                    # Search Pinecone for this specific item
                    try:
                        # Generate embedding for the specific item name
                        embedding = self.embedding_service.generate_embedding(name)
                        if embedding:
                            matches = self.pinecone_service.search_foods(
                                query_embedding=embedding,
                                top_k=1
                            )
                            if matches:
                                signature_items.append(matches[0][0])
                    except Exception as e:
                        print(f"⚠️  Error finding signature item {name}: {e}")
        
        return signature_items
    
    def _find_with_keyword_matching(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict]
    ) -> List[Tuple[Dict, float]]:
        """
        Fallback keyword-based matching when vector search unavailable
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
        
        Returns:
            List of (food_item, score) tuples
        """
        if not self.food_items:
            return []
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        # Score each food item
        scored_foods = []
        for food in self.food_items:
            # Apply filters first
            if filters and not self._apply_filters(food, filters):
                continue
            
            score = self._calculate_relevance_score(food, keywords, query)
            if score > 0:
                scored_foods.append((food, score))
        
        # Sort by score and return top_k
        scored_foods.sort(key=lambda x: x[1], reverse=True)
        return scored_foods[:top_k]
    
    def _extract_keywords(self, query: str, conversation_history: List[Dict]) -> List[str]:
        """Extract relevant keywords from query and conversation"""
        keywords = []
        
        # Add current query keywords
        query_lower = query.lower()
        keywords.extend(query_lower.split())
        
        # Extract from recent conversation (last 2 messages)
        for msg in conversation_history[-2:]:
            content = msg.get('content', '').lower()
            keywords.extend(content.split())
        
        # Deduplicate
        return list(set(keywords))
    
    def _calculate_relevance_score(self, food: Dict, keywords: List[str], query: str) -> float:
        """Calculate relevance score for a food item"""
        score = 0.0
        query_lower = query.lower()
        
        # Get food attributes
        name = food.get('ProductName', '').lower()
        description = food.get('Description', '').lower()
        category = food.get('KioskCategoryName', '').lower()
        subcategory = food.get('SubCategoryName', '').lower()
        
        # Parse dietary info
        dietary_info = self._parse_json_field(food.get('dietary', '[]'))
        dietary_str = ' '.join(dietary_info).lower()
        
        # Parse ingredients
        ingredients = self._parse_json_field(food.get('ingredients', '[]'))
        ingredients_str = ' '.join(ingredients).lower()
        
        # Create searchable text
        searchable_text = f"{name} {description} {category} {subcategory} {dietary_str}"
        
        # Exact name match (highest priority)
        if query_lower in name:
            score += 10.0
        
        # Category match
        if any(keyword in category for keyword in keywords):
            score += 5.0
        
        # Description match
        if query_lower in description:
            score += 4.0
        
        # Keyword matching in searchable text
        for keyword in keywords:
            if keyword in searchable_text:
                score += 1.0
            if keyword in name:
                score += 2.0
        
        # Match dietary preferences (healthy, junk, etc.)
        dietary_mappings = {
            'healthy': ['low-calorie', 'high-protein', 'low-fat', 'vegetarian', 'vegan'],
            'junk': ['fried', 'burger', 'pizza', 'fries', 'cheese'],
            'spicy': ['spicy', 'hot', 'chili', 'pepper'],
            'sweet': ['sweet', 'dessert', 'chocolate', 'cake'],
            'protein': ['high-protein', 'chicken', 'egg', 'meat'],
            'vegetarian': ['vegetarian', 'veg'],
            'vegan': ['vegan']
        }
        
        for diet_key, diet_values in dietary_mappings.items():
            if diet_key in query_lower:
                if any(val in searchable_text or val in dietary_str for val in diet_values):
                    score += 3.0
        
        # Calorie-based scoring
        if 'low calorie' in query_lower or 'healthy' in query_lower:
            calories = food.get('calories', 1000)
            if calories and calories < 300:
                score += 2.0
        
        if 'high calorie' in query_lower or 'junk' in query_lower:
            calories = food.get('calories', 0)
            if calories and calories > 400:
                score += 2.0
        
        return score
    
    def _apply_filters(self, food: Dict, filters: Dict) -> bool:
        """Apply filters to food item"""
        # Category filter
        if 'category' in filters:
            if food.get('KioskCategoryName') != filters['category']:
                return False
        
        # Max calories filter
        if 'max_calories' in filters:
            calories = food.get('calories', 0)
            if calories and calories > filters['max_calories']:
                return False
        
        # Min calories filter
        if 'min_calories' in filters:
            calories = food.get('calories', 0)
            if not calories or calories < filters['min_calories']:
                return False
        
        # Dietary filter
        if 'dietary' in filters:
            dietary_info = self._parse_json_field(food.get('dietary', '[]'))
            if filters['dietary'] not in dietary_info:
                return False
        
        return True
    
    def _parse_json_field(self, field_value: str) -> List[str]:
        """Parse JSON string field (handles malformed JSON)"""
        if not field_value:
            return []
        
        try:
            # Try parsing as JSON
            if isinstance(field_value, str):
                # Clean up the string
                field_value = field_value.strip().rstrip(',')
                return json.loads(field_value)
            return field_value
        except json.JSONDecodeError:
            # If JSON parsing fails, try simple extraction
            matches = re.findall(r'"([^"]+)"', field_value)
            return matches
    
    def build_food_context(self, food_matches: List[Tuple[Dict, float]]) -> str:
        """
        Build context string from matched food items for LLM
        
        Args:
            food_matches: List of (food_item, score) tuples
        
        Returns:
            Formatted context string
        """
        if not food_matches:
            return "No matching food items found in database."
        
        context_parts = []
        id_reference = []
        
        for idx, (food, score) in enumerate(food_matches, 1):
            # Try both 'Id' and 'id' to handle different formats
            food_id = food.get('Id') or food.get('id')
            name = food.get('ProductName', '') or food.get('name', 'Unknown')
            description = food.get('Description', '')
            category = food.get('KioskCategoryName', 'N/A')
            calories = food.get('calories', 'N/A')
            price = food.get('Price', 'N/A')
            
            # Parse dietary info
            dietary_info = self._parse_json_field(food.get('dietary', '[]'))
            dietary_str = ', '.join(dietary_info) if dietary_info else 'N/A'
            
            # Format macronutrients
            macros = self._format_macronutrients(food.get('macronutrients', ''))
            
            # Food information WITHOUT ID in the main text
            food_context = f"""
{idx}. {name}
   - Category: {category}
   - Description: {description}
   - Calories: {calories} cal
   - Price: ₹{price}
   - Dietary: {dietary_str}
   - Nutrition: {macros}
""".strip()
            
            context_parts.append(food_context)
            
            # Keep ID reference separate (for internal tracking only)
            # Only add if food_id is valid
            if food_id and str(food_id).strip():
                id_reference.append(f"[{name} = {food_id}]")
        
        # Build final context with ID reference at the end (hidden from main view)
        main_context = "\n\n".join(context_parts)
        reference_section = "\n\nInternal Reference (do not include in response): " + ", ".join(id_reference)
        
        return main_context + reference_section
    
    def _format_macronutrients(self, macros_str: str) -> str:
        """Format macronutrients for display"""
        if not macros_str:
            return "N/A"
        
        try:
            macros_str = macros_str.strip().rstrip(',')
            macros = json.loads(macros_str)
            
            parts = []
            for key, value in macros.items():
                parts.append(f"{key.capitalize()}: {value}")
            
            return ", ".join(parts)
        except:
            return "N/A"
    
    def extract_food_ids_from_response(
        self,
        response: str,
        food_matches: List[Tuple[Dict, float]]
    ) -> List[str]:
        """
        Extract food IDs that were recommended in the LLM response
        Uses multiple matching strategies for reliability
        
        Args:
            response: LLM's response text
            food_matches: The food items that were provided as context
        
        Returns:
            List of food IDs mentioned in the response
        """
        if not food_matches:
            return []
        
        mentioned_ids = []
        response_lower = response.lower()
        
        # Clean response for better matching
        response_clean = response_lower.replace('!', ' ').replace('?', ' ').replace('.', ' ').replace(',', ' ')
        
        # Create synonym mapping for common variations
        synonyms = {
            'veggies': 'vegetables',
            'veggie': 'vegetables',
            'fries': 'fry',
            'burger': 'burger',
            'pizza': 'pizza'
        }
        
        # First pass: Try exact matches from food_matches
        for food, _ in food_matches:
            # Try both 'Id' and 'id' to handle different formats
            food_id = food.get('Id') or food.get('id')
            food_name = food.get('ProductName', '') or food.get('name', '')
            
            # Skip if missing essential data
            if not food_name or not food_id or not str(food_id).strip():
                continue
            
            food_name_lower = food_name.lower()
            
            # Method 1: Exact full name match
            if food_name_lower in response_lower:
                food_id_str = str(food_id).strip()
                if food_id_str and food_id_str not in mentioned_ids:
                    mentioned_ids.append(food_id_str)
                    print(f"   ✓ Matched '{food_name}' (exact) -> ID: {food_id_str}")
                continue
            
            # Method 2: Remove parentheses and special chars, then match
            # E.g., "Jalapeno Cheese Poppers (6.Pcs)" → "Jalapeno Cheese Poppers"
            clean_name = food_name_lower.split('(')[0].strip()
            if clean_name and clean_name in response_lower:
                food_id_str = str(food_id).strip()
                if food_id_str and food_id_str not in mentioned_ids:
                    mentioned_ids.append(food_id_str)
                    print(f"   ✓ Matched '{food_name}' (cleaned) -> ID: {food_id_str}")
                continue
            
            # Method 2b: Try with synonyms (e.g., "veggies" → "vegetables")
            # Check if response contains synonym and food name contains replacement
            for synonym, replacement in synonyms.items():
                if synonym in response_lower and replacement in clean_name:
                    # For better accuracy, check if other words from food name also appear
                    # E.g., "Sauteed Veggies" should match "Sauteed Vegetables"
                    other_words = [w for w in clean_name.split() if w != replacement and len(w) > 3]
                    if not other_words or any(w in response_lower for w in other_words):
                        food_id_str = str(food_id).strip()
                        if food_id_str and food_id_str not in mentioned_ids:
                            mentioned_ids.append(food_id_str)
                            print(f"   ✓ Matched '{food_name}' (synonym: {synonym}→{replacement}) -> ID: {food_id_str}")
                            break
            
            # Method 3: Check significant word combinations
            # For names like "Peri Peri Fries", check if "peri" AND "fries" appear
            name_words = [w.strip() for w in clean_name.split() if len(w) > 3]
            
            if len(name_words) >= 2:
                # Check if multiple words appear
                words_found = [w for w in name_words if w in response_clean]
                # Also check synonyms
                for synonym, replacement in synonyms.items():
                    if synonym in response_clean and replacement in name_words:
                        words_found.append(replacement)
                if len(words_found) >= 2:
                    food_id_str = str(food_id).strip()
                    if food_id_str and food_id_str not in mentioned_ids:
                        mentioned_ids.append(food_id_str)
                        print(f"   ✓ Matched '{food_name}' (multi-word) -> ID: {food_id_str}")
                    continue
        
        # Second pass: If LLM mentioned foods not in food_matches, search entire database
        # Extract potential food mentions from response
        response_words = set(response_clean.split())
        
        # Look for food names that might have been mentioned but weren't in food_matches
        for food in self.food_items:
            food_id = food.get('Id') or food.get('id')
            food_name = food.get('ProductName', '') or food.get('name', '')
            
            if not food_name or not food_id or str(food_id).strip() in mentioned_ids:
                continue
            
            food_name_lower = food_name.lower()
            clean_name = food_name_lower.split('(')[0].strip()
            
            # Check if this food was mentioned but not in food_matches
            # Only do exact/cleaned name matches to avoid false positives
            if clean_name in response_lower or food_name_lower in response_lower:
                food_id_str = str(food_id).strip()
                if food_id_str and food_id_str not in mentioned_ids:
                    mentioned_ids.append(food_id_str)
                    print(f"   ✓ Matched '{food_name}' (database search) -> ID: {food_id_str}")
                    continue
            
            # Try synonym matching for database search
            for synonym, replacement in synonyms.items():
                if synonym in response_lower:
                    # Check if food name contains the replacement
                    if replacement in clean_name:
                        # For better accuracy, check if other words from food name also appear
                        other_words = [w for w in clean_name.split() if w != replacement and len(w) > 3]
                        if not other_words or any(w in response_lower for w in other_words):
                            food_id_str = str(food_id).strip()
                            if food_id_str and food_id_str not in mentioned_ids:
                                mentioned_ids.append(food_id_str)
                                print(f"   ✓ Matched '{food_name}' (synonym search: {synonym}→{replacement}) -> ID: {food_id_str}")
                                break
        
        if not mentioned_ids:
            print(f"   ⚠️  No food IDs extracted from response")
        else:
            print(f"   ✓ Extracted {len(mentioned_ids)} food ID(s)")
        
        return mentioned_ids
    
    def get_food_by_id(self, food_id: str) -> Optional[Dict]:
        """
        Get food item by ID from Pinecone or local cache
        
        Args:
            food_id: Food item ID
        
        Returns:
            Food item dictionary or None
        """
        if not food_id or not str(food_id).strip():
            return None
        
        # Try Pinecone first if available
        if self.use_vector_search:
            food = self.pinecone_service.get_food_by_id(str(food_id).strip())
            if food:
                return food
        
        # Fallback to local index (try both 'Id' and 'id' keys)
        food_id_str = str(food_id).strip()
        food = self.food_index.get(food_id_str)
        if not food:
            # Try finding by iterating if direct lookup fails
            for item in self.food_items:
                if str(item.get('Id', '')).strip() == food_id_str or str(item.get('id', '')).strip() == food_id_str:
                    return item
        return food
    
    def get_all_foods(
        self,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get all food items with optional filtering
        
        Args:
            category: Optional category filter
            limit: Maximum number of items to return
            offset: Number of items to skip
        
        Returns:
            List of food items
        """
        filtered_foods = self.food_items
        
        if category:
            filtered_foods = [
                food for food in self.food_items
                if food.get('KioskCategoryName') == category
            ]
        
        # Apply pagination
        start = offset
        end = offset + limit
        
        return filtered_foods[start:end]
    
    def get_categories(self) -> List[str]:
        """Get all unique food categories"""
        categories = set()
        for food in self.food_items:
            category = food.get('KioskCategoryName')
            if category:
                categories.add(category)
        
        return sorted(list(categories))
    
    def get_food_statistics(self) -> Dict:
        """Get statistics about food database"""
        total_items = len(self.food_items)
        categories = self.get_categories()
        
        # Calculate average calories
        calories_list = [
            food.get('calories', 0) 
            for food in self.food_items 
            if food.get('calories')
        ]
        avg_calories = sum(calories_list) / len(calories_list) if calories_list else 0
        
        # Count by category
        category_counts = {}
        for food in self.food_items:
            category = food.get('KioskCategoryName', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_items": total_items,
            "categories": categories,
            "category_count": len(categories),
            "average_calories": round(avg_calories, 2),
            "category_distribution": category_counts
        }

