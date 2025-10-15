import json
from pathlib import Path

def clean_macros(macro_str):
    """Extract numbers from macro string like '{"protein": "10g"}'"""
    data = json.loads(macro_str.rstrip(','))
    return {k: float(v.replace('g', '')) for k, v in data.items()}

def safe_json_parse(json_str, default=None):
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except:
        return default or []

def process_food_item(item):
    """Convert raw food item to clean format"""
    macros = clean_macros(item.get('macronutrients', '{}'))
    
    processed = {
        'id': item['Id'],
        'name': item['ProductName'],
        'description': item['Description'],
        'category': item.get('KioskCategoryName', 'Unknown'),
        'calories': item.get('calories', 0),
        'price': item.get('Price', 0),
        'macronutrients': clean_macros(item.get('macronutrients', '{}')),
        'ingredients': safe_json_parse(item.get('ingredients', '[]')),
        'dietary_info': safe_json_parse(item.get('dietary', '[]')),
        **macros  # Unpack protein, carbs, fat, fiber
    }
    
    # Create embedding text
    embedding_parts = [
        f"Food: {processed['name']}",
        f"Description: {processed['description']}",
        f"Category: {processed['category']}",
        f"Calories: {processed['calories']}, Protein: {processed['protein']}g",
        f"macronutrients: {processed['macronutrients']}",
        f"Ingredients: {', '.join(processed['ingredients'])}" if processed['ingredients'] else "",
        f"Dietary: {', '.join(processed['dietary_info'])}" if processed['dietary_info'] else ""
    ]
    
    processed['embedding_text'] = '. '.join(filter(None, embedding_parts))
    return processed

def main():
    # Setup paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    
    input_file = data_dir / 'raw' / 'Niloufer_data.json'
    output_file = data_dir / 'processed' / 'Niloufer_data_processed.json'
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process data
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    processed_data = [process_food_item(item) for item in raw_data]
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed {len(processed_data)} food items")
    print(f"📄 Saved to: {output_file}")

if __name__ == "__main__":
    main()
