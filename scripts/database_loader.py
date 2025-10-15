import json
import psycopg2
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT



def create_connection(db_config):
    """Connect to Aurora PostgreSQL"""
    try:
        print(f"[*] Connecting to {db_config['DB_HOST']}...")
        
        conn = psycopg2.connect(
            host=db_config['DB_HOST'],
            database=db_config['DB_NAME'],
            user=db_config['DB_USER'],
            password=db_config['DB_PASSWORD'],
            port=db_config['DB_PORT']
        )
        
        print("[SUCCESS] Connected to Aurora PostgreSQL")
        return conn
        
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        print("[INFO] Make sure:")
        print("   1. Aurora cluster is running")
        print("   2. Security groups allow your IP")
        print("   3. Database credentials are correct")
        return None

def setup_database(conn):
    """Create table and enable pgvector"""
    cursor = conn.cursor()
    
    try:
        # Enable pgvector extension
        print("[*] Enabling pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Drop table if exists (for clean setup)
        cursor.execute("DROP TABLE IF EXISTS nutrition_data CASCADE;")
        
        # Create nutrition_data table
        print("[*] Creating nutrition_data table...")
        cursor.execute("""
            CREATE TABLE nutrition_data (
                id UUID PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                calories INTEGER,
                price REAL,
                protein REAL,
                carbohydrates REAL,
                fat REAL,
                fiber REAL,
                ingredients TEXT[],
                dietary_info TEXT[],
                embedding_text TEXT,
                embedding vector(1024)
            );
        """)
        
        conn.commit()
        print("[SUCCESS] Database setup complete")
        return True
        
    except Exception as e:
        print(f"[ERROR] Database setup failed: {e}")
        conn.rollback()
        return False

def load_food_data(conn):
    """Load all food items with embeddings into database"""
    
    # Path to your embeddings file
    data_file = Path(__file__).parent.parent / 'data' / 'embeddings' / 'Niloufer_data_with_embeddings.json'
    
    if not data_file.exists():
        print(f"[ERROR] Embeddings file not found: {data_file}")
        return False
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            food_items = json.load(f)
        
        print(f"[*] Loaded {len(food_items)} food items from embeddings file")
        
        cursor = conn.cursor()
        success_count = 0
        
        for i, item in enumerate(food_items, 1):
            try:
                cursor.execute("""
                    INSERT INTO nutrition_data 
                    (id, name, description, category, calories, price, protein, carbohydrates, fat, fiber, ingredients, dietary_info, embedding_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item['id'],
                    item['name'],
                    item['description'],
                    item['category'],
                    item['calories'],
                    item['price'],
                    item['protein'],
                    item['carbohydrates'],
                    item['fat'],
                    item['fiber'],
                    item['ingredients'],
                    item['dietary_info'],
                    item['embedding_text'],
                    item['embedding']
                ))
                
                success_count += 1
                print(f"[{i:2d}/{len(food_items)}] [OK] {item['name'][:40]}")
                
            except Exception as e:
                print(f"[{i:2d}/{len(food_items)}] [FAIL] {item['name'][:40]} - Error: {e}")
        
        conn.commit()
        print(f"[SUCCESS] Successfully loaded {success_count}/{len(food_items)} food items")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        conn.rollback()
        return False

def create_vector_index(conn):
    """Create HNSW index for fast vector search"""
    cursor = conn.cursor()
    
    try:
        print("[*] Creating HNSW vector index...")
        cursor.execute("""
            CREATE INDEX idx_nutrition_embedding 
            ON nutrition_data USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        
        conn.commit()
        print("[SUCCESS] Vector index created successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Index creation failed: {e}")
        conn.rollback()
        return False

def test_vector_search(conn):
    """Test vector similarity search"""
    cursor = conn.cursor()
    
    try:
        print("[*] Testing vector search...")
        
        # Get a sample embedding for testing
        cursor.execute("SELECT name, embedding FROM nutrition_data LIMIT 1")
        sample_name, sample_embedding = cursor.fetchone()
        
        # Find similar items
        cursor.execute("""
            SELECT name, calories, category, embedding <=> %s AS similarity
            FROM nutrition_data
            ORDER BY similarity
            LIMIT 5
        """, (sample_embedding,))
        
        results = cursor.fetchall()
        
        print(f"[*] Similar items to '{sample_name}':")
        for name, calories, category, similarity in results:
            print(f"   - {name[:30]:<30} ({category}, {calories} cal) - {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Vector search test failed: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("NUTRIMOOD DATABASE LOADER")
    print("Loading food data with embeddings into Aurora PostgreSQL")
    print("=" * 60)
    
    # Load configuration
    db_config = {
        'DB_HOST': DB_HOST,
        'DB_NAME': DB_NAME,
        'DB_USER': DB_USER,
        'DB_PASSWORD': DB_PASSWORD,
        'DB_PORT': DB_PORT
    }
    
    # Connect to database
    conn = create_connection(db_config)
    if not conn:
        print("\n[INFO] To fix database connection:")
        print("1. Make sure Aurora cluster is running")
        print("2. Add database credentials to .env file:")
        print("   DB_HOST=your-aurora-endpoint.cluster-xxxxx.eu-north-1.rds.amazonaws.com")
        print("   DB_NAME=nutrimood")
        print("   DB_USER=postgres")
        print("   DB_PASSWORD=your-password")
        print("   DB_PORT=5432")
        return
    
    try:
        # Setup database
        if not setup_database(conn):
            return
        
        # Load food data
        if not load_food_data(conn):
            return
        
        # Create vector index
        if not create_vector_index(conn):
            return
        
        # Test vector search
        if test_vector_search(conn):
            print("\n" + "=" * 60)
            print("[SUCCESS] Database setup complete!")
            print("[OK] All food data loaded with vector embeddings")
            print("[OK] Vector search index created")
            print("[OK] Vector similarity search working")
            print("\n[NEXT] Next step: Build the chatbot API!")
            print("=" * 60)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
