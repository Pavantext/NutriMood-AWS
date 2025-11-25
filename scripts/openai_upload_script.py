"""
OpenAI Vector Store Setup Script for Responses API
Creates/updates vector stores and uploads JSON files for semantic search
Optimized for production use with Responses API
"""

import os
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import APIError, RateLimitError

# Load environment variables from project root or perplx directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
perplx_dir = project_root / "perplx"

# Try loading .env from multiple locations
env_files = [
    project_root / ".env",
    perplx_dir / ".env",
    Path.cwd() / ".env"
]

for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file)
        break
else:
    # If no .env found, try default load_dotenv() behavior
    load_dotenv()


class VectorStoreManager:
    """Production-grade vector store manager for OpenAI Responses API"""
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """
        Initialize the vector store manager.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            max_retries: Maximum number of retries for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.max_retries = max_retries
    
    def _retry_api_call(self, func, *args, **kwargs):
        """Retry API call with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 1  # Exponential backoff
                print(f"⚠️  Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 0.5
                print(f"⚠️  API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def validate_json_file(self, json_path: Path) -> Dict[str, Any]:
        """
        Validate JSON file structure and content.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Dictionary with validation results and metadata
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Check file size (OpenAI has limits)
        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 512:  # OpenAI limit is 512MB per file
            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max 512MB)")
        
        print(f"📄 File size: {file_size_mb:.2f}MB")
        
        # Validate JSON structure
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                food_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        if not isinstance(food_data, list):
            raise ValueError("JSON file must contain an array of food items")
        
        if len(food_data) == 0:
            raise ValueError("JSON file is empty")
        
        # Validate required fields in first item
        if food_data[0]:
            required_fields = ["Id", "ProductName"]
            missing_fields = [field for field in required_fields if field not in food_data[0]]
            if missing_fields:
                print(f"⚠️  Warning: Missing recommended fields: {missing_fields}")
        
        return {
            "valid": True,
            "item_count": len(food_data),
            "file_size_mb": file_size_mb,
            "sample_item": food_data[0] if food_data else None
        }
    
    def create_vector_store(
        self,
        name: str = "NutriMood_Menu_Store",
        expires_after_days: Optional[int] = None
    ) -> str:
        """
        Create a new vector store.
        
        Args:
            name: Name of the vector store
            expires_after_days: Days until auto-delete after inactivity (None = no expiration)
            
        Returns:
            Vector store ID
        """
        print(f"📝 Creating vector store: {name}")
        
        create_params = {"name": name}
        if expires_after_days:
            create_params["expires_after"] = {
                "anchor": "last_active_at",
                "days": expires_after_days
            }
        
        vector_store = self._retry_api_call(
            self.client.vector_stores.create,
            **create_params
        )
        
        print(f"✅ Vector Store created: {vector_store.id}")
        return vector_store.id
    
    def get_vector_store(self, vector_store_id: str) -> Dict[str, Any]:
        """
        Retrieve vector store information.
        
        Args:
            vector_store_id: Vector store ID
            
        Returns:
            Vector store information
        """
        vector_store = self._retry_api_call(
            self.client.vector_stores.retrieve,
            vector_store_id
        )
        return {
            "id": vector_store.id,
            "name": vector_store.name,
            "status": vector_store.status,
            "file_counts": {
                "in_progress": vector_store.file_counts.in_progress,
                "completed": vector_store.file_counts.completed,
                "failed": vector_store.file_counts.failed,
                "cancelled": vector_store.file_counts.cancelled
            }
        }
    
    def list_vector_store_files(self, vector_store_id: str) -> List[str]:
        """
        List all files in a vector store.
        
        Args:
            vector_store_id: Vector store ID
            
        Returns:
            List of file IDs
        """
        files = self.client.vector_stores.files.list(vector_store_id=vector_store_id)
        return [file.id for file in files.data]
    
    def delete_vector_store_files(self, vector_store_id: str, file_ids: List[str]) -> bool:
        """
        Delete files from vector store.
        
        Args:
            vector_store_id: Vector store ID
            file_ids: List of file IDs to delete
            
        Returns:
            True if successful
        """
        for file_id in file_ids:
            try:
                self.client.vector_stores.files.delete(
                    vector_store_id=vector_store_id,
                    file_id=file_id
                )
                print(f"🗑️  Deleted file: {file_id}")
            except Exception as e:
                print(f"⚠️  Failed to delete file {file_id}: {e}")
        return True
    
    def upload_file_to_vector_store(
        self,
        vector_store_id: str,
        json_path: Path,
        replace_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Upload JSON file to vector store with production-level error handling.
        
        Args:
            vector_store_id: Vector store ID
            json_path: Path to JSON file
            replace_existing: If True, delete existing files before uploading
            
        Returns:
            Upload result with file batch information
        """
        # Optionally replace existing files
        if replace_existing:
            print("🔄 Replacing existing files...")
            existing_files = self.list_vector_store_files(vector_store_id)
            if existing_files:
                self.delete_vector_store_files(vector_store_id, existing_files)
        
        # Upload file
        print(f"⬆️  Uploading file: {json_path.name}")
        print("   This may take a few minutes for large files...")
        
        with open(json_path, "rb") as file_stream:
            file_batch = self._retry_api_call(
                self.client.vector_stores.file_batches.upload_and_poll,
                vector_store_id=vector_store_id,
                files=[file_stream]
            )
        
        result = {
            "file_batch_id": file_batch.id,
            "status": file_batch.status,
            "file_counts": {
                "in_progress": file_batch.file_counts.in_progress,
                "completed": file_batch.file_counts.completed,
                "failed": file_batch.file_counts.failed,
                "cancelled": file_batch.file_counts.cancelled,
                "total": file_batch.file_counts.total
            }
        }
        
        print(f"✅ Upload batch complete!")
        print(f"   Status: {result['status']}")
        print(f"   Completed: {result['file_counts']['completed']}/{result['file_counts']['total']}")
        
        if result['file_counts']['failed'] > 0:
            print(f"⚠️  Failed files: {result['file_counts']['failed']}")
        
        return result
    
    def wait_for_vector_store_ready(
        self,
        vector_store_id: str,
        max_wait_time: int = 600,
        poll_interval: int = 10
    ) -> bool:
        """
        Wait for vector store to be ready (all files processed).
        
        Args:
            vector_store_id: Vector store ID
            max_wait_time: Maximum wait time in seconds (default: 10 minutes)
            poll_interval: Polling interval in seconds
            
        Returns:
            True if ready, False if timeout
        """
        print("⏳ Waiting for vector store processing to complete...")
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait_time:
            try:
                vs_info = self.get_vector_store(vector_store_id)
                status = vs_info["status"]
                
                # Show progress
                if status != last_status:
                    print(f"   Status: {status}")
                    last_status = status
                
                file_counts = vs_info["file_counts"]
                if file_counts["in_progress"] > 0:
                    print(
                        f"   Processing: {file_counts['completed']} completed, "
                        f"{file_counts['in_progress']} in progress",
                        end="\r"
                    )
                
                # Check if complete
                if status == "completed" and file_counts["in_progress"] == 0:
                    print(f"\n✅ Vector store is ready!")
                    print(f"   Total files: {file_counts['completed']}")
                    return True
                
                # Check if failed
                if status == "failed":
                    raise Exception("Vector store processing failed")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                if "not found" in str(e).lower():
                    raise Exception(f"Vector store not found: {vector_store_id}")
                raise
        
        print(f"\n⚠️  Timeout: Vector store processing exceeded {max_wait_time}s")
        return False
    
    def setup_vector_store(
        self,
        json_file_path: Optional[str] = None,
        vector_store_name: str = "NutriMood_Menu_Store",
        vector_store_id: Optional[str] = None,
        expires_after_days: Optional[int] = None,
        replace_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Complete vector store setup: validate, create/update, upload, and wait for ready.
        
        Args:
            json_file_path: Path to JSON file (defaults to updated_food_items.json)
            vector_store_name: Name for new vector store
            vector_store_id: Existing vector store ID to update
            expires_after_days: Days until auto-delete (None = no expiration)
            replace_existing: If True, replace existing files in vector store
            
        Returns:
            Dictionary with vector_store_id and metadata
        """
        print("=" * 70)
        print("🚀 OpenAI Vector Store Setup for Responses API")
        print("=" * 70)
        
        # Step 1: Validate JSON file
        print("\n📋 Step 1: Validating JSON File")
        print("-" * 70)
        
        if not json_file_path:
            script_dir = Path(__file__).parent
            json_file_path = script_dir.parent / "perplx" / "data" / "updated_food_items.json"
        
        json_path = Path(json_file_path)
        validation = self.validate_json_file(json_path)
        
        print(f"✅ File validated successfully")
        print(f"   Items: {validation['item_count']}")
        print(f"   Size: {validation['file_size_mb']:.2f}MB")
        
        # Step 2: Create or get vector store
        print("\n📦 Step 2: Setting up Vector Store")
        print("-" * 70)
        
        if vector_store_id:
            print(f"🔄 Using existing vector store: {vector_store_id}")
            try:
                vs_info = self.get_vector_store(vector_store_id)
                print(f"✅ Found: {vs_info['name']}")
                print(f"   Status: {vs_info['status']}")
                print(f"   Files: {vs_info['file_counts']['completed']} completed")
            except Exception as e:
                print(f"⚠️  Could not retrieve vector store: {e}")
                print("📝 Creating new vector store...")
                vector_store_id = self.create_vector_store(
                    name=vector_store_name,
                    expires_after_days=expires_after_days
                )
        else:
            vector_store_id = self.create_vector_store(
                name=vector_store_name,
                expires_after_days=expires_after_days
            )
        
        # Step 3: Upload file
        print("\n⬆️  Step 3: Uploading File to Vector Store")
        print("-" * 70)
        
        upload_result = self.upload_file_to_vector_store(
            vector_store_id=vector_store_id,
            json_path=json_path,
            replace_existing=replace_existing
        )
        
        # Step 4: Wait for processing
        print("\n⏳ Step 4: Waiting for Processing")
        print("-" * 70)
        
        is_ready = self.wait_for_vector_store_ready(vector_store_id)
        
        if not is_ready:
            print("⚠️  Warning: Vector store may still be processing")
        
        # Final status
        vs_info = self.get_vector_store(vector_store_id)
        
        result = {
            "vector_store_id": vector_store_id,
            "name": vs_info["name"],
            "status": vs_info["status"],
            "file_counts": vs_info["file_counts"],
            "upload_result": upload_result,
            "json_file": str(json_path),
            "item_count": validation["item_count"]
        }
        
        # Summary
        print("\n" + "=" * 70)
        print("✅ Setup Complete!")
        print("=" * 70)
        print(f"📦 Vector Store ID: {result['vector_store_id']}")
        print(f"📋 Name: {result['name']}")
        print(f"📊 Status: {result['status']}")
        print(f"📁 Files: {result['file_counts']['completed']} completed")
        print(f"🍔 Items: {result['item_count']}")
        print("\n💡 Use this vector_store_id in your Responses API calls:")
        print(f"   tools=[{{'type': 'file_search', 'vector_store_ids': ['{result['vector_store_id']}']}}]")
        print("=" * 70)
        
        return result


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Create/update OpenAI Vector Store for Responses API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses default paths)
  python scripts/openai_upload_script.py

  # Custom JSON file
  python scripts/openai_upload_script.py --json-file path/to/data.json

  # Update existing vector store
  python scripts/openai_upload_script.py --vector-store-id vs_xxxxx

  # Replace existing files
  python scripts/openai_upload_script.py --vector-store-id vs_xxxxx --replace-existing
        """
    )
    parser.add_argument(
        "--json-file",
        type=str,
        help="Path to JSON file (default: perplx/data/updated_food_items.json)"
    )
    parser.add_argument(
        "--vector-store-name",
        type=str,
        default="NutriMood_Menu_Store",
        help="Name for new vector store"
    )
    parser.add_argument(
        "--vector-store-id",
        type=str,
        help="Existing vector store ID to update"
    )
    parser.add_argument(
        "--expires-after-days",
        type=int,
        help="Days until vector store auto-deletes (default: no expiration)"
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing files in vector store"
    )
    parser.add_argument(
        "--max-wait-time",
        type=int,
        default=600,
        help="Maximum wait time for processing (seconds, default: 600)"
    )
    
    args = parser.parse_args()
    
    try:
        manager = VectorStoreManager()
        result = manager.setup_vector_store(
            json_file_path=args.json_file,
            vector_store_name=args.vector_store_name,
            vector_store_id=args.vector_store_id,
            expires_after_days=args.expires_after_days,
            replace_existing=args.replace_existing
        )
        
        # Save configuration
        config_file = Path(__file__).parent / "openai_vector_store_config.json"
        with open(config_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Configuration saved to: {config_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        if os.getenv("DEBUG", "").lower() == "true":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
