import json
import boto3
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

def load_aws_config():
    """Load AWS configuration from .env file"""
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    # Check if credentials are loaded
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if not access_key or not secret_key:
        print("❌ AWS credentials not found in .env file!")
        print("💡 Make sure your .env file contains:")
        print("   AWS_ACCESS_KEY_ID=your-key-here")
        print("   AWS_SECRET_ACCESS_KEY=your-secret-here")
        print("   AWS_DEFAULT_REGION=us-east-1")
        return None, None, None
    
    # Mask credentials for display
    masked_key = access_key[:4] + "*" * 12 + access_key[-4:]
    print(f"✅ Loaded AWS credentials: {masked_key}")
    print(f"✅ Region: {region}")
    
    return access_key, secret_key, region

def test_aws_credentials(access_key, secret_key, region):
    """Test AWS credentials and permissions"""
    
    try:
        # Create session with loaded credentials
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Test basic AWS access
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Account: {identity['Account']}")
        print(f"✅ User/Role: {identity['Arn'].split('/')[-1]}")
        
        return session
        
    except NoCredentialsError:
        print("❌ Invalid AWS credentials!")
        return None
    except PartialCredentialsError:
        print("❌ Incomplete AWS credentials!")
        return None  
    except ClientError as e:
        print(f"❌ AWS Authentication Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected AWS error: {e}")
        return None

def find_and_test_titan_v2_model(session):
    """Find and test specifically Titan Text Embeddings V2 model"""
    
    # Target model (what you specifically want)
    target_model = "amazon.titan-embed-text-v2:0"
    
    try:
        bedrock = session.client('bedrock')
        models = bedrock.list_foundation_models()
        
        # Find all Titan embedding models
        titan_models = []
        for model in models['modelSummaries']:
            model_id = model['modelId']
            if 'titan' in model_id.lower() and 'embed' in model_id.lower():
                titan_models.append(model_id)
        
        print(f"✅ Found {len(titan_models)} Titan embedding models:")
        for model in titan_models:
            if model == target_model:
                print(f"   🎯 {model} (TARGET - Titan Text V2)")
            else:
                print(f"   - {model}")
        
        # Check if our target model is available
        if target_model not in titan_models:
            print(f"❌ Target model {target_model} not found!")
            print("💡 Available alternatives:")
            v2_alternatives = [m for m in titan_models if 'v2' in m]
            for alt in v2_alternatives:
                print(f"   - {alt}")
            return None
        
        # Test the target model
        print(f"🔍 Testing {target_model}...")
        client = session.client('bedrock-runtime')
        
        test_request = {
            "inputText": "test food item for nutrimood chatbot",
            "dimensions": 1024,
            "normalize": True
        }
        
        response = client.invoke_model(
            modelId=target_model,
            body=json.dumps(test_request)
        )
        
        result = json.loads(response['body'].read())
        if 'embedding' in result and len(result['embedding']) == 1024:
            print(f"✅ {target_model} working perfectly!")
            print(f"✅ Vector dimensions: {len(result['embedding'])}")
            print(f"✅ Normalized: {test_request['normalize']}")
            return target_model
        else:
            print(f"❌ {target_model} returned invalid response")
            return None
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code == 'ResourceNotFoundException':
            print(f"❌ {target_model} not available in region {session.region_name}")
            print("💡 Try these solutions:")
            print("   1. Enable model access in Bedrock console")
            print("   2. Try different region (us-west-2)")
            print("   3. Wait 5-10 minutes after enabling access")
        elif error_code == 'AccessDenied':
            print("❌ No access to Bedrock service!")
            print("💡 Add these permissions to your IAM user:")
            print("   - bedrock:ListFoundationModels")
            print("   - bedrock:InvokeModel")
        else:
            print(f"❌ Bedrock error: {error_code}")
        
        return None
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return None

def generate_embedding(session, text, model_id):
    """Generate embedding using Titan Text Embeddings V2"""
    
    try:
        client = session.client('bedrock-runtime')
        
        # Titan V2 request format (1024 dimensions, normalized)
        request_body = {
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        }
        
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Parse response
        result = json.loads(response['body'].read())
        embedding = result['embedding']
        
        return embedding
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code == 'ThrottlingException':
            print("  ⚠️  Rate limit hit, waiting...")
            time.sleep(2)
            return 'retry'
        elif error_code == 'ValidationException':
            print(f"  ❌ Invalid input (text length: {len(text)})")
            return None
        elif error_code == 'ResourceNotFoundException':
            print(f"  ❌ Model not found: {model_id}")
            return None
        elif error_code == 'AccessDenied':
            print("  ❌ No permission to invoke model")
            return None
        else:
            print(f"  ❌ API Error: {error_code}")
            return None
            
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return None

def process_all_embeddings(session, model_id, processed_data):
    """Process embeddings for all food items using Titan V2"""
    
    print(f"🚀 Generating embeddings using {model_id}")
    print(f"📊 Processing {len(processed_data)} food items...")
    print("🎯 Using Titan Text Embeddings V2 (1024D, normalized)")
    print("-" * 60)
    
    success_count = 0
    retry_count = 0
    failed_items = []
    
    for i, item in enumerate(processed_data, 1):
        food_name = item['name'][:35]  # Truncate for display
        print(f"[{i:2d}/{len(processed_data)}] {food_name:<35}", end=" ")
        
        # Generate embedding with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            result = generate_embedding(session, item['embedding_text'], model_id)
            
            if result == 'retry':
                retry_count += 1
                continue
            elif result is not None:
                item['embedding'] = result
                success_count += 1
                vector_size = len(result)
                print(f"✅ ({vector_size}D)")
                break
            else:
                if attempt == max_retries - 1:
                    failed_items.append(food_name)
                    print("❌ Failed")
                else:
                    print("🔄 Retry...", end="")
                    time.sleep(1)
        
        # Rate limiting - be nice to AWS
        if i % 5 == 0:
            time.sleep(0.5)
    
    # Summary
    print("-" * 60)
    print(f"📈 Results Summary:")
    print(f"   ✅ Successful: {success_count}/{len(processed_data)}")
    print(f"   🔄 Retries: {retry_count}")
    print(f"   ❌ Failed: {len(failed_items)}")
    
    if failed_items:
        print(f"   Failed items: {', '.join(failed_items[:3])}")
        if len(failed_items) > 3:
            print(f"   ... and {len(failed_items) - 3} more")
    
    return processed_data

def save_embeddings(data_with_embeddings, output_file):
    """Save data with embeddings to JSON file"""
    
    try:
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with proper formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_with_embeddings, f, indent=2, ensure_ascii=False)
        
        # Verify embeddings were saved
        items_with_embeddings = sum(1 for item in data_with_embeddings 
                                  if 'embedding' in item and item['embedding'])
        
        print(f"💾 Saved to: {output_file}")
        print(f"✅ Verified: {items_with_embeddings} items have embeddings")
        print(f"📁 File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    """Main function"""
    print("🍕 Nutrimood AWS Embedding Generator (Titan Text V2)")
    print("🎯 Forced to use: amazon.titan-embed-text-v2:0")
    print("=" * 60)
    
    # Step 1: Load AWS configuration from .env
    access_key, secret_key, region = load_aws_config()
    if not all([access_key, secret_key, region]):
        return
    
    # Step 2: Test AWS credentials
    session = test_aws_credentials(access_key, secret_key, region)
    if not session:
        return
    
    # Step 3: Find and test specifically Titan V2 model
    model_id = find_and_test_titan_v2_model(session)
    if not model_id:
        return
    
    # Step 4: Setup file paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    
    input_file = data_dir / 'processed' / 'Niloufer_data_processed.json'
    output_file = data_dir / 'embeddings' / 'Niloufer_data_with_embeddings.json'
    
    # Step 5: Load processed data
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("💡 Run data_processor.py first!")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f"📄 Loaded {len(processed_data)} food items from: {input_file.name}")
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return
    
    # Step 6: Generate embeddings
    data_with_embeddings = process_all_embeddings(session, model_id, processed_data)
    
    # Step 7: Save results
    if save_embeddings(data_with_embeddings, output_file):
        print("\n🎉 Embedding generation complete!")
        print("🎯 All embeddings generated using Titan Text Embeddings V2")
        print("📋 Next step: Load data into Aurora PostgreSQL database")
    else:
        print("\n❌ Failed to save embeddings")

if __name__ == "__main__":
    main()
