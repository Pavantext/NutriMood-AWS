"""
Helper script to list available inference profiles in AWS Bedrock
Useful for finding the correct inference profile ID when using regions like ap-south-1
"""

import boto3
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

def load_aws_config():
    """Load AWS configuration from .env file"""
    # Try to load from perplx/.env first, then root .env
    env_paths = [
        Path(__file__).parent.parent / 'perplx' / '.env',
        Path(__file__).parent.parent / '.env'
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded .env from: {env_path}")
            break
    else:
        # Try loading from current directory
        load_dotenv()
    
    # Check if credentials are loaded
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
    
    if not access_key or not secret_key:
        print("❌ AWS credentials not found!")
        print("\n💡 Make sure your .env file contains:")
        print("   AWS_ACCESS_KEY_ID=your-key-here")
        print("   AWS_SECRET_ACCESS_KEY=your-secret-here")
        print("   AWS_DEFAULT_REGION=ap-south-1")
        print("\n   The script will look for .env in:")
        for env_path in env_paths:
            print(f"   - {env_path}")
        return None, None, None
    
    # Mask credentials for display
    masked_key = access_key[:4] + "*" * 12 + access_key[-4:] if len(access_key) > 8 else "****"
    print(f"✅ AWS Access Key: {masked_key}")
    print(f"✅ Region: {region}")
    
    return access_key, secret_key, region

def list_inference_profiles(region=None):
    """List all available inference profiles in the specified region"""
    
    # Load AWS credentials from .env
    access_key, secret_key, default_region = load_aws_config()
    
    if not access_key or not secret_key:
        return
    
    if not region:
        region = default_region or os.getenv('AWS_DEFAULT_REGION', "ap-south-1")
    
    print(f"\n🔍 Listing inference profiles in region: {region}")
    print("=" * 80)
    
    try:
        # Create Bedrock client with explicit credentials
        bedrock = boto3.client(
            'bedrock',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        # List inference profiles
        # Note: This uses the Bedrock control plane API
        try:
            response = bedrock.list_inference_profiles()
            profiles = response.get('inferenceProfileSummaries', [])
            
            if not profiles:
                print("❌ No inference profiles found in this region.")
                print("\n💡 To create an inference profile:")
                print("   1. Go to AWS Bedrock Console")
                print("   2. Navigate to 'Cross-Region Inference' or 'Inference Profiles'")
                print("   3. Create a new inference profile for your model")
                return
            
            print(f"✅ Found {len(profiles)} inference profile(s):\n")
            
            for i, profile in enumerate(profiles, 1):
                profile_id = profile.get('inferenceProfileId', 'N/A')
                profile_name = profile.get('inferenceProfileName', 'N/A')
                profile_arn = profile.get('inferenceProfileArn', 'N/A')
                
                print(f"{i}. Profile Name: {profile_name}")
                print(f"   Profile ID: {profile_id}")
                print(f"   Profile ARN: {profile_arn}")
                
                # Try to get more details
                try:
                    details = bedrock.get_inference_profile(inferenceProfileIdentifier=profile_id)
                    model_id = details.get('inferenceProfile', {}).get('targetModel', 'N/A')
                    print(f"   Target Model: {model_id}")
                except:
                    pass
                
                print()
            
            print("=" * 80)
            print("\n💡 To use an inference profile, set in your .env file:")
            print("   BEDROCK_INFERENCE_PROFILE_ID=<profile-id-or-arn>")
            print("\n   Example:")
            if profiles:
                example_id = profiles[0].get('inferenceProfileId', 'your-profile-id')
                print(f"   BEDROCK_INFERENCE_PROFILE_ID={example_id}")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                print("❌ Access denied. You need the following IAM permissions:")
                print("   - bedrock:ListInferenceProfiles")
                print("   - bedrock:GetInferenceProfile")
            elif error_code == 'ValidationException':
                print("❌ Validation error. The list_inference_profiles API may not be available in this region.")
                print("\n💡 Alternative: Find inference profiles via AWS Console:")
                print("   1. Go to AWS Bedrock Console")
                print("   2. Navigate to 'Cross-Region Inference' or 'Inference Profiles'")
                print("   3. Copy the Inference Profile ID or ARN")
            else:
                print(f"❌ Error: {error_code} - {e.response['Error']['Message']}")
    
    except NoCredentialsError:
        print("❌ AWS credentials not found!")
        print("\n💡 Please configure AWS credentials:")
        print("   1. Create a .env file in the project root or perplx/ directory")
        print("   2. Add your AWS credentials:")
        print("      AWS_ACCESS_KEY_ID=your-key-here")
        print("      AWS_SECRET_ACCESS_KEY=your-secret-here")
        print("      AWS_DEFAULT_REGION=ap-south-1")
        print("\n   Or set them as environment variables before running the script.")
    
    except PartialCredentialsError as e:
        print(f"❌ Incomplete AWS credentials: {e}")
        print("\n💡 Make sure both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")
    
    except Exception as e:
        error_msg = str(e)
        if "Unable to locate credentials" in error_msg or "NoCredentialsError" in error_msg:
            print("❌ AWS credentials not found!")
            print("\n💡 Please configure AWS credentials in your .env file or environment variables.")
        else:
            print(f"❌ Unexpected error: {e}")
            print("\n💡 Manual steps to find inference profile:")
            print("   1. Go to AWS Bedrock Console: https://console.aws.amazon.com/bedrock/")
            print("   2. Select your region (ap-south-1)")
            print("   3. Navigate to 'Cross-Region Inference' or 'Inference Profiles'")
            print("   4. Look for inference profiles containing 'claude-3-5-sonnet' or your model")
            print("   5. Copy the Inference Profile ID or ARN")
            print("   6. Set BEDROCK_INFERENCE_PROFILE_ID in your .env file")

if __name__ == "__main__":
    import sys
    
    region = sys.argv[1] if len(sys.argv) > 1 else None
    list_inference_profiles(region)

