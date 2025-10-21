# 🍽️ NutriMood Chatbot - MCP Integration

A sophisticated food recommendation chatbot powered by AWS Bedrock (Claude 3 Sonnet) with Model Context Protocol (MCP) integration for structured food data access.

## 📋 Overview

NutriMood is an intelligent chatbot that provides personalized food recommendations based on:
- User's mood and preferences
- Dietary restrictions and goals
- Calorie requirements
- Conversational context
- Weather and occasions

### Key Features

✅ **AWS Bedrock Integration** - Uses Claude 3 Sonnet for natural, context-aware conversations  
✅ **MCP Protocol Support** - Structured access to food database via Model Context Protocol  
✅ **Streaming Responses** - Real-time streaming of chat responses for better UX  
✅ **Session Management** - Maintains conversation history and user preferences  
✅ **Smart Food Matching** - Keyword-based relevance scoring for accurate recommendations  
✅ **RESTful API** - Comprehensive FastAPI-based REST endpoints  
✅ **Multi-format Data** - Handles rich food data with nutrition, ingredients, and dietary info

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────┐
│       FastAPI Server            │
│  ┌──────────────────────────┐   │
│  │  Chat Endpoints          │   │
│  │  - /chat                 │   │
│  │  - /recommend            │   │
│  │  - /session              │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  MCP Endpoints           │   │
│  │  - /mcp/tools            │   │
│  │  - /mcp/resources        │   │
│  │  - /mcp/prompts          │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
       │           │
       ↓           ↓
┌──────────┐  ┌──────────┐
│ Bedrock  │  │   MCP    │
│ Service  │  │  Server  │
└──────────┘  └──────────┘
       │           │
       ↓           ↓
┌──────────┐  ┌──────────┐
│  Claude  │  │   Food   │
│  3 LLM   │  │ Service  │
└──────────┘  └──────────┘
                   │
                   ↓
            ┌─────────────┐
            │ Food Data   │
            │ (JSON)      │
            └─────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- AWS CLI configured (optional)
- Food data JSON file

### Installation

1. **Clone the repository** (or navigate to the perplx folder)
   ```bash
   cd perplx
   ```

2. **Run the setup script**
   ```bash
   bash setup_script.sh
   ```
   
   Or manually:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (Linux/Mac)
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # Copy the example env file
   cp env.example .env
   
   # Edit .env with your AWS credentials
   # Required:
   # - AWS_ACCESS_KEY_ID
   # - AWS_SECRET_ACCESS_KEY
   # - AWS_DEFAULT_REGION
   ```

4. **Ensure food data is available**
   
   The application looks for food data in the following locations (in order):
   - Path specified in `FOOD_DATA_PATH` environment variable
   - `data/Niloufer_data.json`
   - `../data/raw/Niloufer_data.json`
   - `data/food_items.json`

5. **Run the application**
   ```bash
   python main.py
   ```
   
   The server will start at `http://localhost:8000`

6. **Access API Documentation**
   
   Visit `http://localhost:8000/docs` for interactive Swagger UI documentation

## 📚 API Endpoints

### Chat Endpoints

#### POST `/chat`
Start or continue a conversation with the chatbot (JSON response).

**Request:**
```json
{
  "message": "I want something spicy and healthy",
  "session_id": "optional-session-id",
  "user_preferences": {
    "dietary": "vegetarian",
    "calorie_goal": 500
  }
}
```

**Response:** Clean JSON response:
```json
{
  "message": "Response text...",
  "session_id": "uuid",
  "food_recommendation_id": "id1,id2,id3"
}
```

#### POST `/chat/stream`
Streaming version of chat endpoint for real-time text display.

**Request:** Same as `/chat`

**Response:** Streaming text with metadata at the end:
```
Response text streaming...

__METADATA__:{"session_id": "uuid", "food_recommendation_id": "id1,id2"}
```

#### POST `/recommend`
Get food recommendations without conversation context.

**Request:**
```json
{
  "query": "healthy breakfast",
  "top_k": 5,
  "filters": {
    "category": "Breakfast",
    "max_calories": 400
  }
}
```

#### GET `/session/{session_id}`
Retrieve session history and information.

#### DELETE `/session/{session_id}`
Delete a session and its history.

#### GET `/foods`
List available food items with pagination and filtering.

**Query Parameters:**
- `category` - Filter by category
- `limit` - Number of items (default: 50)
- `offset` - Pagination offset

### MCP Endpoints

#### GET `/mcp/info`
Get MCP server information and capabilities.

#### GET `/mcp/tools`
List all available MCP tools.

#### POST `/mcp/tools/{tool_name}`
Execute an MCP tool.

**Available Tools:**
- `search_foods` - Search for food items
- `get_food_by_id` - Get specific food details
- `list_categories` - List all categories
- `get_food_statistics` - Get database stats
- `recommend_foods` - Get personalized recommendations

**Example:**
```bash
curl -X POST "http://localhost:8000/mcp/tools/search_foods" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "spicy",
    "max_calories": 300,
    "top_k": 5
  }'
```

#### GET `/mcp/resources`
List available MCP resources.

#### GET `/mcp/resources/{uri}`
Read an MCP resource.

**Available Resources:**
- `foods` - Complete food database
- `categories` - All food categories
- `statistics` - Database statistics

#### GET `/mcp/prompts`
List available MCP prompts for common use cases.

## 🎮 Usage Examples

### Interactive Mode (Client)

```bash
python "client_example (1).py" interactive
```

### Demo Mode

```bash
python "client_example (1).py" demo
```

### Python Client Usage

```python
from client_example import NutrimoodClient

client = NutrimoodClient()

# Chat with the bot
response = client.chat("I want something healthy and filling")

# Get recommendations
recommendations = client.get_recommendations(
    query="low calorie snacks",
    top_k=5,
    filters={"max_calories": 200}
)

# List foods by category
foods = client.list_foods(category="Snacks", limit=10)

# Get session history
history = client.get_session_history()
```

### MCP Tool Usage

```python
import requests

# Search foods via MCP
response = requests.post(
    "http://localhost:8000/mcp/tools/search_foods",
    json={
        "query": "healthy protein",
        "max_calories": 400,
        "top_k": 5
    }
)

# Get categories
response = requests.post(
    "http://localhost:8000/mcp/tools/list_categories",
    json={}
)

# Get food by ID
response = requests.post(
    "http://localhost:8000/mcp/tools/get_food_by_id",
    json={"food_id": "e754af8d-bb53-421a-ace5-c28ab216b4d2"}
)
```

## 📁 Project Structure

```
perplx/
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── env.example                 # Environment variables template
├── setup_script.sh            # Setup automation script
├── client_example (1).py      # Example client implementation
├── services/
│   ├── __init__.py
│   ├── bedrock_service.py     # AWS Bedrock integration
│   ├── food_service.py        # Food matching & management
│   ├── session_service.py     # Session management
│   └── mcp_server.py          # MCP protocol implementation
├── utils/
│   ├── __init__.py
│   └── response_formatter.py  # Response formatting utilities
├── tests/
│   └── test_api.py            # API tests
└── data/
    └── Niloufer_data.json     # Food database (or at ../data/raw/)
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `BEDROCK_MODEL_ID` | Bedrock model ID | `anthropic.claude-3-sonnet...` |
| `BEDROCK_MAX_TOKENS` | Max tokens per response | `1000` |
| `BEDROCK_TEMPERATURE` | Model temperature | `0.7` |
| `FOOD_DATA_PATH` | Path to food data JSON | `../data/raw/Niloufer_data.json` |
| `APP_HOST` | Server host | `0.0.0.0` |
| `APP_PORT` | Server port | `8000` |

### Bedrock Service Configuration

Edit `services/bedrock_service.py` to customize:
- Model selection
- Temperature and parameters
- System prompt personality
- Response streaming settings

### Food Service Configuration

Edit `services/food_service.py` to customize:
- Relevance scoring weights
- Keyword matching rules
- Filter logic
- Category mappings

## 🎯 MCP Integration Details

### What is MCP?

Model Context Protocol (MCP) is a standard protocol for providing structured context to Large Language Models. This implementation provides:

1. **Tools** - Callable functions for food operations
2. **Resources** - Read-only access to food data
3. **Prompts** - Templated prompts for common scenarios

### MCP Tools Available

| Tool | Purpose | Parameters |
|------|---------|------------|
| `search_foods` | Search food database | query, filters, top_k |
| `get_food_by_id` | Get specific food | food_id |
| `list_categories` | List all categories | - |
| `get_food_statistics` | Database stats | - |
| `recommend_foods` | Personalized recommendations | preferences, exclude_ids, top_k |

### MCP Resources Available

| Resource | URI | Description |
|----------|-----|-------------|
| All Foods | `nutrimood://foods` | Complete food database |
| Categories | `nutrimood://categories` | All categories |
| Statistics | `nutrimood://statistics` | Database statistics |

## 🤝 Integration with Claude Desktop

To integrate with Claude Desktop using MCP:

1. Configure Claude Desktop to use this MCP server
2. Point to the MCP endpoints at `http://localhost:8000/mcp/`
3. Claude can now use the food database as structured context

## 📊 Food Data Format

The system expects food data in the following JSON format:

```json
[
  {
    "Id": "uuid",
    "ProductName": "Food Name",
    "Description": "Description",
    "KioskCategoryName": "Category",
    "SubCategoryName": "Subcategory",
    "calories": 250,
    "Price": 380,
    "macronutrients": "{\"protein\": \"10g\", \"carbohydrates\": \"25g\"}",
    "ingredients": "[\"ingredient1\", \"ingredient2\"]",
    "dietary": "[\"vegetarian\", \"high-protein\"]",
    "Image": "image_url"
  }
]
```

## 🐛 Troubleshooting

### Food data not loading
- Check the `FOOD_DATA_PATH` environment variable
- Ensure the JSON file exists at the specified path
- Verify JSON file is valid

### AWS Bedrock errors
- Verify AWS credentials are configured correctly
- Check that Bedrock is enabled in your AWS region
- Ensure you have access to Claude models in Bedrock

### MCP server not initialized
- Ensure food data is loaded successfully before making MCP calls
- Check startup logs for initialization messages

## 📝 License

This project is part of the NutriMood AWS integration.

## 🙏 Acknowledgments

- AWS Bedrock for LLM capabilities
- FastAPI for the excellent web framework
- Model Context Protocol for structured LLM context

## 📧 Support

For issues or questions, please check the API documentation at `/docs` or review the test examples in `tests/test_api.py`.

