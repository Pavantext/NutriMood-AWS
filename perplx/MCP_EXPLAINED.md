# How MCP Works in Your NutriMood Project 🔧

## 🎯 **What is MCP?**

**MCP (Model Context Protocol)** is a standardized protocol that allows **LLMs to call functions and access data** in a structured way.

Think of it as:
- **REST API** = For humans and web apps
- **MCP** = For AI assistants and LLMs

---

## 🏗️ **MCP in Your Project**

### **You Have TWO APIs:**

```
┌─────────────────────────────────────────┐
│         YOUR FASTAPI SERVER             │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │ REST API     │  │ MCP API         │ │
│  │ (Humans)     │  │ (LLMs)          │ │
│  └──────────────┘  └─────────────────┘ │
│       │                    │            │
│       │                    │            │
│  /chat, /foods        /mcp/tools       │
│  /recommend           /mcp/resources   │
│  /session             /mcp/prompts     │
└─────────────────────────────────────────┘
           │                    │
           ↓                    ↓
    ┌──────────┐         ┌─────────────┐
    │ Your     │         │ Claude      │
    │ Frontend │         │ Desktop or  │
    │          │         │ Other LLMs  │
    └──────────┘         └─────────────┘
```

---

## 📚 **MCP Provides 3 Things**

### **1. TOOLS** - Functions LLMs can call

| Tool Name | What It Does | Example Use |
|-----------|--------------|-------------|
| `search_foods` | Search for food items | LLM: "Find spicy snacks under 300 cal" |
| `get_food_by_id` | Get specific food details | LLM: "Get details for ID abc123" |
| `list_categories` | List all food categories | LLM: "What categories exist?" |
| `get_food_statistics` | Database stats | LLM: "How many items total?" |
| `recommend_foods` | Personalized recommendations | LLM: "Recommend based on preferences" |

### **2. RESOURCES** - Read-only data access

| Resource URI | What It Contains |
|-------------|------------------|
| `nutrimood://foods` | All food items (up to 100) |
| `nutrimood://categories` | List of all categories |
| `nutrimood://statistics` | Database statistics |

### **3. PROMPTS** - Templates for common tasks

| Prompt Name | Purpose |
|-------------|---------|
| `food_recommendation` | Generate recommendations |
| `meal_planning` | Plan complete meals |

---

## 🔄 **How MCP Works: Step-by-Step Example**

### **Scenario: Claude Desktop wants to search for spicy foods**

```
┌──────────────────────────────────────────────────────────┐
│ STEP 1: LLM/Client Calls MCP Tool                        │
│ ─────────────────────────────────────────────            │
│ POST /mcp/tools/search_foods                             │
│ {                                                        │
│   "query": "spicy snacks",                               │
│   "max_calories": 300,                                   │
│   "top_k": 5                                             │
│ }                                                        │
└─────────────────────┬────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│ STEP 2: MCP Server Receives Request                      │
│ ─────────────────────────────────────────────            │
│ File: main.py (Line 366-377)                             │
│                                                          │
│ @app.post("/mcp/tools/{tool_name}")                      │
│ async def mcp_call_tool(tool_name, arguments):           │
│     result = mcp_server.call_tool(tool_name, arguments)  │
│     return result                                        │
└─────────────────────┬────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│ STEP 3: MCP Server Routes to Tool Function               │
│ ─────────────────────────────────────────────            │
│ File: mcp_server.py (Line 220-258)                       │
│                                                          │
│ def call_tool(tool_name, arguments):                     │
│     if tool_name == "search_foods":                      │
│         return self._tool_search_foods(arguments)        │
└─────────────────────┬────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│ STEP 4: Tool Calls FoodService                           │
│ ─────────────────────────────────────────────            │
│ File: mcp_server.py (Line 277-281)                       │
│                                                          │
│ matches = self.food_service.find_matching_foods(         │
│     query="spicy snacks",                                │
│     conversation_history=[],                             │
│     top_k=5,                                             │
│     filters={"max_calories": 300}                        │
│ )                                                        │
└─────────────────────┬────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│ STEP 5: FoodService → Pinecone Search                    │
│ ─────────────────────────────────────────────            │
│ 1. Generate embedding for "spicy snacks"                 │
│    → AWS Titan: [1024 numbers]                           │
│                                                          │
│ 2. Search Pinecone                                       │
│    → Returns top 5 with similarity scores                │
│                                                          │
│ 3. Apply calorie filter (max 300)                        │
│    → Filters out high-calorie items                      │
└─────────────────────┬────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────┐
│ STEP 6: Format and Return Results                        │
│ ─────────────────────────────────────────────            │
│ File: mcp_server.py (Line 285-301)                       │
│                                                          │
│ Return JSON:                                             │
│ {                                                        │
│   "query": "spicy snacks",                               │
│   "count": 5,                                            │
│   "results": [                                           │
│     {                                                    │
│       "id": "f02a3b4e-...",                              │
│       "name": "Peri Peri Fries",                         │
│       "description": "Spicy, crispy fries",              │
│       "category": "SNACKS",                              │
│       "calories": 280,                                   │
│       "price": 150,                                      │
│       "relevance_score": 0.89                            │
│     },                                                   │
│     ... (4 more items)                                   │
│   ]                                                      │
│ }                                                        │
└──────────────────────────────────────────────────────────┘
```

---

## 🔧 **MCP Tools Detailed**

### **Tool 1: search_foods**

**What LLM Sends:**
```json
POST /mcp/tools/search_foods
{
  "query": "healthy breakfast",
  "category": "BREAKFAST",
  "max_calories": 400
}
```

**What Happens:**
```python
# Line 277-281 in mcp_server.py
matches = self.food_service.find_matching_foods(
    query="healthy breakfast",
    top_k=5,
    filters={"category": "BREAKFAST", "max_calories": 400}
)

# This calls Pinecone:
# 1. Generate embedding for "healthy breakfast"
# 2. Search Pinecone index
# 3. Apply filters (category=BREAKFAST, calories≤400)
# 4. Return top 5 matches
```

**What LLM Gets Back:**
```json
{
  "query": "healthy breakfast",
  "count": 3,
  "results": [
    {"id": "825fad9f-...", "name": "Masala Quinoa", "calories": 250},
    {"id": "...", "name": "Oats", "calories": 180},
    {"id": "...", "name": "Fruit Salad", "calories": 120}
  ]
}
```

---

### **Tool 2: get_food_by_id**

**What LLM Sends:**
```json
POST /mcp/tools/get_food_by_id
{
  "food_id": "825fad9f-fec1-41c9-9722-6f084faac651"
}
```

**What Happens:**
```python
# Line 310 in mcp_server.py
food = self.food_service.get_food_by_id(food_id)

# This checks:
# 1. Pinecone (if available) - fetches by ID
# 2. Local cache (fallback)
```

**What LLM Gets Back:**
```json
{
  "food": {
    "Id": "825fad9f-fec1-41c9-9722-6f084faac651",
    "ProductName": "Masala Quinoa",
    "calories": 250,
    "Price": 276,
    "macronutrients": "{\"protein\":\"20g\",...}",
    ... (full details)
  }
}
```

---

### **Tool 3: list_categories**

**What LLM Sends:**
```json
POST /mcp/tools/list_categories
{}
```

**What LLM Gets:**
```json
{
  "categories": ["SNACKS", "BREAKFAST", "BEVERAGES", "MOCKTAILS", ...],
  "count": 8
}
```

---

### **Tool 4: get_food_statistics**

**What LLM Sends:**
```json
POST /mcp/tools/get_food_statistics
{}
```

**What LLM Gets:**
```json
{
  "total_items": 81,
  "categories": ["SNACKS", "BREAKFAST", ...],
  "category_count": 8,
  "average_calories": 285.5,
  "category_distribution": {
    "SNACKS": 25,
    "BREAKFAST": 15,
    "BEVERAGES": 20,
    ...
  }
}
```

---

### **Tool 5: recommend_foods**

**What LLM Sends:**
```json
POST /mcp/tools/recommend_foods
{
  "preferences": {
    "mood": "hungry",
    "dietary": "vegetarian",
    "calorie_max": 500
  },
  "exclude_ids": ["abc123"],
  "top_k": 3
}
```

**What Happens:**
```python
# Line 334-394 in mcp_server.py

# Builds query from preferences
query = "hungry vegetarian"  # Combines mood + dietary

# Builds filters
filters = {"max_calories": 500}

# Searches using FoodService (Pinecone)
matches = self.food_service.find_matching_foods(
    query=query,
    top_k=6,  # Gets extra to filter
    filters=filters
)

# Filters out excluded IDs
# Returns top 3
```

---

## 📁 **2. MCP RESOURCES**

Resources are **read-only data** that LLMs can access:

### **Resource 1: `nutrimood://foods`**

**What LLM Requests:**
```
GET /mcp/resources/foods
```

**What LLM Gets:**
```json
{
  "uri": "nutrimood://foods",
  "mimeType": "application/json",
  "content": [
    { "Id": "...", "ProductName": "Peri Peri Fries", ... },
    { "Id": "...", "ProductName": "Masala Quinoa", ... },
    ... (up to 100 items)
  ]
}
```

### **Resource 2: `nutrimood://categories`**

**Returns:** List of all food categories

### **Resource 3: `nutrimood://statistics`**

**Returns:** Database statistics

---

## 📝 **3. MCP PROMPTS**

Prompts are **templates** for common use cases:

### **Prompt 1: food_recommendation**

```python
# LLM can ask for this prompt template
prompt = mcp_server.get_prompt("food_recommendation", {
    "mood": "happy",
    "dietary_preference": "vegetarian",
    "calorie_goal": "500"
})

# Returns formatted prompt:
"""
Generate a food recommendation based on the following:
- User's mood: happy
- Dietary preference: vegetarian
- Calorie goal: 500

Provide 3-5 food recommendations with explanations.
"""
```

---

## 🔄 **Real-World Example: Claude Desktop Using MCP**

### **Scenario:** Claude Desktop app wants to help user find food

```
┌────────────────────────────────────────────────────────┐
│ USER (in Claude Desktop):                              │
│ "Find me healthy snacks under 300 calories"            │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ Claude Desktop:                                        │
│ "I should use the MCP tool 'search_foods'"            │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ Claude Desktop calls your MCP endpoint:                │
│ ────────────────────────────────────────              │
│ POST http://localhost:8000/mcp/tools/search_foods      │
│ {                                                      │
│   "query": "healthy snacks",                           │
│   "max_calories": 300,                                 │
│   "top_k": 5                                           │
│ }                                                      │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ Your MCP Server (mcp_server.py):                      │
│ ────────────────────────────────────────              │
│ 1. Receives tool call                                  │
│ 2. Calls FoodService.find_matching_foods()             │
│ 3. FoodService → AWS Titan → Pinecone                  │
│ 4. Gets top 5 healthy snacks under 300 cal             │
│ 5. Returns formatted results                           │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ Response to Claude Desktop:                            │
│ ────────────────────────────────────────              │
│ {                                                      │
│   "query": "healthy snacks",                           │
│   "count": 5,                                          │
│   "results": [                                         │
│     {"name": "Masala Quinoa", "calories": 250, ...},   │
│     {"name": "Fruit Salad", "calories": 120, ...},     │
│     ...                                                │
│   ]                                                    │
│ }                                                      │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ Claude Desktop reads results and responds to user:     │
│ ────────────────────────────────────────              │
│ "I found 5 healthy snacks under 300 calories:          │
│  1. Masala Quinoa (250 cal)                            │
│  2. Fruit Salad (120 cal)                              │
│  ..."                                                  │
└────────────────────────────────────────────────────────┘
```

---

## 📊 **MCP vs Regular Chat Endpoint**

### **Regular /chat Endpoint:**

```
User → /chat → FoodService → Pinecone → Claude → User
                                          ↓
                            Claude generates conversational response
```

**Use for:** Your web app, mobile app

### **MCP Endpoints:**

```
LLM → /mcp/tools/search_foods → FoodService → Pinecone → Structured Data → LLM
                                                                            ↓
                                            LLM uses data however it wants
```

**Use for:** Claude Desktop, AI assistants, developer tools

---

## 🎯 **MCP Endpoints in Your App**

### **GET /mcp/info**
**Returns:** Server information and capabilities

```json
{
  "name": "nutrimood-food-mcp",
  "version": "1.0.0",
  "protocol_version": "2024-01-01",
  "capabilities": {
    "tools": true,
    "resources": true,
    "prompts": true
  }
}
```

### **GET /mcp/tools**
**Returns:** List of all available tools

```json
{
  "tools": [
    {
      "name": "search_foods",
      "description": "Search for food items based on query and filters",
      "inputSchema": { ... }
    },
    ... (4 more tools)
  ]
}
```

### **POST /mcp/tools/{tool_name}**
**Executes:** A specific tool

```bash
curl -X POST "http://localhost:8000/mcp/tools/search_foods" \
  -H "Content-Type: application/json" \
  -d '{"query": "spicy", "max_calories": 300}'
```

### **GET /mcp/resources**
**Returns:** List of available resources

```json
{
  "resources": [
    {
      "uri": "nutrimood://foods",
      "name": "All Foods",
      "description": "Access to complete food database"
    },
    ...
  ]
}
```

### **GET /mcp/resources/{uri}**
**Returns:** Resource content

```bash
curl "http://localhost:8000/mcp/resources/foods"

# Returns all food items
```

### **GET /mcp/prompts**
**Returns:** Available prompt templates

---

## 💡 **Why Have MCP?**

### **For AI Assistants:**
```
Claude Desktop, GPT custom actions, etc. can:
1. Discover what tools are available (/mcp/tools)
2. Call tools to get food data (/mcp/tools/search_foods)
3. Access resources directly (/mcp/resources/foods)
4. Use structured data in their responses
```

### **For Developers:**
```
- Test food search without full chat context
- Integrate with other AI tools
- Build custom workflows
- Debug food matching logic
```

---

## 🔄 **MCP Data Flow**

```
┌─────────────────────────────────────────────────────────┐
│                   MCP CLIENT                            │
│            (Claude Desktop, Dev Tools)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ↓ POST /mcp/tools/search_foods
┌─────────────────────────────────────────────────────────┐
│               FASTAPI MCP ENDPOINT                      │
│              (main.py Line 366-377)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ↓ mcp_server.call_tool()
┌─────────────────────────────────────────────────────────┐
│                  MCP SERVER                             │
│              (mcp_server.py)                            │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Tool Router  │  │ Resource     │  │ Prompt       │ │
│  │              │  │ Handler      │  │ Generator    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼──────────────────┼──────────────────┼─────────┘
          │                  │                  │
          ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│                   FOOD SERVICE                          │
│         (Pinecone + AWS Titan Search)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────┐
│                  PINECONE INDEX                         │
│             (81 food vectors)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🎨 **Code Walkthrough**

### **1. MCP Server Initialization** (main.py)

```python
# Line 83-85 in main.py
global mcp_server
mcp_server = MCPServer(food_service)
print("✅ MCP Server initialized")
```

### **2. Tool Definition** (mcp_server.py Line 45-137)

```python
def _define_tools(self):
    return [
        {
            "name": "search_foods",
            "description": "Search for food items",
            "inputSchema": {
                "properties": {
                    "query": {"type": "string"},
                    "max_calories": {"type": "number"},
                    ...
                }
            }
        },
        ... (4 more tools)
    ]
```

### **3. Tool Execution** (mcp_server.py Line 260-301)

```python
def _tool_search_foods(self, args):
    query = args.get("query")
    filters = {
        "max_calories": args.get("max_calories"),
        "category": args.get("category"),
        ...
    }
    
    # THIS is where it uses Pinecone!
    matches = self.food_service.find_matching_foods(
        query=query,
        top_k=5,
        filters=filters
    )
    
    # Format for LLM
    return {"results": [...], "count": 5}
```

---

## 🎯 **Key Differences**

| Feature | /chat Endpoint | /mcp/tools Endpoint |
|---------|----------------|---------------------|
| **For** | Your web/mobile app | AI assistants, LLMs |
| **Input** | Natural language | Structured function calls |
| **Output** | Conversational text | Structured JSON data |
| **LLM** | Generates response | Uses data to generate |
| **Use Case** | Direct user chat | Tool-augmented AI |

---

## 🚀 **How to Use MCP**

### **Option 1: Test with cURL**

```bash
# Search foods
curl -X POST "http://localhost:8000/mcp/tools/search_foods" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "spicy snacks",
    "max_calories": 300,
    "top_k": 5
  }'

# Get food by ID
curl -X POST "http://localhost:8000/mcp/tools/get_food_by_id" \
  -H "Content-Type: application/json" \
  -d '{"food_id": "825fad9f-fec1-41c9-9722-6f084faac651"}'

# List categories
curl -X POST "http://localhost:8000/mcp/tools/list_categories" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### **Option 2: Python Client**

```python
import requests

# Call MCP tool
response = requests.post(
    'http://localhost:8000/mcp/tools/search_foods',
    json={
        "query": "healthy breakfast",
        "max_calories": 400
    }
)

results = response.json()
print(f"Found {results['count']} items:")
for item in results['results']:
    print(f"- {item['name']} ({item['calories']} cal)")
```

### **Option 3: Configure Claude Desktop**

Add to Claude Desktop's MCP config:

```json
{
  "mcpServers": {
    "nutrimood": {
      "url": "http://localhost:8000/mcp",
      "tools": ["search_foods", "get_food_by_id", "list_categories"]
    }
  }
}
```

Then Claude Desktop can use your food database!

---

## ✨ **Summary**

### **MCP in Your Project:**

✅ **Provides 5 tools** for structured food operations  
✅ **Provides 3 resources** for direct data access  
✅ **Provides 2 prompts** for common scenarios  
✅ **Uses same FoodService** as your chat endpoint  
✅ **Uses Pinecone** for semantic search  
✅ **Returns structured JSON** instead of conversational text  

### **How Data Flows:**

```
MCP Tool Call → MCP Server → FoodService → Pinecone Search → Results → MCP Client
```

### **Why It Exists:**

- 🤖 **For AI assistants** to access your food data
- 🔧 **For developers** to test and debug
- 🎯 **For integration** with other LLM tools
- 📊 **For structured data access** (not conversational)

---

**The MCP server is like a "special API for AIs" that sits alongside your regular chat API!** 🎉

Both use the same **FoodService** → **Pinecone** → **AWS Titan** search, but MCP returns **structured data** while chat returns **conversational responses**! 🚀
