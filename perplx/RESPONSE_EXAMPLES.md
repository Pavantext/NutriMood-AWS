# Response Examples - Before & After Word Limit

## Example 1: Dessert Request

### User Query
```
"something for dessert"
```

### Before (169 words) ❌
```
Oooh desserts, my favorite! 😋 While our menu doesn't have traditional sweets, 
I can suggest some tasty treats that could satisfy your sweet tooth.

How about the refreshing Pina Colada? It's a tropical mocktail with pineapple 
and coconut flavors that's like a vacation in a glass! 🍍🥥🌴

Or if you prefer something with a little zing, the Virgin Mojito is a zingy 
and refreshing mocktail made with lime, mint, and sparkling water. It's like 
a sweet-tart party in your mouth! 🍸🌿

Both of these mocktails are vegetarian, gluten-free, and relatively low in 
calories compared to most desserts. They'll give you that sweet fix without 
too much guilt! 😇

Let me know if either of those appeal to your dessert cravings or if you'd 
like me to suggest something else from our menu. I'm happy to keep exploring 
yummy options with you! 🤗
```

### After (54 words) ✅
```
Try the Pina Colada! Tropical pineapple and coconut flavors - refreshing and sweet 🍍

Or the Virgin Mojito with lime and mint for something zingy 🍸

Both are vegetarian, gluten-free, and low-calorie. Perfect sweet fix! 😋
```

---

## Example 2: Spicy Food

### User Query
```
"I want something spicy"
```

### Before (might be long) ❌
```
Absolutely, I've got more sizzling suggestions for you, spice lover! 🌶️🔥 
How about adding some Paneer Hyderabadi to your fiery feast? This grilled 
sandwich packs a punch with its spicy paneer filling. It's vegetarian and 
gluten-free too!

To balance out the heat, you could sip on a refreshing Lemon Tea With Honey 
or the Green Lemon Tea With Honey. The honey adds a touch of sweetness to 
cool your palate. 😋

With this spicy-sweet combo, your taste buds are in for an unforgettable 
ride! Let me know if you need any other zingy recommendations! 🥵🔥
```

### After (45 words) ✅
```
Try Peri Peri Fries! Super spicy and crispy 🍟🔥

Or Paneer Hyderabadi - a grilled sandwich with spicy paneer filling 🌶️

Both pack serious heat! Which sounds better? 😋
```

---

## Example 3: Healthy Options

### User Query
```
"something healthy"
```

### Before (too long) ❌
```
Great choice! I love when people prioritize their health! 🥗💪 

Let me recommend the Masala Quinoa - it's a nutritious powerhouse! 
This dish is packed with protein, it's vegetarian, gluten-free, and has 
balanced macronutrients. Quinoa is a superfood that will keep you energized 
throughout the day.

You could also try the Chicken Salad if you prefer something with lean protein. 
It's fresh, light, and loaded with vegetables that are great for your health.

Both options are under 400 calories and provide excellent nutrition. They're 
perfect for maintaining a balanced diet while still enjoying delicious food! 
Let me know which one appeals to you more! 😊
```

### After (48 words) ✅
```
Try Masala Quinoa! High-protein, vegetarian, and gluten-free 💪

Or the Chicken Salad for lean protein with fresh veggies 🥗

Both under 400 calories and super nutritious! Which appeals to you? 😊
```

---

## Example 4: Detailed Request (Allowed to be longer)

### User Query
```
"Can you compare the Peri Peri Fries and Jalapeno Poppers in detail?"
```

### Response (135 words - OK because user asked for detail) ✅
```
Sure! Here's a detailed comparison 🌶️

**Peri Peri Fries:**
- Crispy fries with spicy peri peri seasoning
- 280 calories
- Lighter snack option
- Great for sharing
- Price: ₹150
- Best as a side dish

**Jalapeno Cheese Poppers:**
- 6 breaded jalapenos stuffed with cheese
- 320 calories
- More filling and indulgent
- Perfect as an appetizer
- Price: ₹380
- High protein content

**Main Differences:**
Fries are crispier and lighter, while poppers are cheesier and more substantial. 
Choose fries if you want something to munch on, or poppers if you want a proper 
appetizer that's more filling.

Both are spicy, but poppers have creamy cheese to balance the heat! 🧀🍟
```

---

## Example 5: Simple Greeting

### User Query
```
"Hi"
```

### Before (might be too welcoming) ❌
```
Hey there, food lover! 👋 Welcome to NutriMood! I'm so excited to help you 
find the perfect meal today. I'm here to suggest amazing food based on your 
mood, dietary preferences, and cravings. Whether you're looking for something 
healthy, indulgent, spicy, or sweet, I've got you covered! Just let me know 
what you're in the mood for and I'll recommend something delicious from our 
menu. What sounds good to you today? 🍽️
```

### After (32 words) ✅
```
Hey! 👋 I'm NutriMood, your food buddy! Tell me what you're craving - 
spicy, healthy, sweet, or indulgent? I'll find the perfect match! 🍽️
```

---

## Example 6: Calorie Question

### User Query
```
"how many calories are these?"
```

### Before ❌
```
Great question! Let me break down the calorie information for the items 
I recommended earlier. The Peri Peri Fries contain 280 calories, which is 
moderate for a snack. The Jalapeno Cheese Poppers have 320 calories for 
6 pieces, so that's about 53 calories per popper, which is quite reasonable 
for a cheesy appetizer! Both are within a healthy range if you're watching 
your calorie intake. Would you like to know more nutritional details?
```

### After (42 words) ✅
```
Peri Peri Fries: 280 calories
Jalapeno Cheese Poppers: 320 calories (6 pieces)

Both are moderate for snacks! Need any other nutritional info? 😊
```

---

## Word Count Guidelines

| Response Type | Target Words | Max Words |
|--------------|--------------|-----------|
| Greeting | 25-35 | 50 |
| Simple request | 40-60 | 80 |
| Standard recommendation | 60-80 | 100 |
| Follow-up question | 30-50 | 60 |
| Detailed comparison | 100-150 | 150 |

## Key Principles

1. **Get to the point** - No unnecessary fluff
2. **1-3 recommendations max** - Don't overwhelm
3. **Use line breaks** - Make it scannable
4. **Short sentences** - Easy to read
5. **Emojis sparingly** - 2-3 maximum
6. **No long explanations** - Unless specifically asked

## Testing Your Responses

**Word Count Tool:**
```python
def count_words(text):
    # Remove emojis and extra spaces
    import re
    text = re.sub(r'[^\w\s]', '', text)
    return len(text.split())
```

**Target:** 50-80 words for most requests, maximum 100 words

---

**Remember:** Users prefer **short, actionable responses** over long explanations! 🎯

