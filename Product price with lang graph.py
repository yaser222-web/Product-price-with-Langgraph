import os
import difflib
from dotenv import load_dotenv

# ==========================================
# 1. Setup the network and local variables
# ==========================================

os.environ["HTTP_PROXY"]  = "http://127.0.0.1:8010"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8010"
os.environ["NO_PROXY"]    = "localhost,127.0.0.1"
load_dotenv()

# ==========================================
# 2. Imports LangChain و LangGraph
# ==========================================

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = "qwen3:1.7b"

# ==========================================
# 3. Tools
# ==========================================
@tool
def get_product_price(product: str) -> float | str:
    """Look up the price of the product in catalog. The input 'product' must be a simple string 
    of the exact product name, like 'laptop' or 'keyboard'"""
    print(f"   >> Executing get_product_price on product: {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.6}
    valid_products = list(prices.keys())
    matches = difflib.get_close_matches(product.lower(), valid_products, n=1, cutoff=0.6)
    
    if matches:
        matched_product = matches[0] 
        print(f"   >> Smart match: changed '{product}' to '{matched_product}'")
        return prices.get(matched_product)
    else:
        return "Error: product not found in catalog"

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: silver, bronze, gold."""
    print(f"   >> Executing apply_discount for price: {price}, with discount: {discount_tier}")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
# ==========================================
# ۴. آماده‌سازی مدل و قیچی حافظه
# ==========================================
tools = [get_product_price, apply_discount]
llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# دروازه‌بانِ هوشمند: نگهداری ۶ پیام آخر، حفظ SystemMessage، و شروع همیشه با پیام کاربر
trimmer = trim_messages(
    max_tokens=6,           
    strategy="last",        
    token_counter=len,      
    include_system=True,    
    allow_partial=False,    
    start_on="human"        
)

# ==========================================
# ۵. معماری LangGraph (گرافِ ایجنت)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState):
    """ایستگاه مغز متفکر: ابتدا حافظه را قیچی می‌کند، سپس به مدل می‌دهد"""
    messages = state["messages"]
    
    # اعمالِ قیچی قبل از ارسال به LLM
    trimmed_messages = trimmer.invoke(messages)
    
    response = llm_with_tools.invoke(trimmed_messages)
    return {"messages": [response]}

# ایستگاه اجرای ابزارها
tool_node = ToolNode(tools)

# رسم نقشه گراف
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# کامپایل نهایی برنامه
app = workflow.compile()

# ==========================================
# ۶. اجرای حلقه مکالمه با کاربر
# ==========================================
if __name__ == "__main__":
    print("Hello Professional LangGraph Agent (with Memory Trimmer)!\n")
    
    system_msg = SystemMessage(content=
        "You are a helpful shopping assistant. "
        "You have access to a product catalog tool and a discount tool.\n\n"
        "STRICT RULES — you must follow these exactly:\n"
        "1. NEVER guess or assume any product price. You MUST call get_product_price first.\n"
        "2. Only call apply_discount AFTER you have received a price from get_product_price.\n"
        "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n"
        "4. If the user does not specify a discount tier, ask them which tier to use.\n"
        "5. If a tool indicates that a product is not found, you must NOT " 
        "attempt to guess similar words or call the tool again. " 
        "Immediately stop using tools and inform the user."
    )
    
    # تنظیم حافظه اولیه
    current_state = {"messages": [system_msg]}
    
    while True:
        human_message = input("\nPlease enter your request (Goodbye to exit):\n")
        if human_message.lower() == "goodbye":
            break

        # اضافه کردن پیام جدید کاربر به چمدان حافظه
        current_state["messages"].append(HumanMessage(content=human_message))
        
        # اجرای گراف
        final_state = app.invoke(current_state)
        
        # استخراج و چاپ جواب نهایی
        result = final_state["messages"][-1].content
        print(f"\nThe final answer is: '{result}'")
        
        # بروزرسانی حافظه برای سوال بعدی
        current_state = final_state