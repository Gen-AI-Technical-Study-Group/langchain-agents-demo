# video walk through of this piece of the code: https://share.descript.com/view/cYugyjlM3mG

# great tutorials for creating langchain agents:
# https://www.pinecone.io/learn/series/langchain/langchain-tools/
# https://python.langchain.com/v0.1/docs/use_cases/tool_use/multiple_tools/
# https://brightinventions.pl/blog/introducing-langchain-agents-tutorial-with-example/

# explaination of the following code: https://www.perplexity.ai/search/langchain-documenttation-for-LK9bWhMASlC8Kxlt7Hy1_g


# Import necessary modules
import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain

## using langsmith 
# video for how langsmith is used in this demo code: https://share.descript.com/view/k4b3fyvaESB
# To learn about this: https://youtu.be/tFXm5ijih98
# To check the running result of langsmith, please go to: https://smith.langchain.com/
from langsmith import Client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "example_calculator_and_search_agent"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
# Initialize LangSmith client
client = Client()


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Initialize the language model
llm = OpenAI(temperature=0)# Define the search tool

llm_math = LLMMathChain(llm=llm)
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)

ddg_search = DuckDuckGoSearchResults()
websearch_tool = Tool(
        name="Current Search",
        func=ddg_search.run,
        description="Useful for when you need to answer questions about current events or the current state of the world"
    )

# Define the tools the agent will use
tools = [math_tool, websearch_tool]

# Initialize the agent with memory
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

# Define the input query
input_query = "What is the current weather in San Francisco? and what is the weather of beijing? then calculate beijing's degree minus san francisco's degree"

# Run the agent
response = agent.run(input=input_query)
print(response)