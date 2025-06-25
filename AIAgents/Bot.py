from typing import TypedDict, List 
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START,END

model = OllamaLLM(model= "llama3.2:1b")

class State(TypedDict):
    messages : List[HumanMessage]

# define the processing node
def process(state: State) -> State:
    """ Process the input and return the llm response"""
    response = model.invoke(state['messages'])
    print("\nAI Response:", response, "\n")
    
    return state


# define the state graph
graph = StateGraph(State)
graph.add_node("process",process)

graph.add_edge(START,"process")
graph.add_edge("process", END)

app = graph.compile()

user_input = input("Enter:  ")
while user_input != "exit":
    # test the app
    app.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter:  ")

