from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START,END


model = OllamaLLM(model= "llama3.2:1b")

class State(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]


def process(state: State) -> State:
    """ Process the user input and the llm response"""
    print("Current Messages:", state['messages'])
    reponse = model.invoke(state['messages'])
    print("AI Response:", reponse, "\n")
    state['messages'].append(AIMessage(content=reponse))
    return state

# define the state graph
graph = StateGraph(State)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process", END)
app = graph.compile()


converstaion_history = []

user_input = input("Enter:  ")
while user_input.lower() != "exit":
    converstaion_history.append(HumanMessage(content=user_input))
    print("********************************\n")
    print(len(converstaion_history), "messages in history")
    if len(converstaion_history) >= 4:
        converstaion_history = converstaion_history[2:]
    print("Conversation History:", converstaion_history)
    print("********************************\n")
    res = app.invoke({"messages": converstaion_history})
    user_input = input("Enter:  ")
    


