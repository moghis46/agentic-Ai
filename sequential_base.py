import os
from typing import TypedDict

#lets creat the state first
class pipelinestate(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_output:str

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
llm=ChatGroq(model="openai/gpt-oss-120b", temperature=1)


def editor_node(state: pipelinestate) -> dict:
    """Stage 1: Cleans up grammar, removes typos, and refines the tone."""

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transitions "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response=llm.invoke(prompt)
    return{"edited_text":response.content.strip()}

def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""
    print("\n--- [Stage 2] Executing Scriptwriter Node ---")

    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. Make it sound "
        "like a real person speaking passionately. Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following scr[...] "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it senten[...] "
        "or repeat information. Alternating comfortably between Hindi and English phrases [...] "
        "an intellectual tech educator would speak naturally on a live stream. Keep the en[...] "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}

#now your states and nodes are ready now its time create graph
#and for creating the graph you have to connect these nodes for that 
#you have to use the edges
#and edges are very important for the workflows 

from langgraph.graph import StateGraph,START,END
#create the graph 

graph=StateGraph(pipelinestate)

#add nodes in the graph 
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

#add edges(sequential-one after other)
graph.add_edge(START,"editor")    
graph.add_edge('editor',"scriptwriter")
graph.add_edge('scriptwriter',"translator")    
graph.add_edge('translator',END)

#compile the graph
app=graph.compile()

result=app.invoke({
    "raw_input":"AI agents are the future of the tech.They can think,plan and act on their own .And langraph helps you to buid the agents  "
})

#output

print(result['final_output'])

    

