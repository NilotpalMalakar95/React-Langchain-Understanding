from typing import List, Union
from dotenv import load_dotenv

from langchain import hub
from langchain.tools import Tool
from langchain.agents import tool, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_openai.chat_models import ChatOpenAI

from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from callbacks import AgentCallbacksHandler

# Loading the api-keys
load_dotenv()

# Functions


# `tool` if used as a decorator can convert a function and convert it to a langchain tool
# Tt will take the name of the function, its description, arguments, what it returns etc and would populate the same to the langchain Tool class
# All these information would later be used by the LLM reasoning engine to decide whether to use this tool or not
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"{text=}")
    return len(text)


def get_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool_ in tools:
        if tool_.name == tool_name:
            return tool_
    raise ValueError(f'Tool with the name "{tool_name}" not found !')


if __name__ == "__main__":
    # print("Hello ReAct Langchain...")

    # Creating a custom langchain tools object
    # These tools would be supplied to the ReAct agent
    custom_langchain_tools = [get_text_length]

    # # ReAct template by HChase
    # react_prompt = hub.pull('hwchase17/react')

    # Setting the prompt template
    template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    # Setting the template up
    prompt_template = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(custom_langchain_tools),
        tool_names=", ".join([tool_.name for tool_ in custom_langchain_tools]),
    )

    # Initializing the model
    chat_model_llm = ChatOpenAI(
        temperature=0, stop=["\nObservation"], callbacks=[AgentCallbacksHandler()]
    )
    # stop = ['\nObservation'] ---> tells the model to stop generating output once it produces the \nObservations token
    # If we dont put it the llm would continue to generate text even after it produces the \nObservation token
    # The '\Observation' token is the result of the current tool which is something that should come from the tool (See Observation in the template for the prompt)
    #  If it is getting produced by the model, the model is likely hallucinate

    # In order to keep track of the agent history we will be creating the intermediate_steps variable which will initially be initiated as an empty list
    intermediate_steps = []

    # Defining the agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(
                x["agent_scratchpad"]
            ),  # Initializing the AgentAction --> langchain object to string formatter for storing the intermediate results of the model since the LLM takes only strings as inputs
        }
        | prompt_template
        | chat_model_llm
        | ReActSingleInputOutputParser()
    )  # The input to the prompt is left blank so that the user can provide the prompt while invoking the llm
    # Explanation ---> The LCEL creates a sequential chain of elements separated by the pipe '|' operator
    # The element of the right side takes input as the output of the element to it immediate left and vice versa, just like the sequential models in tensorflow 2

    # This expression is written using the LCEL --> Langchain Expression Language
    # A way of defining declaratively and compose the chains together
    # Makes the code more readable, more composable and helps to understand what is happening under the hood while diving into the internals
    # Other benefits --> Parallel processing, Batch and Streaming support and Fallbacks
    # One main reason behind its implementation was to make things simpler and easy to understand what is going on under the hood
    # With the help of LCEL we can understand what exactly is happening at each step and what is the chronogical order of the executions

    agent_step = ""  # Initialize the agent_step and an empty placeholder
    while not isinstance(
        agent_step, AgentFinish
    ):  # Run the loop till we get a AgentFinish instance i.e the LLM has found the final answer

        # Invoking the agent --> Generating the response
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the text in characters : I am Nilotpal Malakar",
                "agent_scratchpad": intermediate_steps,
            }
        )
        # Here we are prescribing the expected data type of agent_step as either AgentAction or AgentFinish types

        # Parsing the response using Output Parsers --> Uses RegEx in backend
        # print(agent_step)

        if isinstance(agent_step, AgentAction):
            # Fetching the name of the tool
            tool_name = agent_step.tool
            # Fetching the tool to be used from the custom_langchain_tools
            tool_to_use = get_tool_by_name(custom_langchain_tools, tool_name)
            # Fetching the tool_input being passed to the agent_step
            tool_input = agent_step.tool_input

            # Generating the observation by plugging in the tool_input to the tool to be used
            observation = tool_to_use.func(str(tool_input))
            # Appending the reasoning engine history of the LLM and the result that was produced in each iteration
            intermediate_steps.append((agent_step, str(observation)))
            print(f"{observation=}")

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
