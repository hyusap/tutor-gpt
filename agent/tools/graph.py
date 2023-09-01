from langchain.tools.base import BaseTool
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os

from dotenv import load_dotenv

load_dotenv()
chat_model = ChatOpenAI(model="gpt-4", temperature="0.3")


class GraphTool(BaseTool):
    name = "graph"
    description = "A tool for graphing functions. Explain exactly what function you want to graph and the tool will create the graph for you."

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """You are an function graphing bot. You can create plots of functions with the python library matplotlib.
The user will provide you instructions on what to function to plot and you will make a plot using the matplotlib python libary.
The code should save an image of the plot to the file "media/plot.png".
DO NOT include anything in your response other than code."""
                ),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        output = chat_model.predict_messages(chat_prompt.format_messages(query=query))
        code = output.content

        script_directory = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_directory, "script.py")
        print(script_path)
        with open(script_path, "w") as f:
            f.write(code)
        
        # TODO: Use a docker container with limited resources/perms before using this in production!
        result = os.system(
            f'python {script_path}'
        )
        if result != 0:
            return "Too complicated for me to understand."
        else:
            return "Plot has been created, in your final answer YOU MUST instruct the user to write {plot.png} to insert the plot."

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("GraphTool does not support async")