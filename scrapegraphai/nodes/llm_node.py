# Imports from standard library
from typing import List
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Imports from the library
from .base_node import BaseNode


class LLM_answer_node(BaseNode):

    def __init__(self, input: str, output: List[str], model_config: dict,
                 node_name: str = "generate_anser_llm"):
        """
        Initializes the GenerateAnswerNode with a language model client and a node name.
        Args:
            llm (Ollama): An instance of the Ollama class.
            node_name (str): name of the node
        """
        super().__init__(node_name, "node", input, output, 2, model_config)
        self.llm_model = model_config["llm_model"]

    def execute(self, state):
        """
        Generates an answer by constructing a prompt from the user's input and the scraped
        content, querying the language model, and parsing its response.

        The method updates the state with the generated answer under the 'answer' key.

        Args:
            state (dict): The current state of the graph, expected to contain 'user_input',
                          and optionally 'parsed_document' or 'relevant_chunks' within 'keys'.

        Returns:
            dict: The updated state with the 'answer' key containing the generated answer.

        Raises:
            KeyError: If 'user_input' or 'document' is not found in the state, indicating
                      that the necessary information for generating an answer is missing.
        """

        print(f"--- Executing {self.node_name} Node ---")

        # Interpret input keys based on the provided input expression
        input_keys = self.get_input_keys(state)

        # Fetching data from the state based on the input keys
        input_data = [state[key] for key in input_keys]

        user_prompt = input_data[0]
        doc = input_data[1]

        llm = Ollama(
            model="llama2",  temperature=0,
            format="json", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Generate response using the language model
        res = llm("""
        You are a web scraper that wants to extract data from the HTML code.

        INSTRUCTIONS: {user_prompt}.

        OUTPUT: list of dictionaries containing as key the name of the article and as value the article. Provide just the output with any other words
        
        OUTPUT FORMAT:   [
        {
        "article name":" description"
        },
        {
        "article name": "description"
        }
        ]

        EXAMPLE:
        {
        "The Keys to a Long Life Are Sleep and a Better Diet—and Money":"Nobel Prize–winning biologist Venki Ramakrishnan explores the science and charlatans of life-extension."
        },
        {
        "Fisker Suspends Its EV Production": "After the difficult launch of its Ocean SUV, Fisker says it’s pausing production for six weeks."
        }
        ]

        HTML CODE:   
        {doc}
        NOTE: you must provide the output of ONLY the list of dictionay containing the article name and description.

        Take a deep breath, think step by step, and provide the output.
        """)

        print("---------------")
        print(str(res))
        return res
