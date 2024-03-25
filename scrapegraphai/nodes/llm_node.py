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

        INSTRUCTIONS: extract the list of all the titles and descriptions of the articles.

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

        HTML CODE:    <body class="fixed-top-nav " style="padding-top: 57px;">
    <header>
        <nav id="navbar" class="navbar navbar-light navbar-expand-sm fixed-top">
            <div class="container">
                <a class="navbar-brand title font-weight-lighter" href="/"><span class="font-weight-bold">Marco&nbsp;</span>Perini</a> <button class="navbar-toggler collapsed ml-auto" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation"> <span class="sr-only">Toggle navigation</span> <span class="icon-bar top-bar"></span> <span class="icon-bar middle-bar"></span> <span class="icon-bar bottom-bar"></span> </button> 
                <div class="collapse navbar-collapse text-right" id="navbarNav">
                <ul class="navbar-nav ml-auto flex-nowrap">
                    <li class="nav-item "> <a class="nav-link" href="/">About</a> </li>
                    <li class="nav-item dropdown active">
                        <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Projects<span class="sr-only">(current)</span></a> 
                        <div class="dropdown-menu dropdown-menu-right" aria-labelledby="navbarDropdown">
                            <a class="dropdown-item" href="/projects/">Projects</a> 
                            <div class="dropdown-divider"></div>
                            <a class="dropdown-item" href="/competitions/">Competitions</a> 
                        </div>
                    </li>
                    <li class="nav-item "> <a class="nav-link" href="/cv/">CV</a> </li>
                    <li class="toggle-container"> <button id="light-toggle" title="Change theme"> <i class="fa-solid fa-moon"></i> <i class="fa-solid fa-sun"></i> </button> </li>
                </ul>
                </div>
            </div>
        </nav>
        <progress id="progress" value="0" max="284" style="top: 57px;">
            <div class="progress-container"> <span class="progress-bar"></span> </div>
        </progress>
    </header>
    <div class="container mt-5">
        <div class="post">
            <header class="post-header">
                <h1 class="post-title">Projects</h1>
                <p class="post-description"></p>
            </header>
            <article>
                <div class="projects">
                <div class="grid" style="position: relative; height: 861.992px;">
                    <div class="grid-sizer"></div>
                    <div class="grid-item" style="position: absolute; left: 0px; top: 0px;">
                        <a href="/projects/rotary-pendulum-rl/">
                            <div class="card hoverable">
                            <figure>
                                <picture>    <img src="/assets/img/rotary_pybullet.jpg" width="auto" height="auto" alt="project thumbnail" onerror="this.onerror=null; $('.responsive-img-srcset').remove();"> </picture>
                            </figure>
                            <div class="card-body">
                                <h4 class="card-title">Rotary Pendulum RL</h4>
                                <p class="card-text">Open Source project aimed at controlling a real life rotary pendulum using RL algorithms</p>
                                <div class="row ml-1 mr-1 p-0"> </div>
                            </div>
                            </div>
                        </a>
                    </div>
                    <div class="grid-sizer"></div>
                    <div class="grid-item" style="position: absolute; left: 260px; top: 0px;">
                        <a href="https://github.com/PeriniM/DQN-SwingUp" rel="external nofollow noopener" target="_blank">
                            <div class="card hoverable">
                            <figure>
                                <picture>    <img src="/assets/img/value-policy-heatmaps.jpg" width="auto" height="auto" alt="project thumbnail" onerror="this.onerror=null; $('.responsive-img-srcset').remove();"> </picture>
                            </figure>
                                <div class="card-body">
                                    <h4 class="card-title">DQN Implementation from scratch</h4>
                                    <p class="card-text">Developed a Deep Q-Network algorithm to train a simple and double pendulum</p>
                                    <div class="row ml-1 mr-1 p-0"> </div>
                                </div>
                                </div>
                            </a>
                        </div>
                        <div class="grid-sizer"></div>
                        <div class="grid-item" style="position: absolute; left: 0px; top: 447.414px;">
                            <a href="https://github.com/PeriniM/Multi-Agents-HAED" rel="external nofollow noopener" target="_blank">
                                <div class="card hoverable">
                                <figure>
                                    <picture>    <img src="/assets/img/multi_agents_haed.gif" width="auto" height="auto" alt="project thumbnail" onerror="this.onerror=null; $('.responsive-img-srcset').remove();"> </picture>
                                </figure>
                                <div class="card-body">
                                    <h4 class="card-title">Multi Agents HAED</h4>
                                    <p class="card-text">University project which focuses on simulating a multi-agent system to perform environment mapping. Agents, equipped with sensors, explore and record their surroundings, considering uncertainties in their readings.</p>
                                    <div class="row ml-1 mr-1 p-0"> </div>
                                </div>
                                </div>
                            </a>
                        </div>
                        <div class="grid-sizer"></div>
                        <div class="grid-item" style="position: absolute; left: 260px; top: 370.172px;">
                            <a href="/projects/wireless-esc-drone/">
                                <div class="card hoverable">
                                <figure>
                                    <picture>    <img src="/assets/img/wireless_esc.gif" width="auto" height="auto" alt="project thumbnail" onerror="this.onerror=null; $('.responsive-img-srcset').remove();"> </picture>
                                </figure>
                                <div class="card-body">
                                    <h4 class="card-title">Wireless ESC for Modular Drones</h4>
                                    <p class="card-text">Modular drone architecture proposal and proof of concept. The project received maximum grade.</p>
                                    <div class="row ml-1 mr-1 p-0"> </div>
                                </div>
                                </div>
                            </a>
                        </div>
                    </div>
                    </div>
                </article>
            </div>
        </div>
        <footer class="fixed-bottom">
            <div class="container mt-0"> © Copyright 2023 Marco Perini. Powered by <a href="https://jekyllrb.com/" target="_blank" rel="external nofollow noopener">Jekyll</a> with <a href="https://github.com/alshedivat/al-folio" rel="external nofollow noopener" target="_blank">al-folio</a> theme. Hosted by <a href="https://pages.github.com/" target="_blank" rel="external nofollow noopener">GitHub Pages</a>. </div>
        </footer> 
        <div class="hiddendiv common"></div>
        </body>
        NOTE: you must provide the output of ONLY the list of dictionay containing the article name and description.

        Take a deep breath, think step by step, and provide the output.
        """)

        print("---------------")
        print(str(res))
        return res
