from langchain.chat_models import ChatOpenAI
from typing import List, Optional

class AgentManager:
    def __init__(self, model=None, memory=None, prompt=None, tools=None, vectorstores=None):
        """Create an Agent Manager with its tools and vectorstores to perf

        Args:
            model (ChatOpenAI, optional): Chat model from langchain.chat_models. Defaults to None.
            memory (langchain.memory, optional): _description_. Defaults to None.
            prompt (ChatPromptTemplate, optional): _description_. Defaults to None.
            tools (List[tool], optional): List of agent tools. Defaults to None.
            vectorstores (List[langchain.vectorstores], optional): List of knowledge base vectorstores. Defaults to None.
        """
        # Init Agent Manager
        self.vectorstores = vectorstores
        self.functions = [format_tool_to_openai_function(tool) for tool in tools] if tools else None  
        if tools:
            self.model = model if model else ChatOpenAI(temperature=0).bind(self.functions)
        else:
            self.model = model if model else ChatOpenAI(temperature=0)
        self.memory = memory if memory else ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.prompt = prompt if prompt else ChatPromptTemplate.from_messages([
            ("system", "You are helpful assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create Runnable Agent Chain
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad = lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        
        # Create Agent Executor
        self.agent_executor = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)
        
    def __call__(self, query: str) -> str:
        """Call the agent on a user query

        Args:
            query (str): User input query to pass to the agent

        Returns:
            str: Agent answer to the user query
        """
        if not query:
            return
        result = self.agent_executor.invoke({"input": query})
        return result["output"]
    