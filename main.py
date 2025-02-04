import gradio as gr
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from tools.vectorizer import VectorizeTool
from tools.scraper import LinkedInJobScraperTool
from tools.retriever import RetrieverTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class JobSearchAssistant:

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.0)
        self.memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(),
            memory_key="chat_history",
            return_messages=True
        )
    
    # Initialize tools
        self.scraper_tool = LinkedInJobScraperTool()
        self.vectorizer_tool = VectorizeTool()
        self.vectorizer_tool._run(
            action_input={"action": "store_jobs"}, 
            csv_path="data/jobs.csv"
        )
        self.retriever_tool = RetrieverTool(vectorstore=self.vectorizer_tool.get_vectorstore())

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful job search assistant. 
        
        When asked to find NEW jobs:
        1. Use the job_scraper tool to find relevant positions
        2. Format and display the results to the user
        3. Use the job_storage tool to store the results
        4. Confirm storage but don't make it the main focus of your response
                          
        When asked to search EXISTING jobs:
        1. Use the job_retriever tool with the user's exact query. When calling the retriever tool, pass the entire user question as query. Do NOT shorten it to a single keyword.
        2. Display all relevant results found
        3. If no matches are found, offer to search for new jobs instead

        Always show the job listings in your response, even after storing them."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = initialize_agent(
            tools=[self.scraper_tool, self.vectorizer_tool, self.retriever_tool],
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            prompt=self.prompt,
            return_intermediate_steps=True   
        )

    def chat(self, message, history, files):
        try:
            # Handle resume file upload
            if files:
                file_content = files[0].read().decode('utf-8')
                message = f"Review this resume:\n{file_content}\n{message}"
            
            # Debug chat history - simplified
            print("\n=== Debug: Chat History ===")
            print(f"Current message: {message}")
            print(f"History type: {type(history)}")
            if history:
                print(f"History content: {history}")

            # Get response with intermediate steps
            response = self.agent.invoke(
                {"input": message},
                return_intermediate_steps=True
            )

            # Extract output and steps
            output = response["output"]
            intermediate_steps = response.get("intermediate_steps", [])
            
            # Process intermediate steps
            for step in intermediate_steps:
                tool = step[0]  # The AgentAction
                tool_output = step[1]  # The tool output
                
                if tool.tool == "job_scraper" and tool_output:
                    print("\n=== Debug: Captured job_scraper output ===")
                    print(f"Tool output: {tool_output}")

                    # Store jobs directly from scraper output
                    vectorizer_result = self.vectorizer_tool._run({
                        "action": "store_jobs",
                        "jobs": tool_output
                    })
                    
                    self.retriever_tool.vectorstore = self.vectorizer_tool.get_vectorstore()
                    print(f"\n=== Vectorizer stored the jobs ===\n{vectorizer_result}")

            return output

        except Exception as e:
            print(f"\n=== Debug: Error ===\n{str(e)}")
            return f"Error processing your request: {str(e)}"

# Initialise the assistant
assistant = JobSearchAssistant()

# Create the interface with file upload
demo = gr.ChatInterface(
    fn=assistant.chat,
    title="AI Job Search Assistant",
    description="Chat with your AI assistant and upload your resume when needed.",
    examples=[
        ["Tell me about Software Engineering roles in Paris", None], 
        ["Can you review my resume?", None] 
    ],
    additional_inputs=[
        gr.File(
            label="Upload Resume",
            file_types=[".pdf", ".doc", ".docx", ".txt"],
            type="binary"
        )
    ],
    type="messages" 
)

if __name__ == "__main__":
    demo.launch(share=True)