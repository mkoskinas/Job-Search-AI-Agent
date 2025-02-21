import gradio as gr
from gradio.components import ChatMessage
from gradio.themes import Soft
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from tools.vectorizer import VectorizeTool
from tools.scraper import LinkedInJobScraperTool
from tools.retriever import RetrieverTool
from tools.cv_analyzer import CVAnalyzerTool, CVAnalyzerInput
from tools.cv_enhancer import CVEnhancerTool
from utils.linkedin_parser import linkedin_to_str
from PyPDF2 import PdfReader
from typing import Dict, Any
from langsmith import traceable
from dotenv import load_dotenv
import json 
import time 
import logging
import openai
import tempfile
import os 

logger = logging.getLogger(__name__)

# Additional import for text-to-speech
from gtts import gTTS
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Load environment variables
load_dotenv()

class JobSearchAssistant:
    VALID_FOCUS_AREAS = {
        "summary": ["summary", "profile", "overview", "introduction"],
        "tools": ["tools", "technologies", "tech stack", "skills"],
        "experience": ["experience", "work history", "roles"],
        "achievements": ["achievements", "accomplishments", "qualifications"],
        "leadership": ["leadership", "management", "cross-functional"]
    }

    def __init__(self):
        print("Initializing assistant...")
        start_time = time.time()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.llm = ChatOpenAI(
            temperature=0.0,
            model="gpt-4-turbo-preview",
            max_tokens=4000
        )
        print(f"LLM initialization took: {time.time() - start_time:.2f} seconds")

        self.memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(),
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
    
        # Initialize tools
        self.scraper_tool = LinkedInJobScraperTool()
        self.vectorizer_tool = VectorizeTool()
        self.vectorizer_tool._run(
            action_input={"action": "store_jobs"}, 
            csv_path="data/jobs.csv"
        )
        self.retriever_tool = RetrieverTool(vectorstore=self.vectorizer_tool.get_vectorstore())
        self.cv_analyzer_tool = CVAnalyzerTool()
        self.cv_enhancer_tool = CVEnhancerTool(llm=self.llm)
        self.last_analysis = None 
        self.supported_cv_formats = ['.pdf']
        self.supported_job_sources = ['text', 'linkedin']

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful job search and CV analysis assistant. 
                          
        For general chat and greetings, respond naturally WITHOUT using any tools.
                          
        When asked to find NEW jobs:
        1. Use the job_scraper tool to find relevant positions.
        2. Format and display the results to the user.
        3. Use the job_storage tool to store the results.
        4. Confirm storage but don't make it the main focus of your response.

        **IMPORTANT:**
        When asked to search EXISTING jobs and job_scraper results have already been provided (e.g., your previous response contained job listings) and the user makes any follow-up query that refers to those listings (using phrases such as "these roles", "the above jobs", "the ones you listed", etc.), you MUST:
        1. Immediately call the job_retriever tool using the user's entire query as input.
        2. Do NOT ask any clarifying questions or provide any answer before doing so.
        3. Display all relevant results from the retriever tool.
        4. If no matches are found, offer to search for new jobs instead.

        When analyzing a CV:
        1. Use the cv_analyzer tool to evaluate the match

        After analyzing a CV with cv_analyzer, ALWAYS ask the user if they would like to receive improvement suggestions. 
        If they agree, use the cv_enhancer tool to provide targeted recommendations. DO NOT re-run cv_analyzer if the user agrees or says anything implying they want improvements. At that point, call cv_enhancer directly, providing cv_analysis, cv_text, and job_description from the last analysis.
        You can provide general improvements or focus on specific areas like:
        - Professional Summary
        - Tools & Technologies
        - Relevance
        - Keywords
        - Format
        - Achievements

        Ask the user if they want to focus on a specific area or receive general improvements.
        If the user’s request or response includes synonyms for these focus areas, 
        (e.g., “intro paragraph,” “intro summary,” “professional overview” for “summary,” 
        “tech stack,” “technical tools” for “tools,” etc.), 
        interpret that as wanting focus_area='summary' or focus_area='tools', respectively. 
        Similarly for “responsibilities match,” “ATS format,” and so on, 
        map them to the closest known category.
                          
        If the user requests a new version of their CV e.g. "Please update my CV", return an UPDATED VERSION of their CV {cv_text} based on your improvement suggestions.

        Always show the job listings in your response, even after storing them."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = initialize_agent(
            tools=[self.scraper_tool, self.vectorizer_tool, self.retriever_tool, self.cv_analyzer_tool, self.cv_enhancer_tool],
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            prompt=self.prompt,
            return_intermediate_steps=True   
        )

    def _process_job_description(self, message: str) -> str:
        # First clean common prefixes that might trigger wrong behavior
        common_prefixes = [
            "help analyze my cv against this role:",
            "analyze my cv against this role:",
            "help analyze against this role:",
            "analyze against this role:"
        ]
        
        cleaned_message = message.lower()
        for prefix in common_prefixes:
            if cleaned_message.startswith(prefix):
                message = message[len(prefix):].strip()
                break
        
        # Then process based on type
        if "linkedin.com/jobs" in message:
            job_text = linkedin_to_str(message)
            if job_text:
                return job_text
            raise ValueError("Could not parse LinkedIn job posting")
        
        # Clean raw text job descriptions
        if message.strip().lower().startswith('"'):
            message = message.strip('"').strip()
        if message.strip().lower().startswith('about the job'):
            return message
            
        return message

    def _clean_text(self, text: str) -> str:
        import re
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)   
        text = re.sub(r'\s+', ' ', text)            
        return text.strip()

    def _extract_text_from_pdf(self, file) -> str:
        try:
            # Handle filepath string
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return self._clean_text(text)
                    
            # Handle binary data
            elif isinstance(file, bytes):
                file = io.BytesIO(file)
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return self._clean_text(text)
                
            # Handle file-like objects
            else:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return self._clean_text(text)
                
        except Exception as e:
            print(f"\n=== Debug: PDF Extraction Error ===\n{str(e)}")
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def _format_cv_evaluation(self, data: Dict[str, Any]) -> str:
        if data.get("type") == "error":
            return "Error: " + str(data.get('error', 'Unknown error occurred'))
        
        if data.get("type") == "full_evaluation":
            output = []
            improvements = data.get("improvements", {})
            
            for section in improvements.get("sections", []):
                output.append(f"\n# {section['name']}\n")
                
                for subsection in section.get("subsections", []):
                    output.append(f"\n## {subsection['title']}")
                    
                    output.append("\n🔍 Current Analysis:")
                    output.append(str(subsection['current_state']))
                    
                    output.append("\n💡 Improvements Needed:")
                    for fix in subsection['fix']:
                        output.append(f"• {str(fix)}")
                    
                    output.append("\n✨ Complete Revision:")
                    if isinstance(subsection['example'], dict):
                        output.append(str(subsection['example']['full_revision']))
                        output.append("\n🚀 Technical Implementation:")
                        for tech in subsection['example']['technical_focus']:
                            output.append(f"• {str(tech)}")
                    else:
                        output.append(str(subsection['example']))
                    
                    output.append("\n")
            
            return "\n".join(str(item) for item in output)
        
        return str(data)

    @traceable 
    def chat(self, message, history, files):
        logger.info("Starting new chat interaction")
        start_time = time.time()
        try:
            message_lower = message.lower()
            logger.debug(f"Processing message: {message_lower[:100]}...")
            enhancement_keywords = ["improve", "enhance", "suggest", "suggestions", "recommend", "update", "yes"]
            new_analysis_keywords = ["score", "analyze", "analyse", "evaluate", "review", "compare", "check"]

            # Convert history to format expected by agent if it exists
            agent_history = []
            if history:
                for msg in history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        if msg["role"] == "user":
                            agent_history.append({"type": "human", "data": {"content": msg["content"]}})
                        else:
                            agent_history.append({"type": "ai", "data": {"content": msg["content"]}})

            # Check if job listings have been stored (using the marker)
            logger.debug("Starting marker check in chat history")
            marker_found = any(
                "[JOB_LISTINGS_STORED]" in msg.content
                for msg in self.memory.chat_memory.messages[-5:]
                if isinstance(msg, (AIMessage, SystemMessage))
            )
            logger.debug(f"Job listings marker found: {marker_found}")

            # If marker exists and the query appears to refer to the stored job listings
            if marker_found:
                # CV/Analysis requests should go through normal flow
                if (files or  # CV upload
                    any(kw in message_lower for kw in enhancement_keywords) or   
                    any(kw in message_lower for kw in new_analysis_keywords) or  
                    "linkedin.com/jobs" in message):   
                    logger.debug("CV Analysis/Enhancement Request detected")
                    # Let it fall through to the normal agent flow
                    pass
                else:
                    # This is a follow-up question about jobs
                    logger.info("Processing job-related follow-up question")
                    response = self.agent.invoke({
                        "input": f"The user is asking about the jobs that were just listed. Use the job_retriever tool to answer: {message}",
                        "chat_history": agent_history
                    })
                    return response["output"]

            # File upload branch
            elif files:
                logger.info("Processing file upload")                
                if isinstance(files, str):  
                    file = files
                elif isinstance(files, list) and files:
                    file = files[0]
                elif isinstance(files, bytes):
                    from io import BytesIO
                    file = BytesIO(files)
                else:
                    logger.error(f"Invalid file format: {type(files)}")
                    return "Error: Please upload a valid PDF file."
                    
                try:
                    cv_text = self._extract_text_from_pdf(file)
                    logger.debug(f"Extracted CV text (length: {len(cv_text)})")
                    
                    self.last_analysis = {
                        "cv_text": cv_text,
                        "job_description": None,
                        "analysis": None
                    }
                    
                    job_description = self._process_job_description(message.strip())
                    if job_description and len(job_description) >= 50:
                        logger.debug("Calling analysis agent")
                        analysis = self.agent.invoke({
                            "input": f"""Use the cv_analyzer tool to analyze this CV against the job description.
                            - cv_text: {cv_text}
                            - job_description: {job_description}

                            After getting the analysis results, display them and ask if the user would like improvement suggestions.""",
                            "chat_history": agent_history
                        })
                        
                        logger.info("CV analysis completed successfully")
                        self.last_analysis.update({
                            "job_description": job_description,
                            "analysis": analysis["output"]
                        })
                        return analysis["output"]
                    else:
                        return "CV uploaded successfully. Please provide a job description to analyze against."
                except Exception as e:
                    logger.error(f"CV processing error: {str(e)}", exc_info=True)
                    return f"Error processing CV: {str(e)}"

            # Job description analysis branch
            elif message and len(self._process_job_description(message.strip())) >= 200 and not self.last_analysis.get("analysis"):
                job_description = self._process_job_description(message.strip())
                logger.debug("Valid job description received for initial analysis")
                
                if not self.last_analysis or not self.last_analysis.get("cv_text"):
                    return "Please upload your CV first before providing a job description."

                analysis = self.agent.invoke({
                    "input": f"""Use the cv_analyzer tool to analyze this CV against the job description.
                    - cv_text: {self.last_analysis["cv_text"]}
                    - job_description: {job_description}

                    After getting the analysis results, display them and ask if the user would like improvement suggestions.""",
                    "chat_history": agent_history
                })
                
                self.last_analysis.update({
                    "job_description": job_description,
                    "analysis": analysis["output"]
                })
                return analysis["output"]

            # CV Enhancement branch
            elif self.last_analysis and self.last_analysis.get("analysis") and any(kw in message_lower for kw in enhancement_keywords):
                logger.info("Processing CV enhancement request")
                focus_area = "general"
                for area, keywords in self.VALID_FOCUS_AREAS.items():
                    if any(keyword in message_lower for keyword in keywords):
                        focus_area = area
                        break
                logger.debug(f"Detected focus area: {focus_area}")
                
                try:
                    try:
                        logger.debug("Parsing analysis scores")
                        score_text = self.last_analysis["analysis"]
                        overall_score = None
                        if "Match Score:" in score_text:
                            try:
                                overall_score = float(score_text.split("Match Score:")[1].split("%")[0].strip())
                            except Exception as e:
                                logger.error(f"Error parsing score: {str(e)}")
                        if overall_score is None:
                            overall_score = 75.0
                        categories = self.last_analysis.get("llm_analysis", {}).get("category_scores", {})
                        
                        analysis_dict = {
                            "match_score": overall_score,
                            "analysis_text": self.last_analysis["analysis"],
                            "structured_data": {
                                "overall_score": overall_score,
                                "categories": {
                                    "relevance": categories.get("relevance", {}).get("score", 75),
                                    "keywords": categories.get("keyword_optimization", {}).get("score", 75),
                                    "formatting": categories.get("formatting", {}).get("score", 75),
                                    "achievements": categories.get("achievements", {}).get("score", 75),
                                    "brevity": categories.get("brevity", {}).get("score", 75)
                                }
                            }
                        }
                    except Exception as parse_error:
                        logger.error(f"Score parsing error: {str(parse_error)}")
                        analysis_dict = {
                            "match_score": 75.0,
                            "analysis_text": self.last_analysis["analysis"],
                            "structured_data": {
                                "overall_score": overall_score,
                                "categories": {
                                    "relevance": 75,
                                    "keywords": 75,
                                    "formatting": 75,
                                    "achievements": 75,
                                    "brevity": 75
                                }
                            }
                        }

                    keywords_score = analysis_dict["structured_data"]["categories"]["keywords"]
                    logger.debug(f"Keywords score: {keywords_score}")

                    # Build weak category list
                    threshold = 85
                    weak_categories = []
                    if keywords_score < 80:
                        weak_categories.append("keyword optimization (semantic and keyword matching)")
                    for cat, sc in analysis_dict["structured_data"]["categories"].items():
                        if cat != "keywords" and sc < threshold and cat not in weak_categories:
                            weak_categories.append(cat)
                    weak_category_str = ', '.join(weak_categories) if weak_categories else 'none'
                    logger.debug(f"Identified weak categories: {weak_category_str}")

                    tool_input = {
                        "cv_analysis": analysis_dict,
                        "cv_text": self.last_analysis["cv_text"],
                        "job_description": self.last_analysis["job_description"],
                        "focus_area": focus_area
                    }

                    logger.debug("Calling CV enhancer tool")
                    enhancement_output = self.cv_enhancer_tool._run(**tool_input)
                    logger.info("CV enhancement completed successfully")
                    return self._format_cv_evaluation(enhancement_output)
                except Exception as e:
                    logger.error(f"Enhancement error: {str(e)}", exc_info=True)
                    return f"Error generating improvements: {str(e)}"
                
            # New analysis branch
            elif (self.last_analysis and self.last_analysis.get("analysis") and 
                (any(kw in message_lower for kw in new_analysis_keywords) or 
                len(self._process_job_description(message.strip())) >= 200)):
                logger.info("Processing new analysis request")
                try:
                    # Get CV text - either from new upload or existing
                    cv_text = None
                    if files:
                        try:
                            cv_text = self._extract_text_from_pdf(files)
                            logger.debug(f"Extracted new CV text (length: {len(cv_text)})")
                        except Exception as e:
                            logger.error(f"CV processing error: {str(e)}")
                            return "Sorry, I had trouble processing your CV file. Please make sure it's a valid PDF and try again."
                    
                    if not cv_text and self.last_analysis and "cv_text" in self.last_analysis:
                        cv_text = self.last_analysis["cv_text"]
                    
                    if not cv_text:
                        logger.warning("No CV text found")
                        return "I couldn't find a CV to analyze. Please upload a CV file."

                    job_description = self._process_job_description(message.strip())
                    if not job_description or len(job_description) < 50:
                        logger.warning("Invalid or missing job description")
                        return "Please provide a detailed job description to analyze against."

                    logger.debug(f"Processing analysis (CV length: {len(cv_text)}, Job desc length: {len(job_description)})")
                    analysis = self.agent.invoke({
                        "input": f"""Use the cv_analyzer tool to analyze this CV against the job description.
                        - cv_text: {cv_text}
                        - job_description: {job_description}

                        After getting the analysis results, display them and ask if the user would like improvement suggestions.""",
                        "chat_history": agent_history
                    })

                    self.last_analysis = {
                        "cv_text": cv_text,
                        "job_description": job_description,
                        "analysis": analysis["output"]
                    }
                    
                    return analysis["output"]

                except Exception as e:
                    logger.error(f"Analysis error: {str(e)}", exc_info=True)
                    return "I apologize, but I encountered an error analyzing your CV. Please try again."
            # Regular chat branch
            else:
                logger.info("Processing regular chat request")
                try:
                    # Add special handling for CV improvement requests without uploads
                    if any(phrase in message_lower for phrase in ["improve my cv", "review my cv", "check my cv"]) and not files:
                        return "To analyze or improve your CV, please upload your CV and provide a job description to analyze against."
        
                    response = self.agent.invoke(
                        {"input": message},
                        return_intermediate_steps=True
                    )
                    logger.info(f"Chat response completed in {time.time() - start_time:.2f} seconds")
                    output = response["output"]
                    intermediate_steps = response.get("intermediate_steps", [])
                    
                    # Process intermediate steps
                    for step in intermediate_steps:
                        tool = step[0]
                        tool_output = step[1]
                        
                        if tool.tool == "job_scraper" and tool_output:
                            logger.debug("Processing job scraper output")
                            
                            vectorizer_result = self.vectorizer_tool._run({
                                "action": "store_jobs",
                                "jobs": tool_output
                            })
                            logger.debug(f"Vectorizer result: {vectorizer_result}")
                            
                            self.retriever_tool.vectorstore = self.vectorizer_tool.get_vectorstore()
                            logger.debug(f"Updated retriever vectorstore (total docs: {self.vectorizer_tool.get_total_docs()})")
                            
                            marker_exists = any("[JOB_LISTINGS_STORED]" in msg.content
                                            for msg in self.memory.chat_memory.messages 
                                            if isinstance(msg, (AIMessage, SystemMessage)))
                            
                            if not marker_exists:
                                marker_message = SystemMessage(content="[JOB_LISTINGS_STORED]")
                                self.memory.chat_memory.add_message(marker_message)
                                logger.debug("Added job listings stored marker")
                        
                    return output
                
                except Exception as e:
                    logger.error(f"Chat error: {str(e)}", exc_info=True)
                    return f"Error processing chat: {str(e)}"
        except Exception as e:
            logger.error(f"Critical error in chat method: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
        
# Initialise the assistant
assistant = JobSearchAssistant()

# ----- Helper functions for the custom Gradio Blocks interface -----
@traceable
def chat_interaction(message, file, history, assistant_instance=assistant):
    try:
        if history is None:
            history = []
            
        current_history = []
        for msg in history:
            if isinstance(msg, dict):
                current_history.append(ChatMessage(content=msg["content"], role=msg["role"]))
            else:
                current_history.append(msg)
                
        # Append the user's message and yield immediately
        user_message = ChatMessage(content=message or "", role="user")  # Ensure message is not None
        current_history.append(user_message)
        yield current_history, current_history, "", None, ""
        
        # Append a placeholder for the assistant's response with an animated spinner
        spinner_html = (
            "<div class='typing-indicator'>"
            "  <span></span><span></span><span></span>"
            "</div>"
        )
        placeholder = ChatMessage(content=spinner_html, role="assistant")
        current_history.append(placeholder)
        yield current_history, current_history, "", None, ""
        
        # Compute the actual response
        response = assistant_instance.chat(message, current_history, file)
        if response is None:
            response = "I apologize, but I encountered an error processing your request. Please try again."
            
        # Replace the placeholder with the actual response
        current_history[-1] = ChatMessage(content=response, role="assistant")
        yield current_history, current_history, "", None, ""

    except Exception as e:
        print(f"Error in chat interaction: {str(e)}")
        error_msg = f"Error: {str(e)}"
        if not current_history:
            current_history = []
        error_history = current_history + [ChatMessage(content=error_msg, role="assistant")]
        yield error_history, error_history, "", None, ""


def read_aloud(history):
    if history and len(history) > 0:
        last_response = next((msg.content for msg in reversed(history)
                              if msg.role == "assistant"), None)
        print(f"Found last response: {last_response[:50]}...")  # Debug print
        
        if last_response:
            try:
                # Truncate or summarize the text if it's too long
                max_chars = 4000  # Leave some buffer from the 4096 limit
                if len(last_response) > max_chars:
                    truncated_text = "Here's a summary of the key points: " + last_response[:max_chars] + "..."
                else:
                    truncated_text = last_response

                client = openai.OpenAI()
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=truncated_text
                )
                
                # Create a temporary file and keep it open
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.write(response.content)
                temp_file.flush()  # Ensure all data is written
                temp_path = temp_file.name
                temp_file.close()
                
                print(f"Audio saved to: {temp_path}")  # Debug print
                print(f"File exists: {os.path.exists(temp_path)}")  # Debug print
                print(f"File size: {os.path.getsize(temp_path)} bytes")  # Debug print
                
                return temp_path
                
            except Exception as e:
                print(f"Error in OpenAI TTS conversion: {str(e)}")
                return None
    return None

def file_uploaded(file_path):
    if file_path is not None:
        try:
            filename = os.path.basename(file_path)
            return f"📄 {filename}"
        except Exception as e:
            print(f"Error handling file upload: {str(e)}")
            return "Error processing uploaded file"
    return "No file uploaded"

# ----- Build custom UI with Gradio Blocks -----
css_path = os.path.join(os.path.dirname(__file__), "css", "style.css")
with open(css_path, 'r') as f:
        css_content = f.read()


with gr.Blocks(css=css_content, theme=Soft()) as demo:
    gr.Markdown("# AI Job Search Assistant")
    
    # Main chat area using our custom callback and state
    chatbot = gr.Chatbot(label="Conversation", elem_classes=["chatbot"], type="messages", render_markdown=True)
    state = gr.State([])

    with gr.Row():        
        with gr.Column():
            # Put textbox & send button in the same row with a custom class
            with gr.Row(elem_classes=["message-row"]):
                txt = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False,
                    lines=1,  
                    elem_classes=["fixed-textbox"],
                    scale=18,
                    interactive=True,
                )
                
                send_btn = gr.Button(
                    "Send",
                    elem_classes=["send-button", "bigger-button"],
                    scale=1
                )

            gr.Examples(
                examples=[
                    "Tell me about Software Engineering roles in Berlin",
                    "Help me improve my CV"
                ],
                inputs=txt,  
                label="Try these examples:"
            )

            with gr.Row():
                file_upload = gr.UploadButton("📎", file_types=[".pdf", ".doc", ".docx", ".txt"], type="filepath", elem_classes=["icon-button", "tooltip-icon-button"]) 
                tts_btn = gr.Button("🔊", elem_classes=["icon-button", "tooltip-icon2-button"])
                # New component to show file upload status 
                file_status = gr.Markdown("")
    
    # Hidden audio output for TTS
    audio_output = gr.Audio(
        label="Read Aloud Output", 
        interactive=False,
        type="filepath",
        visible=False,
        autoplay=True
    )
    # Chat callbacks
    txt.submit(
        fn=chat_interaction, 
        inputs=[txt, file_upload, state], 
        outputs=[chatbot, state, txt, file_upload, file_status],
        show_progress="full"
    )
    send_btn.click(
        fn=chat_interaction, 
        inputs=[txt, file_upload, state], 
        outputs=[chatbot, state, txt, file_upload, file_status],
        show_progress="full"
    )
    tts_btn.click(
        fn=read_aloud, 
        inputs=[state], 
        outputs=[audio_output]
    )

    # File upload status
    file_upload.upload(fn=file_uploaded, inputs=file_upload, outputs=file_status)

if __name__ == "__main__":  
    demo.launch(share=True)
