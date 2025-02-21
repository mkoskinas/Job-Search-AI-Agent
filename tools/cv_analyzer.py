from langchain.tools import BaseTool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  
from langchain_community.vectorstores import Chroma
from pydantic import Field, BaseModel
from typing import Dict, Any, List, Type, ClassVar
import spacy
import json
import logging
import chromadb
from chromadb.config import Settings
from time import time
import cProfile
import pstats
import io
 

class CVAnalyzerInput(BaseModel):
    cv_text: str = Field(description="The text content of the CV/resume to analyse")
    job_description: str = Field(description="The job description to compare against")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cv_text": "Resume content here...",
                    "job_description": "Job description here..."
                }
            ]
        }
    }


class CVAnalyzerTool(BaseTool):
    name: str = "cv_analyzer"
    description: str = """This tool performs detailed CV/resume analysis against job descriptions. Can process BOTH raw job description text AND LinkedIn job URLs.
    
    WHEN TO USE:
    - Only when both a CV and a specific job description are provided
    - When a detailed comparison between a CV and job requirements is needed
    - For resume scoring and skills matching analysis
    
    DO NOT USE:
    - For general chat or greetings
    - When only a CV or only a job description is available
    - For non-CV related queries
    
    Input must be a dictionary containing:
    - 'cv_text': The full text of the CV/resume
    - 'job_description': The complete job posting or requirements
    
    Returns a comprehensive analysis with matching scores."""

    OUTPUT_TEMPLATE: ClassVar[str] = """
    🎯 CV Match Analysis
    -------------------
    🏆 Overall ATS score: {overall_ats_score}%
    This score combines customed keyword matching and semantic relevance as well as llm generated ATS compatibility.
    
    📊 Core Metrics:
    • 🔑 Keyword Match Score: {keyword_match_score}%
    • 🚀 Technical Fit: {technical_fit}%
    • Role Match: {role_match}%
    • Formatting & Presentation: {formatting}%
    
    💪 Strengths:
    {strengths}
    
    🔍 Areas for Focus:
    {areas_for_focus}
    """

    args_schema: Type[BaseModel] = CVAnalyzerInput   

    # Define Pydantic fields
    embeddings: OpenAIEmbeddings = Field(default_factory=lambda: OpenAIEmbeddings(
        chunk_size=1000,  
        request_timeout=30   
    ))
    nlp: Any = Field(default_factory=lambda: spacy.load("en_core_web_sm"))
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0,
        request_timeout=60
    ))
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))

    lemma_cache: Dict[str, str] = Field(default_factory=dict)   
    similarity_cache: Dict[str, bool] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.setLevel(logging.INFO)
        self.lemma_cache = {}
        self.similarity_cache = {}
    
    def extract_keywords(self, text: str, pos: List[str] = ['NOUN', 'PROPN', 'ADJ']) -> List[str]:
        """Extract important keywords using spaCy with specified parts of speech"""
        doc = self.nlp(text)
        keywords = set()

        blacklist = {
        'this', 'that', 'these', 'those', 'such', 'other', 'many', 'some',
        'any', 'all', 'both', 'each', 'few', 'several', 'exciting', 'current',
        'various', 'different', 'specific', 'particular', 'certain'
        }
        
        for token in doc:
            # Only include tokens that:
            # 1. Match desired part of speech
            # 2. Are not stop words
            # 3. Are longer than 2 chars
            # 4. Are not in blacklist
            if (token.pos_ in pos and 
                not token.is_stop and 
                len(token.text) > 2 and
                token.lemma_.lower() not in blacklist):
                keywords.add(token.lemma_.lower())
        
        # For noun phrases, only include technical or business-relevant ones
        for chunk in doc.noun_chunks:
            # Skip if starts with determiners or contains blacklisted words
            if (not chunk.root.is_stop and
                len(chunk.text) > 2 and
                not any(word in chunk.text.lower() for word in blacklist)):
                keywords.add(chunk.text.lower())
        
        return list(keywords)

    def _get_structured_analysis(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Get structured analysis using Resume-Matcher inspired approach"""
            start_time = time()
            cv_text = inputs.get('cv_text', '')
            job_desc = inputs.get('job_description', '')
            
            # Get keywords
            required_keywords = self.extract_keywords(job_desc, pos=['NOUN', 'PROPN'])
            preferred_keywords = self.extract_keywords(job_desc, pos=['ADJ', 'VERB'])
            cv_keywords = self.extract_keywords(cv_text)
            
            # Calculate matches
            required_matches = sum(1 for k in required_keywords if any(
                self._is_similar_keyword(k, cv_k) for cv_k in cv_keywords
            ))
            preferred_matches = sum(1 for k in preferred_keywords if any(
                self._is_similar_keyword(k, cv_k) for cv_k in cv_keywords
            ))
            
            # Calculate weighted keyword score (70% required, 30% preferred)
            if required_keywords:
                required_score = (required_matches / len(required_keywords)) * 70
                preferred_score = (preferred_matches / len(preferred_keywords)) * 30 if preferred_keywords else 0
                keyword_score = required_score + preferred_score
            else:
                keyword_score = 100
                
            # Calculate semantic similarity using Chroma
            try:
                # Initialise settings for in-memory operation
                client = chromadb.EphemeralClient(Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=False
                ))
                
                # Create vectorstore using the ephemeral client
                vectorstore = Chroma.from_texts(
                    texts=[cv_text],
                    embedding=self.embeddings,
                    client=client
                )
                
                # Add CV text and search against job description
                results = vectorstore.similarity_search_with_relevance_scores(
                    job_desc,
                    k=1
                )
                semantic_score = results[0][1] * 100 if results else 0
                
            except Exception as e:
                print(f"Error calculating semantic score: {e}")
                semantic_score = keyword_score  # Fallback to keyword score

            self.logger.info(f"Structured analysis took {time() - start_time:.2f} seconds")
            
            return {
                "structured_score": round((
                    keyword_score * 0.3 +
                    semantic_score * 0.7  
                ), 2),
                "keyword_match": round(keyword_score, 2),
                "semantic_score": round(semantic_score, 2),
                "matching_keywords": {
                    "required": [k for k in required_keywords if any(
                        self._is_similar_keyword(k, cv_k) for cv_k in cv_keywords
                    )],
                    "preferred": [k for k in preferred_keywords if any(
                        self._is_similar_keyword(k, cv_k) for cv_k in cv_keywords
                    )]
                }
            }

    def _is_similar_keyword(self, k1: str, k2: str) -> bool:
        """Check if keywords are similar using cached results"""
        cache_key = f"{k1}:{k2}"
        if cache_key in self.similarity_cache:   
            return self.similarity_cache[cache_key]
        
        result = self._compute_similarity(k1, k2)
        self.similarity_cache[cache_key] = result   
        return result
    
    def _get_lemma(self, word: str) -> str:
        """Get lemma with caching"""
        if word not in self.lemma_cache:
            self.lemma_cache[word] = self.nlp(word)[0].lemma_
        return self.lemma_cache[word]

    def _compute_similarity(self, k1: str, k2: str) -> bool:
        """Compute actual similarity between keywords"""
        # Exact match
        if k1 == k2:
            return True
            
        # Lemma match using cache
        k1_lemma = self._get_lemma(k1)
        k2_lemma = self._get_lemma(k2)
        if k1_lemma == k2_lemma:
            return True
            
        # Substring match
        if k1 in k2 or k2 in k1:
            return True
            
        return False
    
    def _get_llm_verification(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Get independent LLM verification score based on comprehensive criteria"""
        start_time = time()

        prompt = f"""
        You are a skilled ATS (Applicant Tracking System) scanner and hiring manager with expertise in tech roles such as Product Management, Software Engineering and AI/ML. 
        Analyze this CV against the job description and return ONLY a JSON response with scores.

        CV Content:
        {inputs['cv_text']}
        
        Job Description:
        {inputs['job_description']}
        
        Score each major category (0-100) based on these weighted factors:

        1. Relevance (30% of final score):
        - Core Skills & Technical Fit (30%)
        - Job Responsibilities Match (30%)
        - Domain Knowledge & Industry Fit (20%)
        - Stakeholder & Collaboration Alignment (20%)

        2. Keyword Optimization (20% of final score):
        - Match of Critical Keywords (50%)
        - Tools & Technologies Match (25%)
        - ATS-Friendly Formatting (15%)
        - Keyword Density Balance (10%)

        3. Formatting & Presentation (15% of final score):
        - Clarity & Readability (30%)
        - Logical Section Flow (20%)
        - ATS Compatibility (40%)
        - Conciseness & Spacing (10%)

        4. Achievements & Qualifications (20% of final score):
        - Quantified Business Impact (40%)
        - Leadership & Cross-Functional Collaboration (30%)
        - Innovation & Problem-Solving (30%)

        5. Brevity & Clarity (15% of final score):
        - Length (Max 2 Pages) (30%)
        - Avoidance of Redundant Info (30%)
        - Concise Yet Impactful Statements (40%)

        Return your analysis in the following JSON format EXACTLY:
        {{
            "llm_score": <weighted final score 0-100>,
            "category_scores": {{
                "relevance": {{
                    "score": <0-100>,
                    "technical_fit": <0-100>,
                    "responsibilities_match": <0-100>,
                    "domain_knowledge": <0-100>,
                    "stakeholder_alignment": <0-100>
                }},
                "keyword_optimization": {{
                    "score": <0-100>,
                    "critical_keywords": <0-100>,
                    "tools_match": <0-100>,
                    "ats_formatting": <0-100>,
                    "keyword_balance": <0-100>
                }},
                "formatting": {{
                    "score": <0-100>,
                    "clarity": <0-100>,
                    "section_flow": <0-100>,
                    "ats_compatibility": <0-100>,
                    "conciseness": <0-100>
                }},
                "achievements": {{
                    "score": <0-100>,
                    "business_impact": <0-100>,
                    "leadership_collaboration": <0-100>,
                    "innovation": <0-100>
                }},
                "brevity": {{
                    "score": <0-100>,
                    "length": <0-100>,
                    "redundancy": <0-100>,
                    "impact": <0-100>
                }}
            }}
        }}
        
        Calculate the final llm_score as: 
        (relevance * 0.30) + (keyword_optimization * 0.20) + (formatting * 0.15) + (achievements * 0.20) + (brevity * 0.15)

        ONLY return valid JSON with no additional commentary. 
        If there's missing data, do your best to estimate.
        """
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the response text - remove any markdown code block indicators
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                llm_analysis = json.loads(response_text)
                
                # Calculate exact category scores without rounding
                for category in llm_analysis["category_scores"]:
                    scores = llm_analysis["category_scores"][category]
                    
                    if category == "relevance":
                        scores["score"] = (
                            scores["technical_fit"] * 0.30 +
                            scores["responsibilities_match"] * 0.30 +
                            scores["domain_knowledge"] * 0.20 +
                            scores["stakeholder_alignment"] * 0.20
                        )
                    
                    elif category == "keyword_optimization":
                        scores["score"] = (
                            scores["critical_keywords"] * 0.50 +
                            scores["tools_match"] * 0.25 +
                            scores["ats_formatting"] * 0.15 +
                            scores["keyword_balance"] * 0.10
                        )
                    
                    elif category == "formatting":
                        scores["score"] = (
                            scores["clarity"] * 0.30 +
                            scores["section_flow"] * 0.20 +
                            scores["ats_compatibility"] * 0.40 +
                            scores["conciseness"] * 0.10
                        )
                    
                    elif category == "achievements":
                        scores["score"] = (
                            scores["business_impact"] * 0.40 +
                            scores["leadership_collaboration"] * 0.30 +
                            scores["innovation"] * 0.30
                        )
                    
                    elif category == "brevity":
                        scores["score"] = (
                            scores["length"] * 0.30 +
                            scores["redundancy"] * 0.30 +
                            scores["impact"] * 0.40
                        )
                
                # Calculate final score with full precision
                final_score = (
                    llm_analysis["category_scores"]["relevance"]["score"] * 0.30 +
                    llm_analysis["category_scores"]["keyword_optimization"]["score"] * 0.20 +
                    llm_analysis["category_scores"]["formatting"]["score"] * 0.15 +
                    llm_analysis["category_scores"]["achievements"]["score"] * 0.20 +
                    llm_analysis["category_scores"]["brevity"]["score"] * 0.15
                )
                
                llm_analysis["llm_score"] = final_score

                self.logger.info(f"LLM verification took {time() - start_time:.2f} seconds")
                return {
                    "llm_score": final_score, 
                    "category_scores": llm_analysis["category_scores"]
                }
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON parsing error: {je}")
                self.logger.error(f"Raw response: {response_text}")
                return self._get_default_llm_scores()
                
        except Exception as e:
            self.logger.error(f"Error in LLM verification: {e}")
            return self._get_default_llm_scores()

    def _get_default_llm_scores(self) -> Dict[str, Any]:
        """Return default scores in case of LLM failure"""
        return {
            "llm_score": 0,
            "category_scores": {
                "relevance": {"score": 0, "technical_fit": 0, "responsibilities_match": 0, "domain_knowledge": 0, "stakeholder_alignment": 0},
                "keyword_optimization": {"score": 0, "critical_keywords": 0, "tools_match": 0, "ats_formatting": 0, "keyword_balance": 0},
                "formatting": {"score": 0, "clarity": 0, "section_flow": 0, "ats_compatibility": 0, "conciseness": 0},
                "achievements": {"score": 0, "business_impact": 0, "leadership_collaboration": 0, "innovation": 0},
                "brevity": {"score": 0, "length": 0, "redundancy": 0, "impact": 0}
            }
        }
    
    def _blend_scores(self, structured_score: float, llm_score: float) -> float:
        """Blend structured and LLM scores with weights"""
        return round(
            structured_score * 0.7 +  
            llm_score * 0.3,          
            2
        )
    def _calculate_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate composite scores based on various metrics"""
        structured_analysis = result["structured_analysis"]
        llm_analysis = result["llm_analysis"]
        category_scores = llm_analysis["category_scores"]
        
        # Calculate overall ATS score
        overall_ats_score = (
            0.40 * result["structured_analysis"]["structured_score"] +
            0.40 * result["llm_analysis"]["llm_score"] +
            0.20 * category_scores["formatting"]["ats_compatibility"]
        )
        
        # Calculate keyword match score
        keyword_match_score = (
            0.40 * structured_analysis["keyword_match"] +
            0.60 * category_scores["keyword_optimization"]["score"]
        )
        
        # Calculate technical fit
        technical_fit = (
            0.70 * category_scores["relevance"]["technical_fit"] +
            0.30 * structured_analysis["semantic_score"]
        )
        
        # Calculate formatting score
        formatting = (
            0.70 * category_scores["formatting"]["score"] +
            0.15 * category_scores["brevity"]["score"] +
            0.15 * category_scores["brevity"]["score"]
        )
        
        # Calculate role match
        role_match = (
            0.40 * category_scores["relevance"]["responsibilities_match"] +
            0.25 * category_scores["relevance"]["domain_knowledge"] +
            0.25 * category_scores["relevance"]["stakeholder_alignment"] +
            0.10 * structured_analysis["semantic_score"]
        )
        
        return {
            "overall_ats_score": round(overall_ats_score, 1),
            "keyword_match_score": round(keyword_match_score, 1),
            "technical_fit": round(technical_fit, 1),
            "role_match": round(role_match, 1),
            "formatting": round(formatting, 1)
        }

    def _format_output(self, result: Dict[str, Any]) -> str:
        # Calculate composite scores
        scores = self._calculate_scores(result)
        
        # Get strengths and areas for focus
        strengths = self._get_top_strengths(result)
        areas_for_focus = self._get_areas_for_focus(result)
        
        # Format the output using the template
        return self.OUTPUT_TEMPLATE.format(
            overall_ats_score=scores["overall_ats_score"],
            keyword_match_score=scores["keyword_match_score"],
            technical_fit=scores["technical_fit"],
            role_match=scores["role_match"],
            formatting=scores["formatting"],
            strengths=strengths,
            areas_for_focus=areas_for_focus
        )
    
    def _get_top_strengths(self, result: Dict[str, Any]) -> str:
        """Extract top scoring areas from the analysis results"""
        scores = []
        category_scores = result["llm_analysis"]["category_scores"]
        calculated_scores = self._calculate_scores(result)
        
        # Add our main calculated scores
        scores.extend([
            {"description": "Technical Fit", "score": calculated_scores["technical_fit"]},
            {"description": "Role Match", "score": calculated_scores["role_match"]},
            {"description": "Keyword Match", "score": calculated_scores["keyword_match_score"]},
            {"description": "Formatting", "score": calculated_scores["formatting"]}
        ])
        
        # Map of additional unique metrics we want to track (NO duplicates with calculated scores)
        metrics_map = {
            "achievements": [
                "business_impact",
                "leadership_collaboration",
                "innovation"
            ],
            "formatting": [
                "clarity",
                "section_flow"
            ],
            "brevity": [
                "impact",
                "redundancy"
            ]
        }
        
        # Add unique category scores
        for category, metrics in metrics_map.items():
            for metric in metrics:
                if score := category_scores.get(category, {}).get(metric):
                    scores.append({
                        "description": f"{metric.replace('_', ' ').title()}",
                        "score": score
                    })
        
        # Sort by score and get top 4
        top_scores = sorted(scores, key=lambda x: x["score"], reverse=True)[:4]
        
        # Emoji map for all possible scores
        emoji_map = {
            "technical fit": "💻",
            "role match": "🎯",
            "keyword match": "🔑",
            "formatting": "📝",
            "business impact": "📈",
            "leadership collaboration": "👥",
            "innovation": "💡",
            "clarity": "✨",
            "section flow": "📋",
            "impact": "🎯",
            "redundancy": "✂️"
        }
        
        return "\n".join(
            f"• {emoji_map.get(item['description'].lower(), '•')} {item['description']}"
            for item in top_scores
        )

    def _get_areas_for_focus(self, result: Dict[str, Any]) -> str:
        """Extract areas needing improvement from the analysis results"""
        scores = []
        category_scores = result["llm_analysis"]["category_scores"]
        
        # Map of metrics to track for improvements
        metrics_map = {
            "relevance": [
                "technical_fit",
                "responsibilities_match",
                "domain_knowledge"
            ],
            "keyword_optimization": [
                "critical_keywords",
                "tools_match",
                "keyword_balance"
            ],
            "formatting": [
                "ats_compatibility",
                "clarity",
                "conciseness"
            ],
            "achievements": [
                "business_impact",
                "leadership_collaboration",
                "innovation"
            ]
        }
        
        # Collect scores for specified metrics only
        for category, metrics in metrics_map.items():
            for metric in metrics:
                if score := category_scores.get(category, {}).get(metric):
                    if score < 75:  # Only include scores below 75%
                        scores.append({
                            "description": f"{metric.replace('_', ' ').title()}",
                            "score": score
                        })
        
        # Sort by score and get bottom 3
        bottom_scores = sorted(scores, key=lambda x: x["score"])[:3]
        
        # If no low scores found, provide general improvement areas
        if not bottom_scores:
            return "• All areas score above 75% - Focus on maintaining high standards"
        
        # Format into bullet points with improvement suggestions
        suggestions = {
            "technical fit": "Consider highlighting more relevant technical skills",
            "responsibilities match": "Align experience more closely with job requirements",
            "domain knowledge": "Emphasize industry-specific expertise",
            "critical keywords": "Include more job-specific keywords",
            "tools match": "List more relevant tools and technologies",
            "keyword balance": "Distribute keywords more naturally",
            "ats compatibility": "Improve formatting for ATS systems",
            "clarity": "Make content more clear and concise",
            "conciseness": "Remove redundant information",
            "business impact": "Quantify achievements with metrics",
            "leadership collaboration": "Highlight team leadership experiences",
            "innovation": "Showcase problem-solving initiatives"
        }
        
        return "\n".join(
            f"• {item['description']}\n  ↳ {suggestions.get(item['description'].lower(), 'Focus on improvement in this area')}"
            for item in bottom_scores
        )

    def _run(self, cv_text: str, job_description: str) -> Dict[str, Any]:
        start_time = time()

        pr = cProfile.Profile()
        pr.enable()

        try:
            cv_text = cv_text.replace('\\n', ' ').replace('\n', ' ').replace('  ', ' ').strip()
            job_description = job_description.replace('\\n', ' ').replace('\n', ' ').replace('  ', ' ').strip()
       
            inputs = {"cv_text": cv_text, "job_description": job_description}

            # Get structured analysis
            structured_analysis = self._get_structured_analysis(inputs)
            # Get LLM verification
            llm_analysis = self._get_llm_verification(inputs)

            # Return complete analysis
            result = {
                "structured_analysis": structured_analysis,
                "llm_analysis": llm_analysis
            }

            result["formatted_output"] = self._format_output(result)

            self.logger.info(f"Total analysis took {time() - start_time:.2f} seconds")       
        
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 time-consuming functions
            print(s.getvalue())
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in CV analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}