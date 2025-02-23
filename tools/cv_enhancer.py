"""
CV Enhancement Tool Module

This module provides functionality for analyzing and improving CVs/resumes based on
job descriptions and analysis results. It offers targeted suggestions for various
CV sections including summary, experience, skills, and achievements.
"""

# Standard library imports
import json
import logging
from typing import Dict, Any, List, Type

# Third party imports
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)


class CVEnhancerInput(BaseModel):
    """
    Input schema for CV enhancement requests.

    Attributes:
        cv_analysis (Dict[str, Any]): Analysis results from CV Analyzer
        cv_text (str): Original CV text
        job_description (str): Original job description
        focus_area (str, optional): Specific area to focus improvements on
    """
    cv_analysis: Dict[str, Any] = Field(
        ..., description="Analysis results from CV Analyzer"
    )
    cv_text: str = Field(..., description="Original CV text")
    job_description: str = Field(..., description="Original job description")
    focus_area: str = Field(
        None,
        description="Specific area to focus improvements on (e.g., 'summary', 'tools', etc.)",
    )


class CVEnhancerTool(BaseTool):
    """
    A tool for enhancing CVs based on analysis results.

    This tool provides targeted recommendations for improving CV content
    and format based on detailed analysis scores and job requirements.

    Attributes:
        name (str): Name of the tool
        description (str): Description of the tool's functionality
        args_schema (Type[BaseModel]): Schema for input validation
        llm (BaseChatModel): Language model for generating suggestions
    """

    name: str = "cv_enhancer"
    description: str = (
        "Use this tool to get detailed improvement suggestions for a CV "
        "based on analysis results from the cv_analyzer.\n"
        "ONLY use when:\n"
        "1. CV analysis has been completed\n"
        "2. User explicitly requests CV improvement suggestions\n"
        "3. You have both the original CV and job description available\n\n"
        "The tool provides targeted recommendations for improving CV "
        "content and format."
    )

    args_schema: Type[BaseModel] = CVEnhancerInput

    llm: BaseChatModel = Field(description="LLM for generating suggestions")

    def __init__(self, llm: BaseChatModel):
        # Create the LLM instance if none provided
        if llm is None:
            llm = ChatOpenAI(
                temperature=0.7, model="gpt-4-turbo-preview", max_tokens=4000
            )

        # Initialize parent class with all required fields
        super().__init__(llm=llm)

    def _get_base_rules(self) -> str:
        """
        Get base rules for CV enhancement.

        Returns:
            str: Base rules for CV enhancement process
        """
        return """
        IMPORTANT RULES:
        1. Never modify stated years of experience
        2. Never add skills or experiences not mentioned in the original CV
        3. Never change or fabricate achievements, metrics, or roles
        4. Only reorganize and rephrase existing content to better match the job requirements
        """

    def _extract_scores(
        self, cv_analysis: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract and normalize scores from CV analysis results.

        Args:
            cv_analysis (Dict[str, Any]): Raw CV analysis results

        Returns:
            Dict[str, Dict[str, float]]: Normalized scores by category

        Raises:
            ValueError: If score parsing fails
            TypeError: If score values are invalid
        """
        try:
            # Get structured analysis scores (keyword and semantic matching)
            structured_data = cv_analysis.get("structured_analysis", {})
            keyword_score = float(structured_data.get("keyword_match_score", 75))
            semantic_score = float(structured_data.get("semantic_similarity_score", 75))

            # Get LLM analysis scores for qualitative categories
            llm_data = cv_analysis.get("llm_analysis", {})
            llm_categories = llm_data.get("category_scores", {})

            return {
                "technical_match": {
                    "keyword_score": keyword_score,
                    "semantic_score": semantic_score,
                    "tools_match": float(
                        llm_categories.get("keyword_optimization", {}).get(
                            "tools_match", 75
                        )
                    ),
                },
                "content_quality": {
                    "relevance": float(
                        llm_categories.get("relevance", {}).get("score", 75)
                    ),
                    "technical_fit": float(
                        llm_categories.get("relevance", {}).get("technical_fit", 75)
                    ),
                    "domain_knowledge": float(
                        llm_categories.get("relevance", {}).get("domain_knowledge", 75)
                    ),
                },
                "formatting": {
                    "clarity": float(
                        llm_categories.get("formatting", {}).get("clarity", 75)
                    ),
                    "ats_compatibility": float(
                        llm_categories.get("formatting", {}).get(
                            "ats_compatibility", 75
                        )
                    ),
                },
                "achievements": {
                    "business_impact": float(
                        llm_categories.get("achievements", {}).get(
                            "business_impact", 75
                        )
                    ),
                    "leadership": float(
                        llm_categories.get("achievements", {}).get(
                            "leadership_collaboration", 75
                        )
                    ),
                },
                "brevity": {
                    "length": float(
                        llm_categories.get("brevity", {}).get("length", 75)
                    ),
                    "impact": float(
                        llm_categories.get("brevity", {}).get("impact", 75)
                    ),
                },
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing scores: {e}")
            return {
                "technical_match": {"keyword_score": 75.0, "semantic_score": 75.0},
                "content_quality": {"relevance": 75.0, "technical_fit": 75.0},
                "formatting": {"clarity": 75.0, "ats_compatibility": 75.0},
                "achievements": {"business_impact": 75.0, "leadership": 75.0},
                "brevity": {"length": 75.0, "impact": 75.0},
            }

    def _get_score_based_improvements(
        self, scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[str]]:
        improvements = {}

        # Technical Match Improvements (using both structured and LLM scores)
        if (
            scores["technical_match"]["keyword_score"] < 85
            or scores["technical_match"]["semantic_score"] < 85
        ):
            improvements["technical_match"] = [
                f"Keyword Match (Score: {scores['technical_match']['keyword_score']}): Add missing technical keywords from job description",
                f"Semantic Match (Score: {scores['technical_match']['semantic_score']}): Align terminology with industry standards",
                f"Tools Match (Score: {scores['technical_match']['tools_match']}): Highlight relevant technical tools",
            ]

        # Content Quality Improvements (LLM-based)
        if any(score < 85 for score in scores["content_quality"].values()):
            improvements["content_quality"] = [
                f"Technical Fit (Score: {scores['content_quality']['technical_fit']}): Enhance technical expertise presentation",
                f"Domain Knowledge (Score: {scores['content_quality']['domain_knowledge']}): Strengthen industry alignment",
                f"Relevance (Score: {scores['content_quality']['relevance']}): Improve alignment with job requirements",
            ]

        # Formatting Improvements (LLM-based)
        if any(score < 85 for score in scores["formatting"].values()):
            improvements["formatting"] = [
                f"Clarity (Score: {scores['formatting']['clarity']}): Improve section organization and readability",
                f"ATS Compatibility (Score: {scores['formatting']['ats_compatibility']}): Optimize structure for ATS systems",
            ]

        # Achievements Improvements (LLM-based)
        if any(score < 85 for score in scores["achievements"].values()):
            improvements["achievements"] = [
                f"Business Impact (Score: {scores['achievements']['business_impact']}): Add quantifiable metrics and results",
                f"Leadership (Score: {scores['achievements']['leadership']}): Highlight team leadership and project management",
            ]

        # Brevity Improvements (LLM-based)
        if any(score < 85 for score in scores["brevity"].values()):
            improvements["brevity"] = [
                f"Length (Score: {scores['brevity']['length']}): Optimize content length and conciseness",
                f"Impact (Score: {scores['brevity']['impact']}): Focus on most significant achievements and skills",
            ]

        return improvements

    def _validate_focus_area(self, focus_area: str) -> str:
        """
        Validate and normalize the focus area for CV enhancement.

        Args:
            focus_area (str): Requested focus area

        Returns:
            str: Normalized focus area name
        """

        valid_areas = {
            "summary": ["summary", "profile", "overview"],
            "experience": ["experience", "work history", "roles"],
            "skills": ["skills", "technologies", "tools"],
            "achievements": ["achievements", "accomplishments"],
            "education": ["education", "qualifications"],
            "general": ["general", "all", "complete"],
        }

        focus_area = focus_area.lower() if focus_area else "general"
        for main_area, aliases in valid_areas.items():
            if focus_area in aliases:
                return main_area
        return "general"

    def _generate_analysis_text(self, category: str, scores: Dict) -> str:
        """
        Generate analysis text for a specific category.

        Args:
            category (str): Category to analyze
            scores (Dict): Scores for the category

        Returns:
            str: Formatted analysis text
        """
        category_data = scores.get(category, {})

        if category == "relevance":
            return f"Technical fit: {category_data.get('technical_fit', 0)}, Domain knowledge: {category_data.get('domain_knowledge', 0)}"
        elif category == "keyword_optimization":
            return f"Critical keywords: {category_data.get('critical_keywords', 0)}, Tools match: {category_data.get('tools_match', 0)}"
        elif category == "formatting":
            return f"Clarity: {category_data.get('clarity', 0)}, ATS compatibility: {category_data.get('ats_compatibility', 0)}"
        elif category == "achievements":
            return f"Business impact: {category_data.get('business_impact', 0)}, Leadership: {category_data.get('leadership_collaboration', 0)}"
        elif category == "brevity":
            return f"Length: {category_data.get('length', 0)}, Impact: {category_data.get('impact', 0)}"
        return "Analysis not available"

    def _generate_full_evaluation(
        self, base_context: str, scores: Dict, cv_text: str, job_description: str
    ) -> Dict:
        """
        Generate a comprehensive CV evaluation and improvement suggestions.

        Args:
            base_context (str): Base context for LLM prompt
            scores (Dict): Analysis scores
            cv_text (str): Original CV text
            job_description (str): Job description

        Returns:
            Dict: Full evaluation results and suggestions

        Raises:
            Exception: If evaluation generation fails
        """
        # Get score-based improvements
        score_improvements = self._get_score_based_improvements(scores)

        prompt = f"""{base_context}
        Original CV: {cv_text}
        Job Description: {job_description}

        DETAILED SCORE ANALYSIS AND IMPROVEMENTS NEEDED:
        {'-' * 50}
        """

        # Add score-based improvements to prompt
        for category, improvements in score_improvements.items():
            prompt += f"\n{category.upper()}:"
            for improvement in improvements:
                prompt += f"\n- {improvement}"
            prompt += "\n"

        prompt += f"""
        {'-' * 50}
        
        Based on the above score analysis, provide a comprehensive CV enhancement focusing on:
        1. High Priority Areas: {', '.join(k for k, imps in score_improvements.items() if any('score: ' + str(s) for s in [float(x.split('Score: ')[1].split(')')[0]) for x in imps] if float(s) < 80))}
        2. Medium Priority Areas: {', '.join(k for k, imps in score_improvements.items() if any('score: ' + str(s) for s in [float(x.split('Score: ')[1].split(')')[0]) for x in imps] if 80 <= float(s) < 85))}
        
        REQUIRED OUTPUT STRUCTURE:
        1. Start with "🔍 Key Areas to Improve:" followed by numbered sections
        2. For each section:
        - Current State ("Currently:")
        - Problem Analysis ("Fix:")
        - Specific Example ("💡 Example Revision:")
        - Before/After comparisons ("✔ Before:" / "✔ After:")
        
        3. Example Implementation:
        FULL DETAILED REWRITE with:
        - Complete paragraph(s) for each section
        - Integration of all key technologies
        - Specific metrics and achievements
        - Industry-relevant terminology
        
        4. Technical Focus:
        - Required frameworks and tools
        - Technical skills alignment
        - Industry-specific platforms

        SECTION REQUIREMENTS:
        1. Professional Summary:
        - Minimum 200 words
        - Include ALL relevant technologies
        - Highlight leadership and technical expertise
        - Show industry alignment
        
        2. Work Experience:
        - Minimum 150 words per role
        - Include metrics and achievements
        - Show technical depth
        
        3. Skills & Technical Expertise:
        - Comprehensive technical stack
        - Grouped by category
        - Aligned with job requirements

        FORMAT GUIDELINES:
        1. Provide COMPLETE rewrites, not partial examples
        2. Include ALL technical details and frameworks
        3. Show full context and implementation
        4. Maintain professional tone and terminology
        5. Focus on quantifiable achievements
        6. Ensure job requirement alignment

        FORMAT YOUR RESPONSE IN THIS EXACT JSON STRUCTURE:
        {{
            "type": "full_evaluation",
            "improvements": {{
                "sections": [
                    {{
                        "name": "🔍 Section Name",
                        "subsections": [
                            {{
                                "title": "Detailed Section Title",
                                "current_state": "Comprehensive current analysis...",
                                "fix": [
                                    "Detailed improvement 1 with specific technical focus...",
                                    "Detailed improvement 2 with implementation guidance..."
                                ],
                                "example": {{
                                    "full_revision": "Complete detailed rewrite with all improvements...",
                                    "technical_focus": [
                                        "Specific framework 1 with implementation context",
                                        "Specific framework 2 with usage details"
                                    ]
                                }}
                            }}
                        ]
                    }}
                ]
            }}
        }}
        """

        try:
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            return json.loads(
                response_text.strip().replace("```json", "").replace("```", "").strip()
            )
        except Exception as e:
            return {
                "type": "error",
                "error": f"Failed to generate full evaluation: {str(e)}",
            }

    def _generate_section_improvement(
        self,
        base_context: str,
        focus_area: str,
        scores: Dict,
        cv_text: str,
        job_description: str,
    ) -> Dict:
        """
        Generate improvement suggestions for a specific CV section.

        Args:
            base_context (str): Base context for LLM prompt
            focus_area (str): Section to focus on
            scores (Dict): Analysis scores
            cv_text (str): Original CV text
            job_description (str): Job description

        Returns:
            Dict: Section-specific improvement suggestions

        Raises:
            Exception: If suggestion generation fails
        """

        section_prompts = {
            "summary": """
            Analyze and improve the professional summary section. Provide:
            1. Current State Analysis
            2. Specific Improvements Needed
            3. Complete Revision with focus on:
                - Technical expertise and specialization
                - Leadership and strategic capabilities
                - Industry-specific achievements
                - Key technologies and methodologies
            
            FORMAT YOUR RESPONSE IN THIS EXACT JSON STRUCTURE:
            {
                "type": "full_evaluation",
                "improvements": {
                    "sections": [
                        {
                            "name": "🔍 Professional Summary Improvements",
                            "subsections": [
                                {
                                    "title": "Professional Summary Analysis",
                                    "current_state": "Detailed analysis of current summary...",
                                    "fix": [
                                        "Improvement 1: Technical focus...",
                                        "Improvement 2: Leadership capabilities...",
                                        "Improvement 3: Industry alignment..."
                                    ],
                                    "example": {
                                        "full_revision": "Senior Product Manager with over 5 years of experience specializing in...",
                                        "technical_focus": [
                                            "• AI/ML Technologies: TensorFlow, PyTorch, Scikit-learn",
                                            "• Cloud Platforms: AWS, GCP",
                                            "• Analytics Tools: Google Analytics, Tableau"
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            """,
            "experience": """
            Analyze and improve each work experience entry individually. For each role, maintain the exact format:

            [Job Title] at [Company], [Location]
            [Date Range]
            [Previous role if applicable]
            • [Main responsibilities and achievements in bullet points]

            Key Highlights:
            • [Additional key achievements in bullet points]

            FORMAT YOUR RESPONSE IN THIS EXACT JSON STRUCTURE:
            {
                "type": "full_evaluation",
                "improvements": {
                    "sections": [
                        {
                            "name": "🔍 Work Experience Improvements",
                            "subsections": [
                                {
                                    "title": "[Job Title] at [Company], [Location]",
                                    "current_state": "Current role description analysis...",
                                    "fix": [
                                        "Improvement needed for main bullet points...",
                                        "Improvement needed for key highlights..."
                                    ],
                                    "example": {
                                        "full_revision": "Senior Product Manager at Ströer SSP GmbH, Berlin\nMay 2023 — December 2023\nPrevious role: Product Manager (July 2021 - April 2023)\n• Led the development of AI-powered ad server solutions\n• Implemented machine learning optimization strategies\n\nKey Highlights:\n• Increased revenue by 30% through ML-based optimizations\n• Reduced prediction time from minutes to 7 seconds\n• Improved user engagement by 20% through UX enhancements",
                                        "technical_focus": [
                                            "• ML/AI Implementation: TensorFlow, PyTorch",
                                            "• Analytics Tools: Google Analytics, Tableau"
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            }

            IMPORTANT FORMAT RULES:
            1. Always use bullet points (•) for achievements and responsibilities
            2. Maintain the exact structure:
            - Title, company, location on first line
            - Date range on second line
            - Previous role (if applicable) on third line
            - Main bullet points
            - "Key Highlights:" section with bullet points
            3. Each bullet point should:
            - Start with an action verb
            - Include specific metrics
            - Highlight technical implementations
            - Be concise but detailed
            """,
            "tools": """
            Analyze and improve the technical tools section. Organize by categories with bullet points:
            1. Current State Analysis
            2. Tools Categorization
            3. Complete Revision focusing on:
                - Programming Languages
                - Frameworks & Libraries
                - Cloud Platforms
                - Development Tools
                - Industry-specific Software
            
            FORMAT YOUR RESPONSE IN THIS EXACT JSON STRUCTURE:
            {
                "type": "full_evaluation",
                "improvements": {
                    "sections": [
                        {
                            "name": "🔍 Technical Tools Improvements",
                            "subsections": [
                                {
                                    "title": "Tools & Technologies Analysis",
                                    "current_state": "Current tools organization analysis...",
                                    "fix": [
                                        "Categorization improvements...",
                                        "Missing tools additions...",
                                        "Relevance enhancements..."
                                    ],
                                    "example": {
                                        "full_revision": "Tools & Technologies:\n\n• Programming Languages:\n  - Python\n  - SQL\n  - JavaScript\n\n• Frameworks & Libraries:\n  - TensorFlow\n  - PyTorch\n  - React\n\n• Cloud & Infrastructure:\n  - AWS\n  - Google Cloud Platform\n  - Docker\n\n• Analytics & Monitoring:\n  - Google Analytics\n  - Tableau\n  - Grafana\n\n• Project Management:\n  - Jira\n  - Confluence\n  - Trello\n\n• Version Control:\n  - Git\n  - GitHub\n  - GitLab",
                                        "technical_focus": [
                                            "• Primary Tools: AI/ML frameworks, Cloud platforms",
                                            "• Secondary Tools: Analytics, Project Management"
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            """,
            "skills": """
            Analyze and improve the technical skills section. Organize with bullet points by:
            1. Current State Analysis
            2. Skills Categorization
            3. Complete Revision organizing by:
                - Technical Skills
                - Domain Expertise
                - Methodologies
                - Soft Skills
                - Certifications
            
            FORMAT YOUR RESPONSE IN THIS EXACT JSON STRUCTURE:
            {
                "type": "full_evaluation",
                "improvements": {
                    "sections": [
                        {
                            "name": "🔍 Skills & Technical Expertise Improvements",
                            "subsections": [
                                {
                                    "title": "Technical Skills Analysis",
                                    "current_state": "Current skills organization analysis...",
                                    "fix": [
                                        "Technical skills improvements...",
                                        "Domain expertise enhancements...",
                                        "Methodology alignments..."
                                    ],
                                    "example": {
                                        "full_revision": "Skills & Technical Expertise:\n\n• Technical Skills:\n  - AI/ML Development\n  - Data Analytics\n  - Cloud Architecture\n  - Full-Stack Development\n\n• Domain Expertise:\n  - Product Management\n  - Agile Methodologies\n  - User Experience (UX)\n\n• Leadership & Management:\n  - Team Leadership\n  - Stakeholder Management\n  - Strategic Planning\n\n• Certifications:\n  - AWS Solutions Architect\n  - Professional Scrum Master\n  - Google Analytics",
                                        "technical_focus": [
                                            "• Core Technical Skills: AI/ML, Cloud, Analytics",
                                            "• Domain Expertise: Product, Agile, UX"
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            """,
            "achievements": """
            Analyze and improve the achievements section with bullet points focusing on:
            1. Current State Analysis
            2. Achievement Categories
            3. Complete Revision highlighting:
                - Technical Innovations
                - Business Impact
                - Team Leadership
                - Project Success
                - Industry Recognition
            
            FORMAT YOUR RESPONSE IN THIS EXACT JSON STRUCTURE:
            {
                "type": "full_evaluation",
                "improvements": {
                    "sections": [
                        {
                            "name": "🔍 Achievements Improvements",
                            "subsections": [
                                {
                                    "title": "Professional Achievements Analysis",
                                    "current_state": "Current achievements analysis...",
                                    "fix": [
                                        "Technical achievement improvements...",
                                        "Impact metric enhancements...",
                                        "Leadership highlight additions..."
                                    ],
                                    "example": {
                                        "full_revision": "Key Professional Achievements:\n\n• Technical Innovation:\n  - Developed AI-powered ad server increasing revenue by 30%\n  - Implemented ML models reducing prediction time by 95%\n\n• Business Impact:\n  - Drove 25% revenue growth through marketplace integration\n  - Achieved 40% improvement in customer satisfaction\n\n• Leadership & Team Success:\n  - Led cross-functional team of 15 members\n  - Mentored 5 junior product managers\n\n• Project Delivery:\n  - Completed major platform migration 2 months ahead of schedule\n  - Reduced operational costs by 35% through automation",
                                        "technical_focus": [
                                            "• Technical Achievements: AI/ML implementations",
                                            "• Business Metrics: Revenue, efficiency improvements"
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
            """,
        }

        prompt = f"""{base_context}
        Original CV: {cv_text}
        Job Description: {job_description}
        Focus Area: {focus_area}

        {section_prompts.get(focus_area, '''
        Generate specific improvements for this section in the standard format:
        {
            "type": "full_evaluation",
            "improvements": {
                "sections": [
                    {
                        "name": "🔍 Section Improvements",
                        "subsections": [
                            {
                                "title": "Detailed Section Analysis",
                                "current_state": "Current analysis...",
                                "fix": ["improvement 1", "improvement 2"],
                                "example": {
                                    "full_revision": "Complete revision...",
                                    "technical_focus": ["technical detail 1", "technical detail 2"]
                                }
                            }
                        ]
                    }
                ]
            }
        }
        ''')}

        IMPORTANT GUIDELINES:
        1. Use only information present in the original CV
        2. Do not invent new experiences or skills
        3. Maintain chronological order where applicable
        4. Include specific metrics and achievements
        5. Focus on technical implementations and skills
        6. Ensure all improvements align with the job description
        7. Provide detailed, actionable improvements
        8. Include minimum word counts where specified
        9. Organize information logically and clearly
        10. Emphasize relevant technical expertise
        
        SECTION-SPECIFIC REQUIREMENTS:
        - Professional Summary: Minimum 200 words, highlight technical expertise
        - Work Experience: Minimum 150 words per role, include metrics
        - Skills/Tools: Organize by categories, emphasize proficiency levels
        - Achievements: Quantify results and technical impact
        """

        try:
            response = self.llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            return json.loads(
                response_text.strip().replace("```json", "").replace("```", "").strip()
            )
        except Exception as e:
            return {
                "type": "error",
                "error": f"Failed to generate section improvement: {str(e)}",
            }

    def _run(
        self,
        cv_analysis: Dict[str, Any],
        cv_text: str,
        job_description: str,
        focus_area: str = None,
    ) -> Dict[str, Any]:
        """
        Run CV enhancement process.

        Args:
            cv_analysis (Dict[str, Any]): Analysis results from CV Analyzer
            cv_text (str): Original CV text
            job_description (str): Job description
            focus_area (str, optional): Specific area to focus on. Defaults to None.

        Returns:
            Dict[str, Any]: Enhancement suggestions and improvements

        Raises:
            Exception: If enhancement process fails
        """
        try:
            logger.debug("Starting CV enhancement process")
            scores = self._extract_scores(cv_analysis)
            logger.debug("Extracted scores: %s", scores)

            validated_focus_area = self._validate_focus_area(focus_area)
            logger.debug("Validated focus area: %s", validated_focus_area)

            # Updated base context to use new score structure
            base_context = f"""
            You are an expert tech recruiter and CV consultant. Your task is to enhance a CV based on the following scores:
            
            Technical Match Scores:
            - Keyword Match: {scores['technical_match']['keyword_score']}
            - Semantic Match: {scores['technical_match']['semantic_score']}
            - Tools Match: {scores['technical_match']['tools_match']}

            Content Quality Scores:
            - Relevance: {scores['content_quality']['relevance']}
            - Technical Fit: {scores['content_quality']['technical_fit']}
            - Domain Knowledge: {scores['content_quality']['domain_knowledge']}

            Formatting Scores:
            - Clarity: {scores['formatting']['clarity']}
            - ATS Compatibility: {scores['formatting']['ats_compatibility']}

            Achievement Scores:
            - Business Impact: {scores['achievements']['business_impact']}
            - Leadership: {scores['achievements']['leadership']}

            Brevity Scores:
            - Length: {scores['brevity']['length']}
            - Impact: {scores['brevity']['impact']}

            IMPORTANT: Use these exact scores to guide your suggestions. Do not generate new scores.
            
            {self._get_base_rules()}
            """

            logger.debug(
                "%s",
                "Generating full evaluation"
                if validated_focus_area == "general"
                else "Generating section improvement"
            )
            if validated_focus_area == "general":
                return self._generate_full_evaluation(
                    base_context,
                    scores,
                    cv_text,
                    job_description
                )
            
            return self._generate_section_improvement(
                base_context,
                validated_focus_area,
                scores,
                cv_text,
                job_description
            )

        except Exception as e:
            logger.error("Error processing CV analysis: %s", str(e), exc_info=True)
            return {"type": "error", "error": f"Failed to process analysis: {str(e)}"}
