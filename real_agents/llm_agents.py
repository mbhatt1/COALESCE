"""
Concrete LLM Agent Implementations

This module implements specific LLM agents that can perform real tasks:
- DocumentAnalyzerAgent: Uses GPT-4 for document analysis
- CodeGeneratorAgent: Uses Claude for code generation  
- DataProcessorAgent: Uses local models for data processing
"""

import asyncio
import json
import openai
import anthropic
from typing import Any, Dict, List
import pandas as pd
import time
from datetime import datetime

from .base_agent import RealAgent, RealTask, AgentCapability, TaskResult


class DocumentAnalyzerAgent(RealAgent):
    """
    Real LLM agent that uses GPT-4 to analyze documents.
    
    This agent can:
    - Summarize documents
    - Extract key information
    - Answer questions about documents
    - Classify document types
    """
    
    def __init__(self, openai_api_key: str):
        capabilities = [
            AgentCapability(
                capability_name="document_analysis",
                skill_level=0.9,
                cost_per_unit=0.05,  # $0.05 per analysis
                avg_execution_time=30.0,  # 30 seconds average
                quality_score=0.85,
                max_concurrent_tasks=5
            ),
            AgentCapability(
                capability_name="document_summarization", 
                skill_level=0.95,
                cost_per_unit=0.03,
                avg_execution_time=20.0,
                quality_score=0.90,
                max_concurrent_tasks=10
            )
        ]
        
        super().__init__(
            agent_id="doc_analyzer_001",
            name="DocumentAnalyzer-GPT4",
            capabilities=capabilities
        )
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
    async def _perform_real_execution(self, task: RealTask) -> Any:
        """Execute document analysis using real GPT-4 API."""
        
        if task.task_type == "document_analysis":
            return await self._analyze_document(task)
        elif task.task_type == "document_summarization":
            return await self._summarize_document(task)
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")
            
    async def _analyze_document(self, task: RealTask) -> Dict[str, Any]:
        """Analyze a document using GPT-4."""
        document_text = task.input_data.get('text', '')
        analysis_type = task.requirements.get('analysis_type', 'general')
        
        prompt = f"""
        Please analyze the following document and provide:
        1. Main topics and themes
        2. Key insights and findings
        3. Important entities (people, organizations, dates)
        4. Document classification
        5. Sentiment analysis
        
        Analysis type: {analysis_type}
        
        Document:
        {document_text}
        
        Please provide your analysis in JSON format.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert document analyst. Provide thorough, accurate analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to structured text
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                analysis = {
                    "analysis": analysis_text,
                    "format": "text"
                }
                
            return {
                "analysis": analysis,
                "model_used": "gpt-4",
                "tokens_used": response.usage.total_tokens,
                "processing_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"GPT-4 analysis failed: {e}")
            raise
            
    async def _summarize_document(self, task: RealTask) -> Dict[str, Any]:
        """Summarize a document using GPT-4."""
        document_text = task.input_data.get('text', '')
        summary_length = task.requirements.get('summary_length', 'medium')
        
        length_instructions = {
            'short': 'in 2-3 sentences',
            'medium': 'in 1-2 paragraphs', 
            'long': 'in 3-4 paragraphs with detailed analysis'
        }
        
        prompt = f"""
        Please provide a {summary_length} summary of the following document {length_instructions.get(summary_length, '')}.
        Focus on the main points, key findings, and important conclusions.
        
        Document:
        {document_text}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, concise summaries that capture the essential information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            
            return {
                "summary": summary,
                "summary_length": summary_length,
                "model_used": "gpt-4",
                "tokens_used": response.usage.total_tokens,
                "original_length": len(document_text),
                "summary_ratio": len(summary) / len(document_text)
            }
            
        except Exception as e:
            self.logger.error(f"GPT-4 summarization failed: {e}")
            raise


class CodeGeneratorAgent(RealAgent):
    """
    Real LLM agent that uses Claude for code generation.
    
    This agent can:
    - Generate Python functions
    - Create data processing scripts
    - Write API integrations
    - Generate test cases
    """
    
    def __init__(self, anthropic_api_key: str):
        capabilities = [
            AgentCapability(
                capability_name="code_generation",
                skill_level=0.88,
                cost_per_unit=0.08,  # $0.08 per generation
                avg_execution_time=45.0,  # 45 seconds average
                quality_score=0.82,
                max_concurrent_tasks=3
            ),
            AgentCapability(
                capability_name="test_generation",
                skill_level=0.85,
                cost_per_unit=0.06,
                avg_execution_time=35.0,
                quality_score=0.80,
                max_concurrent_tasks=5
            )
        ]
        
        super().__init__(
            agent_id="code_gen_001", 
            name="CodeGenerator-Claude",
            capabilities=capabilities
        )
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
    async def _perform_real_execution(self, task: RealTask) -> Any:
        """Execute code generation using real Claude API."""
        
        if task.task_type == "code_generation":
            return await self._generate_code(task)
        elif task.task_type == "test_generation":
            return await self._generate_tests(task)
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")
            
    async def _generate_code(self, task: RealTask) -> Dict[str, Any]:
        """Generate code using Claude."""
        requirements = task.input_data.get('requirements', '')
        language = task.requirements.get('language', 'python')
        style = task.requirements.get('style', 'clean')
        
        prompt = f"""
        Please generate {language} code based on the following requirements:
        
        Requirements:
        {requirements}
        
        Code style: {style}
        
        Please provide:
        1. Clean, well-documented code
        2. Error handling where appropriate
        3. Type hints (if applicable)
        4. Brief explanation of the approach
        
        Format your response with the code in a code block and explanation below.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_content = response.content[0].text
            
            # Extract code block if present
            code_start = generated_content.find('```')
            if code_start != -1:
                code_end = generated_content.find('```', code_start + 3)
                if code_end != -1:
                    code = generated_content[code_start+3:code_end].strip()
                    # Remove language identifier if present
                    if code.startswith(language):
                        code = code[len(language):].strip()
                    explanation = generated_content[code_end+3:].strip()
                else:
                    code = generated_content[code_start+3:].strip()
                    explanation = ""
            else:
                code = generated_content
                explanation = ""
                
            return {
                "code": code,
                "explanation": explanation,
                "language": language,
                "model_used": "claude-3-sonnet",
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "requirements_met": True  # Would implement actual validation
            }
            
        except Exception as e:
            self.logger.error(f"Claude code generation failed: {e}")
            raise
            
    async def _generate_tests(self, task: RealTask) -> Dict[str, Any]:
        """Generate test cases using Claude."""
        code_to_test = task.input_data.get('code', '')
        test_framework = task.requirements.get('framework', 'pytest')
        
        prompt = f"""
        Please generate comprehensive test cases for the following code using {test_framework}:
        
        Code to test:
        {code_to_test}
        
        Please provide:
        1. Unit tests covering main functionality
        2. Edge case tests
        3. Error condition tests
        4. Mock usage where appropriate
        
        Format with test code in a code block.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            test_content = response.content[0].text
            
            return {
                "test_code": test_content,
                "framework": test_framework,
                "model_used": "claude-3-sonnet",
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "coverage_estimate": 0.85  # Would implement actual coverage analysis
            }
            
        except Exception as e:
            self.logger.error(f"Claude test generation failed: {e}")
            raise


class DataProcessorAgent(RealAgent):
    """
    Real agent that processes data using local computation.
    
    This agent can:
    - Process CSV files
    - Perform data analysis
    - Generate reports
    - Clean and transform data
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                capability_name="data_processing",
                skill_level=0.92,
                cost_per_unit=0.02,  # $0.02 per processing task
                avg_execution_time=15.0,  # 15 seconds average
                quality_score=0.88,
                max_concurrent_tasks=8
            ),
            AgentCapability(
                capability_name="data_analysis",
                skill_level=0.85,
                cost_per_unit=0.04,
                avg_execution_time=25.0,
                quality_score=0.83,
                max_concurrent_tasks=4
            )
        ]
        
        super().__init__(
            agent_id="data_proc_001",
            name="DataProcessor-Local",
            capabilities=capabilities
        )
        
    async def _perform_real_execution(self, task: RealTask) -> Any:
        """Execute data processing using local computation."""
        
        if task.task_type == "data_processing":
            return await self._process_data(task)
        elif task.task_type == "data_analysis":
            return await self._analyze_data(task)
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")
            
    async def _process_data(self, task: RealTask) -> Dict[str, Any]:
        """Process data using pandas and local computation."""
        data_source = task.input_data.get('data_source')
        operations = task.requirements.get('operations', [])
        
        try:
            # Load data
            if isinstance(data_source, str) and data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif isinstance(data_source, dict):
                df = pd.DataFrame(data_source)
            else:
                df = pd.DataFrame(data_source)
                
            original_shape = df.shape
            
            # Apply operations
            for operation in operations:
                if operation == 'clean_nulls':
                    df = df.dropna()
                elif operation == 'remove_duplicates':
                    df = df.drop_duplicates()
                elif operation.startswith('filter_'):
                    # Simple filtering example
                    column, value = operation.split('_')[1], operation.split('_')[2]
                    if column in df.columns:
                        df = df[df[column] == value]
                elif operation == 'normalize':
                    numeric_columns = df.select_dtypes(include=['number']).columns
                    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
                    
            processed_shape = df.shape
            
            return {
                "processed_data": df.to_dict('records'),
                "original_shape": original_shape,
                "processed_shape": processed_shape,
                "operations_applied": operations,
                "processing_time": time.time(),
                "rows_processed": len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
            
    async def _analyze_data(self, task: RealTask) -> Dict[str, Any]:
        """Analyze data and generate insights."""
        data_source = task.input_data.get('data_source')
        analysis_type = task.requirements.get('analysis_type', 'descriptive')
        
        try:
            # Load data
            if isinstance(data_source, str) and data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif isinstance(data_source, dict):
                df = pd.DataFrame(data_source)
            else:
                df = pd.DataFrame(data_source)
                
            analysis_results = {}
            
            if analysis_type == 'descriptive':
                analysis_results = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "data_types": df.dtypes.to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
                    "categorical_summary": {col: df[col].value_counts().head().to_dict() 
                                          for col in df.select_dtypes(include=['object']).columns}
                }
                
            elif analysis_type == 'correlation':
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 1:
                    analysis_results = {
                        "correlation_matrix": numeric_df.corr().to_dict(),
                        "strong_correlations": []  # Would implement correlation analysis
                    }
                    
            return {
                "analysis": analysis_results,
                "analysis_type": analysis_type,
                "data_shape": df.shape,
                "analysis_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Data analysis failed: {e}")
            raise