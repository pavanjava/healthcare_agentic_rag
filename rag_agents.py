import os
import sys
from qdrant_client import QdrantClient
from fastembed.text import TextEmbedding
from pydantic import BaseModel, Field
from typing import Type, Any
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize qdrant client
qdrant_client = QdrantClient(url=os.environ.get('QDRANT_URL'), api_key=os.environ.get('QDRANT_API_KEY'))

# initialize the text embedding
embedding_model = TextEmbedding(model_name='snowflake/snowflake-arctic-embed-m')


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query")


class SearchMedicalHistoryTool(BaseTool):
    name: str = "search_medical_records"
    description: str = "Search through medical records using vector similarity"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> Any:
        # Use OpenAI embeddings to match data_loader.py
        query_vector = next(embedding_model.query_embed(query=query))

        search_results = qdrant_client.search(
            collection_name='medical_records',
            query_vector=query_vector,
            limit=10,
            score_threshold=0.7
        )

        return [
            {
                "score": hit.score,
                "text": hit.payload.get('text', 'N/A'),
            }
            for hit in search_results
        ]


def trigger_crew(query: str) -> str:
    # initialize the tools
    search_tool = SearchMedicalHistoryTool()

    # Create agents
    researcher = Agent(
        role='Research Assistant',
        goal='Find and analyze relevant information',
        backstory="""You are an expert at finding and analyzing information.
                  You know when to search medical history records, and when 
                  to perform detailed analysis.""",
        tools=[search_tool],
        verbose=True
    )

    synthesizer = Agent(
        role='Information Synthesizer',
        goal='Create comprehensive and clear responses',
        backstory="""You excel at taking raw information and analysis
                  and creating clear, and present them as actionable insights.""",
        verbose=True
    )

    # Create tasks with expected_output
    research_task = Task(
        description=f"""Process this query: '{query}'
                    2. If it needs medical history information, use the search tool.
                    3. For detailed analysis, use search tool.
                    Explain your tool selection and process.""",
        expected_output="""A dictionary containing:
                       - The tools used
                       - The raw results from each tool
                       - Any analysis performed""",
        agent=researcher
    )

    synthesis_task = Task(
        description="""Take the research results and create a clear response.
                    Explain the process used and why it was appropriate.
                    Make sure the response directly addresses the original query.""",
        expected_output="""A clear, structured response that includes:
                       - Direct answer to the query
                       - Supporting evidence from the research
                       - present it in the form of bullets""",
        agent=synthesizer
    )

    # Create and run crew
    crew = Crew(
        agents=[researcher, synthesizer],
        tasks=[research_task, synthesis_task],
        verbose=True
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (type 'bye' or 'quit' to exit): ").strip()

        if query.lower() in ['bye', 'quit']:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        try:
            result = trigger_crew(query)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
