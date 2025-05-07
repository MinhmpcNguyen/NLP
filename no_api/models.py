from pydantic import BaseModel, Field


class CrawlParams(BaseModel):
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "base_url": "https://docs.trava.finance/portal",
                    "max_depth": 1,
                    "bypass": True,
                }
            ]
        }

    base_url: str = Field(
        ..., description="The base URL to start fetching content from"
    )
    max_depth: int = Field(..., description="The maximum depth of sublink to crawl")
    bypass: bool = Field(False, description="Whether to bypass cache when crawling")


class DocumentRequest(BaseModel):
    
    base_threshold: float = Field(0.5, description="The base similarity threshold")
    lower_bound: float = Field(0.2, description="The lower bound for the similarity threshold")
    upper_bound: float = Field(0.8, description="The upper bound for the similarity threshold")
    variance_lower: float = Field(0.01, description="The lower bound for the variance threshold")
    variance_upper: float = Field(0.05, description="The upper bound for the variance threshold")
    average_lower: float = Field(0.3, description="The lower bound for the average threshold")
    average_upper: float = Field(0.7, description="The upper bound for the average threshold")
    decrease_by: float = Field(0.1, description="The decrease value for the similarity threshold")
    increase_by: float = Field(0.1, description="The increase value for the similarity threshold")
    num_similarity_paragraphs_lookahead: int = Field(8, description="The number of sentences to look ahead for similarity")


