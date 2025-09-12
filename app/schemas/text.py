from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class TextDetectRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw text to analyze")
    url: Optional[HttpUrl] = Field(None, description="URL to fetch and analyze")
    html: Optional[str] = Field(None, description="Raw HTML content to parse and analyze")
    file_path: Optional[str] = Field(None, description="Path to local text file to analyze")
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_at_least_one_input(cls, v, info):
        if info.field_name == 'text' or info.field_name == 'url' or info.field_name == 'html' or info.field_name == 'file_path':
            return v
        
        # This is a workaround - we'll do the actual validation in the service
        return v

class DetectionMethod(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    weight: float = Field(..., ge=0.0, le=1.0)

class HighlightSpan(BaseModel):
    start: int
    end: int
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str

class TextMetadata(BaseModel):
    word_count: int
    character_count: int
    unique_words: int
    avg_word_length: float
    processing_time_ms: float
    source_type: str  # "text", "url", "html"

class TextDetectResponse(BaseModel):
    probability_ai: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    methods: List[DetectionMethod]
    highlight_spans: List[HighlightSpan] = Field(default_factory=list)
    metadata: TextMetadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = "This is a probabilistic assessment, not definitive ground truth."
