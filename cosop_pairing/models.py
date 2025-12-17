from __future__ import annotations

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    content: str = Field(..., json_schema_extra={"example": "This the content of the sentence."})
    document_id: str = Field(..., json_schema_extra={"example": "The ID of the document."})
    page_number: int = Field(..., json_schema_extra={"example": 1})
    sentence_index: int = Field(..., json_schema_extra={"example": 42})


class SentencePair(BaseModel):
    sentence_1: Sentence
    sentence_2: Sentence


class SentencePairs(BaseModel):
    sentence_pairs: list[SentencePair]

