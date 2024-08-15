"""Llama Dataset Class."""

import asyncio
import time
from typing import List, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llama_dataset.base import (
    BaseLlamaDataExample,
    BaseLlamaDataset,
    BaseLlamaExamplePrediction,
    BaseLlamaPredictionDataset,
    CreatedBy,
)
from pandas import DataFrame as PandasDataFrame

class RagExamplePrediction(BaseLlamaExamplePrediction):
    """RAG example prediction class.

    Args:
        response (str): The response generated by the LLM.
        contexts (Optional[List[str]]): The retrieved context (text) for generating
                                        response.
    """

    response: str = Field(
        default_factory=str,
        description="The generated (predicted) response that can be compared to a reference (ground-truth) answer.",
    )
    contexts: Optional[List[str]] = Field(
        default_factory=None,
        description="The contexts in raw text form used to generate the response.",
    )

    @property
    def class_name(self) -> str:
        """Data example class name."""
        return "RagExamplePrediction"
    
    

from pydantic import Field
from typing import Optional, List, Dict, Any

class LabelledRagDataExample(BaseLlamaDataExample):
    """RAG example class with metadata support."""

    query: str = Field(
        default_factory=str,
        description="The user query for the example."
    )
    query_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the query."
    )
    reference_contexts: Optional[List[str]] = Field(
        default_factory=list,
        description="The contexts used to generate the reference answer."
    )
    reference_answer: str = Field(
        default_factory=str,
        description="The reference (ground-truth) answer to the example."
    )
    reference_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the reference answer."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata associated with the example."
    )

    def class_name(self) -> str:
        """Returns the class name of the example."""
        return "LabelledRagDataExample"


class RagPredictionDataset(BaseLlamaPredictionDataset):
    """RagDataset class."""

    _prediction_type = RagExamplePrediction

    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""
        data = {}
        if self.predictions:
            data = {
                "response": [t.response for t in self.predictions],
                "contexts": [t.contexts for t in self.predictions],
            }

        return PandasDataFrame(data)

    @property
    def class_name(self) -> str:
        """Class name."""
        return "RagPredictionDataset"
    
    
class LabelledRagDataset(BaseLlamaDataset[BaseQueryEngine]):
    """RagDataset class with metadata support."""

    _example_type = LabelledRagDataExample

    def to_pandas(self) -> PandasDataFrame:
        """Create pandas dataframe."""
        data = {
            "query": [t.query for t in self.examples],
            "reference_contexts": [t.reference_contexts for t in self.examples],
            "reference_answer": [t.reference_answer for t in self.examples],
            "reference_answer_by": [str(t.reference_answer_by) for t in self.examples],
            "query_by": [str(t.query_by) for t in self.examples],
            "metadata": [t.metadata for t in self.examples],  # Include metadata
        }

        return PandasDataFrame(data)
    
    async def _apredict_example(
        self,
        predictor: BaseQueryEngine,
        example: LabelledRagDataExample,
        sleep_time_in_seconds: int,
    ) -> RagExamplePrediction:
        """Async predict RAG example with a query engine."""
        await asyncio.sleep(sleep_time_in_seconds)
        response = await predictor.aquery(example.query)
        return RagExamplePrediction(
            response=str(response), contexts=[s.text for s in response.source_nodes]
        )

    def _predict_example(
        self,
        predictor: BaseQueryEngine,
        example: LabelledRagDataExample,
        sleep_time_in_seconds: int = 0,
    ) -> RagExamplePrediction:
        """Predict RAG example with a query engine."""
        time.sleep(sleep_time_in_seconds)
        response = predictor.query(example.query)
        return RagExamplePrediction(
            response=str(response), contexts=[s.text for s in response.source_nodes]
        )

    def _construct_prediction_dataset(
        self, predictions: List[RagExamplePrediction]
    ) -> RagPredictionDataset:
        """Construct prediction dataset."""
        return RagPredictionDataset(predictions=predictions)

    @property
    def class_name(self) -> str:
        """Class name."""
        return "LabelledRagDataset"
