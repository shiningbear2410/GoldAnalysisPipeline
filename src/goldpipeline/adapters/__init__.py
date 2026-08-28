"""Input and provider adapters.

The domain depends on the protocols here, never on a vendor SDK. The real
Anthropic client is intentionally not imported at package level so the offline
test suite never needs the SDK loaded.
"""

from goldpipeline.adapters.base import AnalysisSource, LoadedSource, MarketDataSource
from goldpipeline.adapters.fake_finalizer import FakeFinalizerClient
from goldpipeline.adapters.fake_publisher import FakePublisherClient
from goldpipeline.adapters.fake_reviewer import FakeReviewerClient
from goldpipeline.adapters.fake_writer import FakeWriterClient
from goldpipeline.adapters.file_source import (
    JsonFileAnalysisSource,
    JsonFileMarketDataSource,
)
from goldpipeline.adapters.finalizer_client import (
    FinalizerClient,
    FinalizeRequest,
    FinalizeResponse,
)
from goldpipeline.adapters.publisher_client import (
    PublisherClient,
    SendOutcome,
    SendRequest,
)
from goldpipeline.adapters.reviewer_client import (
    ReviewerClient,
    ReviewRequest,
    ReviewResponse,
)
from goldpipeline.adapters.writer_client import WriterClient, WriterRequest, WriterResponse

__all__ = [
    "AnalysisSource",
    "FakeFinalizerClient",
    "FakePublisherClient",
    "FakeReviewerClient",
    "FakeWriterClient",
    "JsonFileAnalysisSource",
    "JsonFileMarketDataSource",
    "LoadedSource",
    "FinalizeRequest",
    "FinalizeResponse",
    "FinalizerClient",
    "MarketDataSource",
    "PublisherClient",
    "ReviewRequest",
    "ReviewResponse",
    "ReviewerClient",
    "SendOutcome",
    "SendRequest",
    "WriterClient",
    "WriterRequest",
    "WriterResponse",
]
