from app.analyzers.base import Analyzer
from app.domain.errors import AnalyzerNotFoundError

_REGISTRY: dict[str, Analyzer] = {}
_DEFAULT_ID = "demo_rules"


def register(analyzer: Analyzer) -> None:
    _REGISTRY[analyzer.id] = analyzer


def resolve(analyzer_id: str | None = None) -> Analyzer:
    key = analyzer_id or _DEFAULT_ID
    try:
        return _REGISTRY[key]
    except KeyError as e:
        raise AnalyzerNotFoundError(key) from e


def get_default_analyzer_id() -> str:
    return _DEFAULT_ID


def set_default_analyzer_id(analyzer_id: str) -> None:
    global _DEFAULT_ID
    if analyzer_id not in _REGISTRY:
        raise AnalyzerNotFoundError(analyzer_id)
    _DEFAULT_ID = analyzer_id
