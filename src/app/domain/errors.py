class AnalyzerNotFoundError(LookupError):
    def __init__(self, analyzer_id: str) -> None:
        self.analyzer_id = analyzer_id
        super().__init__(f"Analyzer not found: {analyzer_id}")


class AnalyzerExecutionError(RuntimeError):
    def __init__(self, analyzer_id: str, message: str) -> None:
        self.analyzer_id = analyzer_id
        super().__init__(message)
