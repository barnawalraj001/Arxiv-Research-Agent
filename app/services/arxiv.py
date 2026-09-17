class ArxivService:
    def get_paper(self, arxiv_id: str) -> PaperMetadata:
        ...

    def search_papers(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[PaperMetadata]:
        ...