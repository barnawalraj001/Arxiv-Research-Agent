import arxiv


def main():
    client = arxiv.Client()

    search = arxiv.Search(
        id_list=["2401.12345"],
        max_results=1,
    )

    results = list(client.results(search))

    if not results:
        print("Paper not found.")
        return

    paper = results[0]

    print("Paper found!")
    print()
    print("Title:", paper.title)
    print("Authors:", [author.name for author in paper.authors])
    print("arXiv ID:", paper.entry_id)
    print("Published:", paper.published)
    print("PDF:", paper.pdf_url)


if __name__ == "__main__":
    main()