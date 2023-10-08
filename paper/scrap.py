import arxiv
import re

search_query = ("Reinforcement Learning AND (\"Market Making\" OR \"LOB\" OR \"limit order book\" OR \"order book\" OR "
                "\"algorithmic trading\")")

papers = arxiv.Search(query=search_query)

with open('papers.bib', 'w', encoding='utf-8') as bibtex:
    for paper in papers.results():
        title = re.sub(r"[^a-zA-Z0-9]+", '_', paper.title)
        authors = ' and '.join(author.name for author in paper.authors)
        year = paper.published.year
        doi = paper.doi

        bibtex_entry = f"@article{{{title}_{year},\n"
        bibtex_entry += f"  author = {{{authors}}},\n"
        bibtex_entry += f"  title = {{{paper.title}}},\n"
        bibtex_entry += f"  year = {{{year}}},\n"
        bibtex_entry += f"  doi = {{{doi}}},\n"
        bibtex_entry += f"  url = {{{paper.links[1].href}}}\n}}\n"

        bibtex.write(bibtex_entry)

print(f"Saved BibTeX entries to 'papers.bib' file.")
