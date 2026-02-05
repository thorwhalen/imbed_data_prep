# Jersey Laws

Data preparation for Jersey (Channel Islands) legislation.

## Data source

[Jersey Law](https://www.jerseylaw.je/) is the official online library of
current legislation for the Bailiwick of Jersey, maintained by the Jersey
Legal Information Board (JLIB).  It provides consolidated versions of laws,
court judgments, and legal texts as part of the Free Access to Law Movement.

Jersey's legislation includes Laws (the highest form passed by the States
Assembly), Regulations (subsidiary legislation), and Orders (made by
Ministers under statutory powers).

The module scrapes metadata and PDF URLs from HTML pages downloaded from
`https://www.jerseylaw.je/laws/current/`.

**Note:** The HTML pages must be downloaded manually from the website before
running this module.

## What it does

1. **Parse HTML** -- extract law names, reference codes, and URLs from the
   downloaded HTML files using BeautifulSoup.
2. **Gather PDF URLs** -- resolve links to the PDF versions of each law.
3. **Build metadata list** -- produce a structured list of law records.

## Output

A list of dicts, each with:

| Field | Description |
|---|---|
| `name` | Name of the law (e.g. "Bankruptcy (Désastre) (Jersey) Law 1990") |
| `url` | URL to the law's page on jerseylaw.je |
| `ref` | Reference code |
| `pdf_url` | Direct URL to the PDF version |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `jersey_laws.ipynb` | Notebook exploring the Jersey laws data |
