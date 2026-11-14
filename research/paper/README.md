# Research paper (IEEE conference format)

`main.tex` is a complete draft of the paper in the official IEEE template (`IEEEtran`).
All numbers in it come from notebooks 03–07. All references were checked for authors, venue and DOI.

## Make the PDF (5 minutes, no installation)

1. Go to https://www.overleaf.com and sign in (free account).
2. **New Project → Upload Project** and choose `paper_overleaf.zip` from this folder.
3. Menu → **Compiler: pdfLaTeX**, then click **Recompile**. Overleaf already includes `IEEEtran`.
4. **Download PDF**, and you have your IEEE paper.

To rebuild the zip after editing locally: zip `main.tex` and the `figures/` folder together.

## Before you submit: your team's part

- [ ] **Authors:** replace "Author One/Two/Three", department, city and emails in the `\author{...}` block.
- [ ] **Read the cited papers.** At least the Pakistan-specific ones: del Río 2013, Iqbal 2016, Hussain 2024, Mishra 2020. You must be able to explain why each is cited.
- [ ] **Extend Related Work (Section II)** with 3–5 more papers you find yourselves, e.g. on ML climate forecasting in South Asia. Add them to `thebibliography` in the same IEEE style.
- [ ] **Rewrite in your own words** anything you cannot explain in the viva. Plagiarism checkers flag text, not ideas.
- [ ] **Code and Data Availability:** replace the two `[link to be added]` placeholders with your GitHub and dataset links.
- [ ] Check the page limit your teacher expects. To shorten, the easiest cuts are Fig. 1 (pipeline) and Table II.
