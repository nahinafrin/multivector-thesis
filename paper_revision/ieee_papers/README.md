# IEEE-format papers

| File | What it is | Source |
|---|---|---|
| `ICCIT2026_paper_IEEE.pdf` | 6-page ICCIT 2026 camera-ready paper (IEEE conference template, A4) | `../main_iccit.tex` |
| `ICCIT2026_paper_IEEE.docx` | Editable Word version of the same paper in IEEE A4 layout (two columns, Times New Roman 10 pt); Table VII is set in one column; it runs to 6 pages in Word | generated from `../main_iccit.tex` |
| `Extended_version_IEEE.pdf` | 17-page extended version with all tables and analyses, including the closed-form detector argument (IEEE conference template, US Letter) | `../main.tex` |

The PDFs are compiled with Tectonic (`tectonic -X compile main_iccit.tex`). The author block still reads
"Anonymous Authors" and must be filled in before submission.
