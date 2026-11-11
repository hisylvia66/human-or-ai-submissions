# Group 11 final report

Final selection: pure DeBERTa-v3-small, fixed-validation AUROC 0.989423 and confirmed public AUROC 0.92433. The rank blend (0.92164) and robustness continuation (0.91085) were not selected. Private scores remain unknown.

Team: Ansh Raj Bhargava (ab3396), Yilin Gao (yg695), Yingzi Gu (yg667).

The report follows Q0–Q6 in the supplied template. Q2 and Q3 retain classical controlled experiments; Q4 evaluates the selected neural model. Appendices explain the distribution audit, neural method and failed continuation. No raw passages or competition data are included.

From the team directory, run `build_report.py`, `build_audit_report.py`, `build_neural_report.py`, and `build_last_refinement_report.py`, then compile `report/main.tex` with Tectonic and run `package_report.py`. The latter updates `report/report.pdf` and creates `artifacts/group11_overleaf.zip`. Upload that standalone ZIP to Overleaf, choose `main.tex` and pdfLaTeX. All sources and figures are included.

The course submission link is https://github.com/sekhari/human-or-ai-submissions/pull/13 and appears on the title page. `status.json` distinguishes final selection from the latest chronological submission. Confirmed scores are associated with recorded prediction hashes. Automated checks and visual PDF review do not assert completed human review of contributions or disclosures.
