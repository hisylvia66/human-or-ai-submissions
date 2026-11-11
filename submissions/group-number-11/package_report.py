"""Package a compiled, standalone Overleaf project without competition data."""
from pathlib import Path
import shutil
import zipfile

TEAM=Path(__file__).resolve().parent
report=TEAM/'report'
built=TEAM/'artifacts/report/main.pdf'
if not built.exists():
    raise SystemExit('Compile report/main.tex to artifacts/report/main.pdf first.')
shutil.copy2(built,report/'report.pdf')
target=TEAM/'artifacts/group11_overleaf.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ('main.tex','metrics.tex','audit_appendix.tex','neural_appendix.tex','neural_method.tex','neural_q4.tex','last_refinement_appendix.tex','status.json','README.md'):
        z.write(report/name,name)
    for directory in ('tables','figures'):
        for file in sorted((report/directory).iterdir()):
            z.write(file,file.relative_to(report))
print(f'PDF: {report / "report.pdf"}')
print(f'Overleaf ZIP: {target}')
