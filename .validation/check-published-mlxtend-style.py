import json
import os
from pathlib import Path
import subprocess

os.chdir("style-project")
results = Path(".style-results").resolve()
results.mkdir(exist_ok=True)
versions = [("base", "1aee767ef57b84a8191511a08955c583b65de382"), ("1306", "9cd05e45234e7e904c507b81c9309c0496ec1adb"), ("1307", "b4baa76d7f4c95e2999770c224bcdbde49005269"), ("1310", "00039441e569753ed772a18fe5973bf606406ad6"), ("1311", "a08bba3825b275cc1e0d84e95f57087c88626b5d")]
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", *[sha for _, sha in versions]], check=True)
summary=[]
uv=["uv","run","--frozen","--group","dev","python"]
for name,sha in versions:
    subprocess.run(["git","checkout","--detach",sha],check=True)
    paths=list(map(str,sorted(Path("mlxtend").glob("*"))))
    commands={
      "isort":uv+["-m","isort","-p","mlxtend","--check","--diff","--line-length","88","--multi-line","3","--py","39","--profile","black",*paths],
      "black":uv+["-m","black","--check","--diff",*paths],
      "flake8-report":uv+["-m","flake8",".","--config=.flake8","--count","--exit-zero","--statistics"]}
    record={"name":name,"head":sha,"checks":{}}
    for kind,command in commands.items():
        p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (results/(name+"-"+kind+".log")).write_text(p.stdout)
        record["checks"][kind]={"exit_code":p.returncode,"output":p.stdout}
        print(name,kind,p.returncode,p.stdout[-6000:],flush=True)
    summary.append(record)
(results/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("PUBLISHED_STYLE_RESULTS="+json.dumps(summary),flush=True)
assert all(r["checks"][kind]["exit_code"]==0 for r in summary for kind in ["isort","black"]), summary
