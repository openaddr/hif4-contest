"""Run bench.py against the v8 reference file (only full variant)."""
import re

src = open("dev/bench.py", encoding="utf-8").read()
old_sol = 'SOL_PATH = os.path.join(os.path.dirname(__file__), "..", "example", "solution", "solution.py")'
new_sol = 'SOL_PATH = r"C:/Users/ning/Desktop/wt/dev/_sol_v8_ref.py"'
assert old_sol in src
src = src.replace(old_sol, new_sol)
src = re.sub(r"VARIANTS = \{.*?\n\}", 'VARIANTS = {"v3_full": []}', src, flags=re.S)
src = src.replace('REF_PATH = r"C:\\Users\\ning\\Downloads\\solution\\solution.py"', "REF_PATH = SOL_PATH")
g = {"__name__": "bench_v8", "__file__": "C:/Users/ning/Desktop/wt/dev/bench_v8_ab.py"}
exec(compile(src, "bench_v8", "exec"), g)
g["main"]()
