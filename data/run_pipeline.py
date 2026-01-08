import subprocess
import sys
# docker compose exec app python data/run_pipeline.py
def run(name, cmd):
    print(f"\n=== Startar {name} ===", flush=True)
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in p.stdout:
        print(f"[{name}] {line}", end="")

    p.wait()

    if p.returncode != 0:
        print(f"Fel i {name}. Avbryter.")
        raise SystemExit(p.returncode)

    print(f"=== Klar med {name} ===", flush=True)

def main():
    py = sys.executable
    run("retrieve", [py, "data/retrieve.py"])
    run("judge", [py, "data/judge_clean_rag.py"])
    run("summarize", [py, "data/summarize_judgments.py"])

if __name__ == "__main__":
    main()
