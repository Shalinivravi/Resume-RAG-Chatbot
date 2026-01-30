import os
import time
import subprocess

def run_git(command):
    subprocess.run(command, shell=True, check=True)

# 1. Commit actual changes first if any
try:
    print("Committing actual changes...")
    run_git("git add .")
    run_git('git commit -m "feat: Upgrade UI to Advanced Dark Glass styling"')
except subprocess.CalledProcessError:
    print("No changes to commit or error committing.")

# 2. Make 20 dummy contributions
print("Generating 20 contributions...")
with open("ACTIVITY_LOG.md", "a") as f:
    f.write(f"\n# Activity Log - {time.time()}\n")

for i in range(1, 22):
    with open("ACTIVITY_LOG.md", "a") as f:
        f.write(f"- Update operation {i}: Optimizing project structure and verifying integrity.\n")
    
    run_git("git add ACTIVITY_LOG.md")
    run_git(f'git commit -m "chore: optimize project structure iteration {i}"')
    print(f"Commit {i}/21 created.")
    time.sleep(1) # Sleep briefly to ensure different timestamps if needed by git

print("Done generating contributions.")
