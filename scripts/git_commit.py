"""
Git repository initialization and commit script using dulwich.
"""

import os
from dulwich import porcelain

repo_path = "d:/Workspace/topographer"
git_dir = os.path.join(repo_path, ".git")

if not os.path.exists(git_dir):
    print("Initializing git repository in d:/Workspace/topographer...")
    repo = porcelain.init(repo_path)
else:
    repo = porcelain.open_repo(repo_path)

print("Staging repository files...")
porcelain.add(repo_path)

status = porcelain.status(repo_path)
staged_count = len(status.staged["add"]) + len(status.staged["modify"])
print(f"Staged {staged_count} files.")

commit_msg = b"feat: TopoGrapher v1.0.0 - Autonomous Neural Road Network & Topology Graph Extraction Engine"
author = b"TopoGrapher AI Team <team@topographer.ai>"

commit_id = porcelain.commit(repo_path, message=commit_msg, author=author)
print(f"Git commit created successfully! Commit SHA: {commit_id.decode('ascii')}")
