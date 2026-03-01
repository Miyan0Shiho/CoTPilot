# Plan: Codebase Cleanup and GitHub Push

## 1. Objective
Prepare the CoT-Pilot codebase for public release on GitHub by removing temporary artifacts, sensitive data, and redundant files, then push the clean state to the remote repository.

## 2. Detailed Task Breakdown

### Task 1: Cleanup (Housekeeping)
- **Goal**: Remove large/temporary files generated during experiments to keep the repo light and clean.
- **Actions**:
    - **Delete Workspace Artifacts**: Remove `workspace/` (except for perhaps a `README.md` or `.gitkeep` to indicate structure) as it contains gigabytes of logs and predictions.
    - **Remove Temporary Scripts**: Delete `step1_pop.txt`, `dev_result.txt` etc. from root or subdirs if they leaked.
    - **Clean `__pycache__`**: Standard Python cleanup.
    - **Update `.gitignore`**: Ensure `workspace/`, `*.log`, `__pycache__/`, `.DS_Store` are ignored.

### Task 2: Documentation Polish
- **Goal**: Ensure the repo looks professional.
- **Actions**:
    - **Update `README.md`**: Add "Quick Start" using the new `reproduce/` scripts. Mention the scientific workflow.
    - **Add `requirements.txt`**: Ensure all dependencies (OpenCompass, EvoPrompt adapters) are listed.

### Task 3: Git Operations
- **Goal**: Push to GitHub.
- **Actions**:
    - Initialize git (if not already).
    - Add remote (User needs to provide URL or I'll assume one/ask).
    - Commit all changes.
    - Push to `main`.

## 3. Execution Steps

1.  **Analyze current file structure** (`ls -R`) to identify junk.
2.  **Execute cleanup commands** (`rm -rf ...`).
3.  **Update `.gitignore`** to prevent future pollution.
4.  **Ask user for GitHub Repo URL** (or instructions to create one).
5.  **Commit and Push**.
