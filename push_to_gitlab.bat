@echo off
REM Push Project Chimera to GitLab - Manual commands
echo Pushing Project Chimera to GitLab...
echo.

cd "C:\Users\Dell\Downloads\project-chimera-gitlab"

echo Step 1: Adding core files...
git add chimera_core.py chimera_gitlab.py chimera.py main.py requirements.txt .gitignore .env.example knowledge_base.txt .gitlab-ci.yml
git add .gitlab/

echo.
echo Step 2: Committing...
git commit -m "feat: Project Chimera - 4-agent security remediation with self-healing validation"

echo.
echo Step 3: Pushing with force (to overwrite initial repo content)...
git push -u origin main --force

echo.
echo Done! Check your GitLab repo at:
echo https://gitlab.com/gitlab-ai-hackathon/participants/34972906
pause
