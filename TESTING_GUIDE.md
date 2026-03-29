# 🧪 Project Chimera — Local Testing Guide

## Quick Start (5 minutes)

### Step 1: Start the Application

Open your terminal/command prompt and run:

```bash
cd C:\Users\Dell\Downloads\project-chimera-gitlab
streamlit run app.py
```

**The browser should open automatically at:** `http://localhost:8501`

If it doesn't open automatically, copy this link: **`http://localhost:8501`** and paste in your browser.

---

## 🎯 What You Should See

### Expected Behavior:
1. **Page loads** with title "Project Chimera"
2. **Left sidebar** shows:
   - GitHub Username input
   - Target Repository URL input
   - High-Level Goal input
   - "Run Chimera Analysis" button
3. **Main area** shows instructions

---

## 📋 Test Plan (30 minutes)

### TEST 1: Your Own Demo Repo (5 min)

**What to test:** The built-in vulnerable files

**Steps:**
1. In the sidebar, enter:
   - **GitHub Username:** `2003-umme` (or any username)
   - **Target Repository URL:** Copy this exact URL:
     ```
     https://github.com/sibtc/django-multiple-user-types-example.git
     ```
   - **High-Level Goal:** `Find hardcoded secrets`

2. Click **"Run Chimera Analysis"**

3. **Watch the magic happen** in the "Agent Activity Log"

**Expected Results:**
- ✅ "Cloning fresh repo..."
- ✅ "Found X potentially vulnerable files"
- ✅ Shows analysis of files
- ✅ "Test validation..."
- ✅ Final results displayed

---

### TEST 2: SQL Injection Detection (10 min)

**What to test:** The SQL injection vulnerability

**Steps:**
1. Use the **same repo URL**
2. Change the goal to: `Find SQL injection vulnerabilities`
3. Click **Run**

**Expected Results:**
- Should find SQL injection patterns
- Show the vulnerable SQL code
- Generate a fix using parameterized queries

---

### TEST 3: Flask App (15 min)

**What to test:** Flask framework detection

**Steps:**
1. Use this Flask repository:
   ```
   https://github.com/pallets/flask.git
   ```
2. Goal: `Find security vulnerabilities in the auth module`
3. Click **Run**

**Expected Results:**
- Should detect Flask framework
- Install Flask dependencies
- Run Flask tests
- Report findings

---

## 🔧 Troubleshooting

### Problem: "No API key found" Error
**Fix:** Your `.env` file should have:
```
GROQ_API_KEY=your_groq_api_key_here
```

**Make sure:**
1. File is named `.env` (not `.env.txt`)
2. File is in the project root folder
3. No spaces around the `=` sign

---

### Problem: "Module not found" Error
**Fix:** Run this in terminal:
```bash
cd C:\Users\Dell\Downloads\project-chimera-gitlab
pip install langchain langchain-groq langchain-google-genai gitpython streamlit
```

---

### Problem: Browser doesn't open
**Fix:**
1. Look for the URL in the terminal output
2. It will say: `Local URL: http://localhost:8501`
3. Copy that URL and paste in browser manually

---

## 📊 What to Monitor During Testing

### I Will Track:

While you test, I can monitor these files:
- `temp_repo/` - The cloned repository
- Console logs for errors
- Test results

### You Should Watch For:

✅ **GOOD Signs:**
- "Repository cloned successfully"
- "Found X potentially vulnerable files"
- "Test Runner Agent Initializing"
- "SUCCESS: X tests passed!"

❌ **BAD Signs:**
- "Test runner not implemented" (OLD VERSION!)
- "GITLAB_TOKEN not set" (Expected for local test)
- "ImportError" or "ModuleNotFoundError"

---

## 🎯 Success Criteria

Your test is **SUCCESSFUL** if:

- [ ] Streamlit app loads in browser
- [ ] You can enter repo URL and goal
- [ ] "Run" button works
- [ ] Agent Activity Log shows progress
- [ ] Files are scanned (shows "Analyzing: file.py")
- [ ] Test runner executes (shows "Test Runner Agent Initializing")
- [ ] No critical errors in log

**If you see "Test runner not implemented" - that's the OLD VERSION. Let me know and I'll fix it.**

---

## 📁 Test Repositories List

### Recommended (Safe to test):

1. **Django Tutorial** (Good for testing framework detection)
   ```
   https://github.com/sibtc/django-multiple-user-types-example.git
   ```

2. **Flask Mega-Tutorial** (Good for Flask testing)
   ```
   https://github.com/miguelgrinberg/microblog.git
   ```

3. **Python Testing Examples** (Good for test validation)
   ```
   https://github.com/pytest-dev/pytest.git
   ```

4. **Your Own Repo** (Best - test on your code)
   - Any public GitHub repo works

---

## 🔴 CRITICAL: Testing Checklist

Before you declare "It works!", verify:

### MUST WORK:
- [ ] App starts with `streamlit run app.py`
- [ ] Sidebar inputs are responsive
- [ ] "Run" button triggers analysis
- [ ] Shows "Found X vulnerable files"
- [ ] Shows framework detection (Django/Flask/Python)
- [ ] Shows test runner executing
- [ ] Shows test results

### Bonus (Nice to have):
- [ ] Creates virtual environment
- [ ] Installs dependencies
- [ ] Runs actual tests
- [ ] Shows side-by-side code comparison

---

## 📸 While Testing - SCREENSHOT THESE:

**For your demo video, capture:**

1. **The web UI** with the sidebar
2. **"Found X vulnerable files"** message
3. **Test runner running** (shows setup)
4. **Test results** (SUCCESS message)
5. **Side-by-side comparison** (original vs fixed code)

---

## 🆘 Emergency Contacts (Me!)

**If you get stuck:**

1. Check the terminal - what's the error?
2. Screenshot the error
3. Tell me exactly what you clicked
4. I'll debug it with you

---

## ✅ After Testing - What's Next

Once you confirm it works:

1. **Record a demo video** (while it's fresh)
2. **Screenshot the working UI**
3. **Document any bugs** found
4. **Celebrate** 🎉 (you now have a working hackathon project!)

---

## 🚀 Ready to Start?

**Copy this command and paste in your terminal:**

```
cd C:\Users\Dell\Downloads\project-chimera-gitlab && streamlit run app.py
```

**Then open:** http://localhost:8501

**Test with:** https://github.com/sibtc/django-multiple-user-types-example.git

**Goal:** Find hardcoded secrets

---

**GO TEST IT NOW!**

Let me know:
- ✅ "It works!" (what did you see?)
- ❌ "Got an error" (what error message?)
- 🔄 "Something weird happened" (describe it)

I'm monitoring this conversation and will help you debug in real-time! 🎯