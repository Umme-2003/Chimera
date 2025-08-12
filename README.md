# 🤖 Chimera: An Autonomous AI Agent Swarm for Codebase Remediation

Project Chimera is a sophisticated, multi-agent AI system designed to autonomously analyze, refactor, and remediate vulnerabilities in public software repositories. This system demonstrates a full, end-to-end AI-powered software development lifecycle, from high-level goal analysis to the automatic creation of GitHub Pull Requests with validated code fixes.

---

##  Demo & Results

Here is a live example of Project Chimera in action. The system was tasked with finding and fixing a potential SQL injection vulnerability in the `mjhea0/flask-boilerplate` repository.

### Screenshot of the Application
*Replace this text with a screenshot of your Streamlit UI in action. Show the logs and the final diff view.*
`![Chimera UI](https://github.com/Umme-2003/Chimera/blob/main/chimera1.png)`
`![Chimera UI](https://github.com/Umme-2003/Chimera/blob/main/chimera2.png)`

---

## 🚀 Key Features

This project is built on a swarm of collaborating, specialized agents:

*   **Orchestrator Agent:** The central "brain" that takes a high-level user goal and manages the entire workflow.
*   **Discovery Agent:** Uses a deterministic keyword search to reliably identify potentially vulnerable files in a codebase, ensuring no stone is left unturned.
*   **Hunter Agent (RAG-Powered):** Analyzes specific code snippets using a **Retrieval-Augmented Generation (RAG)** model. It queries a vector database of vulnerability knowledge to identify flaws with high precision.
*   **Engineer Agent:** An LLM-powered agent that takes the Hunter's analysis and **autonomously writes the corrected, secure code**.
*   **Reviewer Agent:** Acts as an automated quality gate, ensuring any AI-generated code is syntactically valid before it proceeds.
*   **Git & GitHub Agents:** An automated toolchain that:
    *   Dynamically creates a **fork** of the target repository via the GitHub API.
    *   Creates a new branch for the proposed changes.
    *   Commits the validated fixes.
    *   Pushes the new branch to the user's fork.
    *   Opens a clean, professional **Pull Request** against the original repository.

---

## 🛠️ Tech Stack

*   **Core Logic:** Python
*   **AI & Orchestration:** LangChain, Groq API (for Llama 3)
*   **Embeddings & RAG:** Hugging Face Sentence Transformers, FAISS Vector Store
*   **UI:** Streamlit
*   **Version Control:** GitPython, PyGithub

---

## 🏃‍♀️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Umme-2003/Chimera.git
    cd [YOUR-REPO-NAME]
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure your API keys:**
    *   Create a `.env` file in the root directory.
    *   Add your API keys:
        ```env
        GROQ_API_KEY="gsk_..."
        GITHUB_TOKEN="ghp_..."
        ```

4.  **Launch the application:**
    ```bash

    streamlit run app.py
    ```
5.  Open your browser to the local Streamlit URL, enter your GitHub username and a target repository, and run the analysis!

---
