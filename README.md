# FleetCareAI — Intelligent Fleet Maintenance Decision Support System

**Course:** Intelligent Reasoning Systems (IRS)  
**Project Name:** FleetCareAI  
**Group:** Group 47 
**Institution:** National University of Singapore — Institute of Systems Science (NUS-ISS)

---

## SECTION 1: PROJECT TITLE

# FleetCareAI — Intelligent Fleet Maintenance Decision Support System

**Predict Risks. Explain Decisions. Support Maintenance.**

FleetCareAI is an AI-powered fleet maintenance decision support platform designed to help fleet managers and maintenance staff identify high-risk vehicles, understand predictive maintenance recommendations, and access relevant maintenance knowledge.

The system integrates machine learning, temporal risk analysis, rule-based reasoning, explainable AI, and retrieval-augmented generation (RAG) into a unified web application.

**Core Features**

- Fleet Dashboard and Risk Monitoring
- Vehicle Risk Analysis and Prediction History
- Explainable Maintenance Recommendations
- Maintenance Plan Summary
- AI Assistant for Vehicle Analysis and Maintenance Knowledge

## 🎬 Product Demonstration

FleetCare AI is an intelligent predictive maintenance
decision-support system designed to help fleet managers
identify high-risk vehicles and prioritise inspections.

[![Watch FleetCare AI Demo](https://github.com/user-attachments/assets/9f09e543-c782-4502-93de-eb386263cd94)](YOUR_VIDEO_URL)

Click the image to watch our 5-minute product demonstration.

---

## SECTION 2: EXECUTIVE SUMMARY

Commercial fleet maintenance is essential for ensuring vehicle reliability, operational efficiency, and road safety. Traditional maintenance processes often depend on scheduled inspections, manual monitoring, and fragmented maintenance records. These approaches may make it difficult for fleet operators to identify emerging risks, prioritise maintenance activities, and interpret complex vehicle data efficiently.

FleetCareAI aims to address these challenges by providing an intelligent fleet maintenance decision support system that combines predictive analytics with explainable reasoning and AI-assisted knowledge retrieval.

The system uses machine learning models to assess vehicle conditions and generate risk scores across multiple prediction horizons. A temporal reasoning component examines historical predictions, risk persistence, and deterioration indicators to provide additional context for vehicle risk assessments.

A rule-based reasoning engine then translates these assessments into maintenance priorities and recommended actions. Each decision includes supporting evidence and explanations, helping users understand why a particular vehicle requires inspection, enhanced monitoring, or continued operation.

To improve interpretability, the system incorporates SHAP-based explanations that identify important feature contributions to model predictions. These explanations support model transparency but do not constitute confirmed mechanical fault diagnoses.

FleetCareAI also provides an interactive web-based Dashboard. Fleet managers and maintenance staff can monitor fleet-wide risk distribution, identify high-priority vehicles, review maintenance capacity, and organise inspection activities based on recommended actions.

In addition, an AI Assistant powered by Qwen supports natural-language interactions. Through Function Calling, the assistant can retrieve vehicle-specific predictions and analysis results from PostgreSQL. It also integrates retrieval-augmented generation using multilingual embeddings and pgvector to retrieve relevant maintenance guidance, project knowledge, and Singapore transport policy references.

The system is developed using React, FastAPI, SQLAlchemy, PostgreSQL, pgvector, and Qwen. Its modular architecture separates the presentation layer, business logic, AI orchestration, and data access, supporting maintainability and future extension.

Overall, FleetCareAI demonstrates how machine learning, rule-based reasoning, explainable AI, and knowledge-grounded assistance can be integrated to support more informed fleet maintenance decisions.

---

## SECTION 3: PROJECT CONTRIBUTION

| Official Full Name | Student ID | Project Contribution |
|---|---|---|
| Zhang Tianqin | A0356380B | Developed the machine-learning pipeline, including temporal feature engineering, imbalance handling, model experimentation and tuning, ensemble design, rule-engine integration, SHAP explainability, and production inference. Designed the database architecture and table schemas for vehicle data, predictions, explanations, and knowledge retrieval. Built the initial RAG prototype and vehicle data-retrieval tool functions, and contributed to frontend and backend improvements. As project leader, managed overall product functionality, system integration, UI/UX, task allocation, team coordination, meetings, and the project timeline.|
| Zhang Yuyue | A0305487X | [Confirm specific contributions] |
| Ma Jialing | A0305478X | Designed and established the React frontend and FastAPI backend architecture, adopting a modular API–Service–Repository structure. Developed React frontend features and FastAPI backend APIs for the Fleet Dashboard and AI assistant. Designed PostgreSQL database integration and implemented fleet risk aggregation, maintenance statistics, and vehicle data retrieval using SQLAlchemy. Integrated Qwen 3.5 Flash with Function Calling and pgvector-based RAG to support vehicle risk explanations and maintenance knowledge queries. Supported frontend–backend integration, API testing, debugging, and system demonstrations. |
| Gong Yucheng | A0355325E | Contributed to the initial project idea, business use cases, and refinement of the FleetCareAI solution. Co-developed the predictive maintenance models through data exploration, temporal feature engineering, model implementation, comparative experiments, tuning, and evaluation. Built the RAG knowledge base using official maintenance guidance, Singapore transport policy references, and project documentation; implemented document processing, multilingual embeddings, hybrid retrieval, and source citations, and migrated knowledge storage to PostgreSQL with pgvector. Prepared deployment instructions and interface documentation to support integration by the web team.

**Project Workstreams**

The project includes the following major workstreams:

1. Machine Learning Model Development and Evaluation
2. Temporal Risk Analysis and Rule-Based Reasoning
3. Explainable AI and SHAP Integration
4. Backend API and Database Development
5. Frontend Dashboard and Vehicle Analysis
6. AI Assistant, Function Calling, and RAG Integration
7. System Integration, Testing, Documentation, and Demonstration

The final contribution table should specify each member's actual responsibilities and deliverables.


## SECTION 4: VIDEO OF SYSTEM MODELLING & USE CASE DEMO

FleetCareAI includes two video presentations covering the product experience and the technical implementation.

### 4.1 Product Demonstration Video

**Video Link:** [Insert YouTube / Google Drive URL]

The product demonstration presents FleetCare AI from the perspective of fleet managers and maintenance staff.

It demonstrates:

- Fleet-wide monitoring through the dashboard
- Vehicle risk distribution and top-risk alerts
- Maintenance capacity and inspection planning
- Vehicle-specific risk assessments and explanations
- AI-assisted retrieval of vehicle assessments, maintenance guidance, and policy knowledge

### 4.2 Technical System Design Video

**Video Link:** [Insert YouTube / Google Drive URL]

The technical demonstration explains the system architecture and its underlying prediction and reasoning mechanisms.

Topics include:

- High-level system architecture
- Predictive model implementation and evaluation
- Temporal feature engineering and risk analysis
- Rule-based maintenance decision engine
- SHAP-based model explainability
- FastAPI and PostgreSQL integration
- AI assistant with Qwen function calling
- RAG knowledge retrieval using pgvector
- Dashboard and assistant execution sequences


## SECTION 5: USER GUIDE

### 5.1 System Requirements

**Recommended Development Environment**

- Python 3.12
- Node.js and npm
- PostgreSQL 16
- pgvector extension compatible with the installed PostgreSQL version
- Conda or another Python virtual environment manager
- Qwen-compatible API credentials
- Internet access for remote LLM requests

### 5.2 Project Architecture

FleetCareAI follows a modular architecture:

```text
React Frontend
    |
    v
FastAPI Backend
    |
    +-- Dashboard Services
    |       |
    |       +-- SQLAlchemy Repositories
    |               |
    |               +-- PostgreSQL
    |
    +-- AI Assistant
            |
            +-- Qwen LLM
            |
            +-- Function Calling
                    |
                    +-- VehicleAnalysisTool
                    |       |
                    |       +-- PostgreSQL
                    |           (Vehicle Analysis + SHAP)
                    |
                    +-- KnowledgeSearchTool
                            |
                            +-- Multilingual Embedding
                            |
                            +-- pgvector
                                (Maintenance Knowledge)
```

### 5.3 Clone the Repository

```bash
git clone https://github.com/TianQinNus/AIS_Project.git
cd AIS_Project
```

### 5.4 Backend Setup

Navigate to the backend directory:

```bash
cd SystemCode/backend
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fleetcare
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

### 5.5 PostgreSQL Database Setup

Install and start PostgreSQL 16 on macOS using Homebrew:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Create the database:

```bash
createdb fleetcare
```

The project requires the PostgreSQL `vector` extension for RAG knowledge retrieval.

After installing a pgvector build compatible with PostgreSQL 16, enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Business Database**

The structured database contains vehicle records, model predictions, and SHAP explanations.

Core tables include:

- `truck`
- `truck_readout`
- `analysis`
- `prediction_shap`

Import the supplied PostgreSQL dump, if available:

```bash
psql -X -v ON_ERROR_STOP=1 \
  -d fleetcare \
  -f /path/to/fleetcare.sql
```

**Important:** The complete business database dump may not be included in GitHub because of file-size restrictions. Obtain the dataset through the project team's designated distribution channel.

Importing a complete SQL dump into an existing database may produce table conflicts. Follow the database reset instructions before restoring a fresh database.

**RAG Knowledge Database**

The knowledge retrieval component uses PostgreSQL and pgvector.

Core knowledge tables include:

- `knowledge_document`
- `knowledge_chunk`

The knowledge chunks contain source text, metadata, citations, and 384-dimensional embeddings.

If the corresponding setup scripts and seed files are available:

```bash
psql -X -v ON_ERROR_STOP=1 \
  -d fleetcare \
  -f scripts/create_knowledge_tables.sql
```

```bash
python -m scripts.import_pgvector_seed \
  --file data/pgvector_seed.json.gz
```

Verify the imported knowledge:

```sql
SELECT COUNT(*)
FROM public.knowledge_chunk;
```

The knowledge database must be populated before maintenance knowledge retrieval can function.

### 5.6 Environment Configuration

Create a `.env` file inside `SystemCode/backend`.

Example:

```dotenv
DATABASE_URL=postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/fleetcare

DASHSCOPE_API_KEY=YOUR_QWEN_API_KEY
QWEN_MODEL=qwen3.5-flash
QWEN_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1

CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

The environment variable names above should be checked against the final backend configuration before submission.

Do not commit real API keys, passwords, or `.env` files to GitHub.

### 5.7 Start the Backend

From `SystemCode/backend`:

```bash
uvicorn app.main:app --reload
```

Default local server:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5.8 Start the Frontend

Open a new terminal:

```bash
cd SystemCode/fleetcare-frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically:

```text
http://localhost:5173
```

### 5.9 Using FleetCareAI

**Fleet Dashboard**

The Dashboard provides fleet-level information, including:

- Active vehicles
- Vehicles at risk
- Average fleet risk score
- Maintenance capacity
- Risk distribution
- Top risk alerts
- Maintenance plan summary

**Vehicle Analysis**

Staff can inspect individual vehicle assessments, including prediction results, risk indicators, maintenance priorities, recommended actions, and SHAP explanations.

**AI Assistant**

The AI Assistant supports two primary question types.

*Vehicle Analysis Example:*

> How has vehicle 18734's risk changed recently?

The assistant retrieves stored vehicle analysis data and explains relevant risk indicators, prediction history, and recommended actions.

*Maintenance Knowledge Example:*

> What should I check before driving a heavy goods vehicle?

The assistant retrieves relevant maintenance reference material and provides supporting sources where available.

The assistant is intended for decision support. It does not replace professional vehicle inspection, qualified mechanical diagnosis, or official regulatory advice.

---

## SECTION 6: PROJECT REPORT / PAPER

The full project report is available in the GitHub repository:

**Folder:** `ProjectReport/`

**Report Link:** [Insert final report URL]

### Report Contents

1. Executive Summary
2. Business Problem Background
3. Project Objectives and Scope
4. Literature Review and Related Work
5. System Architecture and Design
6. Machine Learning Prediction Models
7. Temporal Risk Analysis
8. Rule-Based Maintenance Reasoning
9. Explainable AI Using SHAP
10. AI Assistant and Retrieval-Augmented Generation
11. Frontend and Backend Implementation
12. Experiments, Testing, and Evaluation
13. Results and Discussion
14. System Limitations
15. Conclusions and Future Improvements

**Appendices**

- Project Proposal
- Installation and User Guide
- System Architecture Diagrams
- API Documentation
- Evaluation Results
- Individual Project Contributions and Reflections
- References

### Evaluation and Validation

FleetCareAI includes evaluation activities covering model prediction, reasoning, knowledge retrieval, and system integration.

The RAG component has undergone an initial ten-question evaluation covering maintenance knowledge, Singapore transport policy, project-specific explanations, and out-of-scope questions.

In the documented retrieval experiment, the eight answerable questions achieved:

- Macro Precision@5: 0.2000
- Macro Recall@5: 0.78125

These are preliminary retrieval metrics based on a small, initially annotated test set. They are not measures of overall system accuracy or generated-answer correctness.

The reported experiment did not complete real LLM answer generation and human scoring. Further end-to-end validation is required.

Refer to the full evaluation report for experimental methodology, findings, and limitations.

---

## SECTION 7: MISCELLANEOUS

### 7.1 Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| ORM | SQLAlchemy |
| Database | PostgreSQL |
| Vector Database Capability | pgvector extension |
| AI Assistant | Qwen 3.5 Flash |
| Agent Integration | Function Calling |
| Embedding Model | Multilingual MiniLM |
| Explainability | SHAP |
| Reasoning | Temporal Analysis and Rule-Based Reasoning |
| API Documentation | FastAPI Swagger UI |
| Version Control | Git and GitHub |

### 7.2 Knowledge Sources

The RAG knowledge base includes selected reference materials from:

- UK Driver and Vehicle Standards Agency (DVSA)
- UK Health and Safety Executive (HSE)
- Singapore Land Transport Authority (LTA)
- FleetCareAI Project Rule Documentation

The documents support general maintenance guidance, regulatory reference, and explanations of project-specific reasoning concepts.

UK maintenance guidance is not presented as Singapore law. Policy applicability must be checked against the relevant vehicle type, registration details, and current official requirements.

### 7.3 Key System Design Principles

**Modular Architecture**

The system separates API routing, business services, repositories, and AI tools to support maintenance and extensibility.

**Explainable Reasoning**

Maintenance decisions include rule identifiers, evidence, and explanations. SHAP provides additional insight into model predictions.

**Grounded AI Assistance**

Vehicle-specific information is retrieved from structured database records, while maintenance knowledge is retrieved from relevant reference documents.

**Human-in-the-Loop Decision Support**

The system provides maintenance recommendations and supporting information. Final inspection and repair decisions remain the responsibility of qualified personnel.

### 7.4 Limitations and Future Work

Current limitations include the dependence on available vehicle prediction records, the scope of the reference knowledge base, and the need for further end-to-end evaluation of AI-generated responses.

Potential future enhancements include:

- Expanded maintenance knowledge coverage
- Improved retrieval relevance and evaluation
- Multi-turn conversational memory
- Role-based access control
- Maintenance workflow and work-order integration
- Production deployment and monitoring
- Additional model validation with operational fleet data

---

## Acknowledgements

This project was developed as part of the Intelligent Reasoning Systems coursework at the National University of Singapore, Institute of Systems Science.

We thank the teaching team for their guidance and feedback throughout the project.

---

**FleetCareAI — From Predictive Insights to Explainable Maintenance Decisions.**
