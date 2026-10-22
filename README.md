# CollegeRank AI — ML-Based College Prediction System

CollegeRank AI is a machine-learning-based college prediction application that compares student entrance-exam ranks with historical admission cutoffs and provides estimated college options.

## Features

* React frontend with a responsive dashboard.
* FastAPI backend with REST API endpoints.
* Machine-learning prediction support for JEE Main and JEE Advanced.
* Historical and estimated closing-rank comparisons.
* College, branch, cutoff trend and explanation cards.
* Input validation and backend connection error handling.
* Clear warnings that predictions are estimates, not admission guarantees.

## Technology Stack

**Frontend:** React, Vite, JavaScript, CSS
**Backend:** Python, FastAPI, Pydantic
**Machine Learning:** scikit-learn, XGBoost, joblib, pandas, NumPy
**Data:** Historical counselling and opening/closing-rank records

## Project Structure

```text
CollegeRank-AI/
├── backend/
│   └── app/
│       ├── api/
│       ├── ml/
│       ├── schemas/
│       ├── services/
│       └── main.py
├── data/
│   └── processed/
├── frontend/
│   ├── src/
│   ├── dist/
│   ├── package.json
│   └── index.html
├── models/
├── notebooks/
├── scripts/
├── docs/
├── app.py
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.12 or a compatible Python environment for the installed dependencies.
* Node.js and npm.
* Project Python dependencies installed from `requirements.txt`.
* The historical datasets and trained model files required by the prediction service.

## Installation

Run these commands from the project root in Windows Command Prompt.

### 1. Activate the Python environment

```bat
.venv\Scripts\activate
```

### 2. Install Python dependencies

```bat
python -m pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bat
cd frontend
npm install
cd ..
```

## Run the Application

Keep both services running in separate terminals.

### Terminal 1 — Backend

From the project root:

```bat
.venv\Scripts\activate
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Backend API: http://127.0.0.1:8000
Interactive API documentation: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/health

### Terminal 2 — Frontend

From the project root:

```bat
cd frontend
npm run dev
```

Open the local URL printed by Vite, usually http://localhost:5173.

## API Endpoints

| Method | Endpoint       | Purpose                              |
| ------ | -------------- | ------------------------------------ |
| GET    | `/health`      | Check backend health                 |
| GET    | `/api/exams`   | List supported exam names            |
| POST   | `/api/predict` | Request college prediction estimates |

## Current Scope and Limitations

* JEE Main predictions use the collected CSE-related NIT, IIIT and GFTI dataset.
* JEE Advanced predictions use the available JoSAA IIT dataset and its implemented filtering rules.
* Other listed entrance examinations are not yet implemented in the prediction service.
* State-based quota matching is not currently fully implemented.
* Dataset coverage, category rules, gender pools and counselling rounds limit the available results.
* Model estimates can differ from actual future closing ranks.
* A rank comparison is not an admission probability, eligibility determination or guarantee.

Always verify current cutoffs, eligibility and seat-allocation rules with the relevant official counselling authority.

## Production Frontend Build

From the frontend directory:

```bat
npm run build
```

Vite generates production files in `frontend/dist`. A successful build confirms frontend compilation; it does not itself deploy the application or verify prediction accuracy.

## Future Improvements

* Add and validate official historical data for additional entrance exams.
* Implement and test domicile and quota-aware filtering.
* Evaluate model performance separately for exams, categories and institute groups.
* Improve uncertainty reporting and cutoff trend visualizations.
* Add automated API and frontend integration tests.
* Deploy the frontend and backend with appropriate production configuration.

## Project Disclaimer

CollegeRank AI is an educational decision-support project. Predictions are experimental estimates based on the collected data and implemented model. They must not be treated as guaranteed admission outcomes.
