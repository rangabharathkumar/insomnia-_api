# Insomnia API (Health Oracle)

Live docs: https://insomnia-api-lf3g.onrender.com/docs

Description

This repository provides the Insomnia prediction API and is a part of the Health Oracle application. The machine learning model powering this service has been trained, tested, and deployed. Use the API to get insomnia risk predictions from patient or survey features. Refer to the live docs for the canonical endpoint definitions and data schema.

Key notes

- Part of the Health Oracle microservices ecosystem.
- ML model: trained, validated, and deployed; served via the API.
- API docs: visit the /docs route for interactive Swagger/OpenAPI documentation.

Quick Start (consume the API)

- Base URL: https://insomnia-api-lf3g.onrender.com
- Interactive docs: https://insomnia-api-lf3g.onrender.com/docs

Example JSON request (example fields — confirm exact names/types in /docs):

POST /predict
Content-Type: application/json

Request body (example):
{
  "age": 29,
  "gender": "female",
  "sleep_duration_hours": 5.5,
  "sleep_latency_minutes": 30,
  "caffeine_intake_mg": 150,
  "stress_level": 3,
  "anxiety_level": 2
}

Example JSON response (example):
{
  "prediction": 1,
  "probability": 0.78,
  "label": "insomnia"
}

(Local field names and response structure may differ — consult /docs for exact details.)

Local development

1. Clone the repository
   git clone https://github.com/rangabharathkumar/insomnia-_api.git
2. Create and activate a virtual environment, then install dependencies
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
3. Run the service locally (example using uvicorn)
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
4. Open http://localhost:8000/docs for interactive API docs

Environment variables

- Place model path, secret keys, or other sensitive configs in environment variables or a .env file. Refer to the codebase for names and usage.

Testing

- Run available tests with pytest or the test runner configured in the repo.

How to use this repo as a reference

- Use this codebase as a template for serving ML models as REST APIs.
- Replace the model artifact and adapt preprocessing/postprocessing to your dataset.
- Confirm and update the request/response schema to match your model's input/output.

Contact / Author

Ranga Bharath Kumar — https://github.com/rangabharathkumar

For exact schema and endpoints, consult the live docs: https://insomnia-api-lf3g.onrender.com/docs
