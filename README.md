# DigiTranVA — AI Voice Assistant Testing Automation Platform

Automated testing framework for the DigiTranVA voice assistant, with OTP sign-in, Playwright automation, and AWS serverless deployment.

## Live URLs
- **CloudFront**: https://d2qrljay7xtfr5.cloudfront.net/
- **API Gateway**: https://eokh12b9xk.execute-api.us-east-1.amazonaws.com/prod/

## Features
- OTP sign-in (demo mode shows OTP on screen, production sends via n8n/SMTP)
- 30 automated test cases across 5 phases
- Playwright browser automation against the VA chatbot
- PDF report generation + n8n email delivery
- AWS Lambda + CloudFront + DynamoDB serverless backend

## Run Locally
```bash
cd flask_app
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Run Automation
Sign in → Dashboard → Click **Run Voice AI Automation Test**

Playwright must be installed:
```bash
pip install playwright
playwright install chromium
```

## Deploy to AWS
```bash
python deploy_via_s3.py
```

## Environment Variables
Copy `.env.example` to `.env` and fill in values.

## Architecture
- **Frontend**: CloudFront → API Gateway → Lambda (Flask)
- **Auth**: HMAC-signed URL tokens (stateless, no cookie dependency)
- **Sessions**: DynamoDB `Sessions` table
- **OTP**: DynamoDB `OTP` table + n8n email / SMTP fallback
- **Results**: DynamoDB `digitranva-conversations` table
