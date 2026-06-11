# n8n Workflow: Send OTP Email

## Webhook URL to activate
`https://n8n.digitransolutions.in/webhook/send-otp`

## Steps to create in n8n

### Node 1 — Webhook (Trigger)
- Type: **Webhook**
- Method: POST
- Path: `send-otp`
- Authentication: None (or add header auth if desired)
- Response Mode: Respond immediately with HTTP 200

### Node 2 — Send Email
- Type: **Send Email** (or Gmail / SMTP node)
- From: `ceo@digitransolutions.in`
- From Name: `DigiTran Solutions`
- To: `{{ $json.body.to_email }}`
- Subject: `{{ $json.body.subject }}`
- HTML Body: `{{ $json.body.html }}`
- Text Body: `{{ $json.body.text }}`

### SMTP Settings (Google Workspace)
- Host: `smtp.gmail.com`
- Port: `465` (SSL) or `587` (TLS)
- User: `ceo@digitransolutions.in`
- Password: App Password from Google Workspace

## Payload received by n8n
```json
{
  "to_email":   "user@example.com",
  "to_name":    "John Doe",
  "from_email": "ceo@digitransolutions.in",
  "from_name":  "DigiTran Solutions",
  "subject":    "Your OTP for AI Testing Automation Platform",
  "otp":        "123456",
  "html":       "<div>...HTML email...</div>",
  "text":       "Your OTP is 123456"
}
```

## Important
- Activate the workflow — production webhooks only work when workflow is **Active**
- Toggle the workflow Active switch (top right in n8n editor)
