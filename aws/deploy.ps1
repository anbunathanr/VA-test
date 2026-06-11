# ─────────────────────────────────────────────────────────────────────────────
# DigiTranVA — AWS Serverless Deployment Script (PowerShell)
# ⚠️  NO AWS credentials in this file — uses 'aws configure' profile
# ─────────────────────────────────────────────────────────────────────────────

param(
    [string]$Stage     = "prod",
    [string]$Region    = "us-east-1",
    [string]$S3Bucket  = "sanjay-ai-va-test-2026",
    [string]$SecretKey = ""
)

$ErrorActionPreference = "Stop"
$AppDir    = "$PSScriptRoot\..\flask_app"
$AwsDir    = $PSScriptRoot
$AccountId = (aws sts get-caller-identity --query Account --output text)

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  DigiTranVA Serverless Deployment" -ForegroundColor Cyan
Write-Host "  Account : $AccountId" -ForegroundColor Cyan
Write-Host "  Region  : $Region" -ForegroundColor Cyan
Write-Host "  Stage   : $Stage" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# ── STEP 1: Install Python dependencies into package dir ─────────────────────
Write-Host "STEP 1: Installing Python dependencies..." -ForegroundColor Yellow
$PackageDir = "$AppDir\package"
if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
New-Item -ItemType Directory -Path $PackageDir | Out-Null

pip install `
    flask==3.0.3 `
    mangum==0.17.0 `
    boto3 `
    xhtml2pdf==0.2.17 `
    --target $PackageDir `
    --quiet

Write-Host "  ✅ Dependencies installed" -ForegroundColor Green

# ── STEP 2: Create Lambda deployment ZIP ─────────────────────────────────────
Write-Host "`nSTEP 2: Creating Lambda ZIP package..." -ForegroundColor Yellow

$ZipPath = "$AwsDir\flask_lambda.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath }

# Copy app files into package dir for zipping
$FilesToCopy = @(
    "app.py", "auth.py", "otp_service.py", "session_service.py",
    "automation.py", "dynamo_session.py", "dynamo_otp.py",
    "dynamo_results.py", "automation_lambda.py"
)
foreach ($f in $FilesToCopy) {
    Copy-Item "$AppDir\$f" "$PackageDir\" -ErrorAction SilentlyContinue
}

# Copy folders
Copy-Item "$AppDir\templates" "$PackageDir\templates" -Recurse -Force
Copy-Item "$AppDir\test_framework" "$PackageDir\test_framework" -Recurse -Force

# Zip everything
Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipPath -Force
$ZipSize = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "  ✅ ZIP created: flask_lambda.zip ($ZipSize MB)" -ForegroundColor Green

# ── STEP 3: Upload ZIP to S3 ──────────────────────────────────────────────────
Write-Host "`nSTEP 3: Uploading to S3..." -ForegroundColor Yellow
$S3Key = "deployments/digitranva/flask_lambda.zip"
aws s3 cp $ZipPath "s3://$S3Bucket/$S3Key" --region $Region
Write-Host "  ✅ Uploaded to s3://$S3Bucket/$S3Key" -ForegroundColor Green

# ── STEP 4: Deploy SAM stack ──────────────────────────────────────────────────
Write-Host "`nSTEP 4: Deploying SAM stack..." -ForegroundColor Yellow

if (-not $SecretKey) {
    $SecretKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
    Write-Host "  Generated Flask secret key (save this): $SecretKey" -ForegroundColor Magenta
}

sam deploy `
    --template-file "$AwsDir\template.yaml" `
    --stack-name "digitranva-$Stage" `
    --s3-bucket $S3Bucket `
    --s3-prefix "sam/digitranva" `
    --region $Region `
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM `
    --parameter-overrides `
        "FlaskSecretKey=$SecretKey" `
        "Stage=$Stage" `
    --no-fail-on-empty-changeset

Write-Host "  ✅ SAM stack deployed" -ForegroundColor Green

# ── STEP 5: Get outputs ───────────────────────────────────────────────────────
Write-Host "`nSTEP 5: Getting deployment outputs..." -ForegroundColor Yellow

$Outputs = aws cloudformation describe-stacks `
    --stack-name "digitranva-$Stage" `
    --region $Region `
    --query "Stacks[0].Outputs" `
    --output json | ConvertFrom-Json

$CloudFrontURL = ($Outputs | Where-Object { $_.OutputKey -eq "CloudFrontURL" }).OutputValue
$ApiURL        = ($Outputs | Where-Object { $_.OutputKey -eq "ApiGatewayURL" }).OutputValue
$FrontendBucket = ($Outputs | Where-Object { $_.OutputKey -eq "FrontendBucket" }).OutputValue

Write-Host "  CloudFront URL : $CloudFrontURL" -ForegroundColor Cyan
Write-Host "  API Gateway URL: $ApiURL" -ForegroundColor Cyan
Write-Host "  Frontend Bucket: $FrontendBucket" -ForegroundColor Cyan

# ── STEP 6: Update Lambda from S3 ────────────────────────────────────────────
Write-Host "`nSTEP 6: Updating Lambda function code..." -ForegroundColor Yellow

aws lambda update-function-code `
    --function-name "digitranva-flask-$Stage" `
    --s3-bucket $S3Bucket `
    --s3-key $S3Key `
    --region $Region `
    --output json | Out-Null

Write-Host "  ✅ Lambda updated" -ForegroundColor Green

# ── STEP 7: Build and deploy frontend to S3 ───────────────────────────────────
Write-Host "`nSTEP 7: Uploading frontend to S3..." -ForegroundColor Yellow

# Create standalone HTML files from Flask templates
# For serverless, pages are served by Lambda — we just upload static assets
if (Test-Path "$AppDir\static") {
    aws s3 sync "$AppDir\static" "s3://$FrontendBucket/static" `
        --region $Region `
        --delete
    Write-Host "  ✅ Static assets uploaded" -ForegroundColor Green
}

# ── STEP 8: Invalidate CloudFront cache ──────────────────────────────────────
Write-Host "`nSTEP 8: Invalidating CloudFront cache..." -ForegroundColor Yellow

$DistId = aws cloudfront list-distributions `
    --query "DistributionList.Items[?Comment=='DigiTranVA AI Testing Platform'].Id" `
    --output text

if ($DistId) {
    aws cloudfront create-invalidation `
        --distribution-id $DistId `
        --paths "/*" `
        --output json | Out-Null
    Write-Host "  ✅ Cache invalidated" -ForegroundColor Green
}

# ── DONE ──────────────────────────────────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  ✅ DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "  🌐 App URL: $CloudFrontURL" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green
