# ── Automation Lambda Container Image ────────────────────────────────────────
# Uses AWS Lambda Python base + installs Playwright Chromium
# ⚠️  No AWS credentials in this file — Lambda execution role provides access
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for Playwright/Chromium
RUN dnf install -y \
    atk \
    cups-libs \
    gtk3 \
    libdrm \
    libgbm \
    libXcomposite \
    libXdamage \
    libXrandr \
    libXScrnSaver \
    pango \
    alsa-lib \
    nss \
    xorg-x11-fonts-100dpi \
    xorg-x11-fonts-75dpi \
    && dnf clean all

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir \
    playwright==1.49.0 \
    xhtml2pdf==0.2.17 \
    boto3 \
    requests

# Install Playwright Chromium browser
RUN playwright install chromium
RUN playwright install-deps chromium

# Lambda handler
CMD ["automation_lambda.lambda_handler"]
