// Load .env if present
const fs = require('fs');
if (fs.existsSync('./.env')) {
  require('fs').readFileSync('./.env', 'utf8')
    .split('\n')
    .forEach(line => {
      const [key, ...rest] = line.split('=');
      if (key && !key.startsWith('#')) process.env[key.trim()] = rest.join('=').trim();
    });
}

const express = require('express');
const nodemailer = require('nodemailer');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// In-memory OTP store: { email: { otp, expiry, userData } }
const otpStore = {};

// Force Node.js to prefer IPv4 over IPv6
const dns = require('dns');
dns.setDefaultResultOrder('ipv4first');

// SMTP transporter — digitransolutions.in uses Google Workspace SMTP
// Using direct IP to avoid DNS timeout issues
const transporter = nodemailer.createTransport({
  host: '192.178.211.109', // smtp.gmail.com IPv4 — avoids IPv6 DNS timeout
  port: 465,
  secure: true, // SSL on port 465
  auth: {
    user: 'ceo@digitransolutions.in',
    pass: process.env.SMTP_PASS || 'YOUR_APP_PASSWORD'
  },
  tls: {
    rejectUnauthorized: false,
    servername: 'smtp.gmail.com' // required for SNI when using IP directly
  },
  debug: true,
  logger: true
});

// Verify SMTP connection on startup
transporter.verify((error, success) => {
  if (error) {
    console.error('❌ SMTP Connection FAILED:', error.message);
    console.error('   → Check host, port, username and password in .env');
  } else {
    console.log('✅ SMTP Connection OK — ready to send emails');
  }
});

// GET /test-smtp — manual connection test endpoint
app.get('/test-smtp', async (req, res) => {
  try {
    await transporter.verify();
    res.json({ success: true, message: 'SMTP connection is working.' });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message, code: err.code });
  }
});

// Generate a 6-digit OTP
function generateOTP() {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

// POST /send-otp — receives form data, sends OTP email
app.post('/send-otp', async (req, res) => {
  const { fullName, email, phone } = req.body;

  if (!fullName || !email || !phone) {
    return res.status(400).json({ success: false, message: 'All fields are required.' });
  }

  // Basic email validation
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ success: false, message: 'Invalid email address.' });
  }

  // Basic phone validation
  const phoneRegex = /^[0-9]{7,15}$/;
  if (!phoneRegex.test(phone.replace(/[\s\-\+]/g, ''))) {
    return res.status(400).json({ success: false, message: 'Invalid phone number.' });
  }

  const otp = generateOTP();
  const expiry = Date.now() + 5 * 60 * 1000; // 5 minutes

  // Store OTP along with user data
  otpStore[email] = { otp, expiry, userData: { fullName, email, phone } };

  // Always print OTP to terminal (useful for testing when email is unavailable)
  console.log('─────────────────────────────────────');
  console.log(`📧 OTP for ${email}`);
  console.log(`🔑 OTP CODE: ${otp}`);
  console.log(`⏰ Expires in 5 minutes`);
  console.log('─────────────────────────────────────');

  try {
    await transporter.sendMail({
      from: '"DigiTransolutions" <ceo@digitransolutions.in>',
      to: email,
      subject: 'Your OTP Code - DigiTransolutions',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">
          <div style="background: #1a73e8; padding: 24px; text-align: center;">
            <h2 style="color: #fff; margin: 0;">DigiTransolutions</h2>
          </div>
          <div style="padding: 32px;">
            <p style="font-size: 16px; color: #333;">Hello <strong>${fullName}</strong>,</p>
            <p style="font-size: 15px; color: #555;">Your One-Time Password (OTP) for sign-in is:</p>
            <div style="text-align: center; margin: 28px 0;">
              <span style="font-size: 40px; font-weight: bold; letter-spacing: 10px; color: #1a73e8;">${otp}</span>
            </div>
            <p style="font-size: 13px; color: #888;">This OTP is valid for <strong>5 minutes</strong>. Do not share it with anyone.</p>
          </div>
          <div style="background: #f5f5f5; padding: 16px; text-align: center;">
            <p style="font-size: 12px; color: #aaa; margin: 0;">© 2026 DigiTransolutions. All rights reserved.</p>
          </div>
        </div>
      `
    });

    console.log(`✅ OTP email sent to ${email}`);
    return res.json({ success: true, message: 'OTP sent successfully to your email.' });
  } catch (err) {
    console.error('❌ SMTP Send Error:', err.message, '| Code:', err.code);
    // OTP is still stored and valid — let user proceed using terminal OTP
    return res.json({
      success: true,
      message: 'OTP generated. Email delivery failed — check the terminal for your OTP.'
    });
  }
});

// POST /verify-otp — validates OTP
app.post('/verify-otp', (req, res) => {
  const { email, otp } = req.body;

  if (!email || !otp) {
    return res.status(400).json({ success: false, message: 'Email and OTP are required.' });
  }

  const record = otpStore[email];

  if (!record) {
    return res.status(400).json({ success: false, message: 'No OTP found for this email. Please request a new one.' });
  }

  if (Date.now() > record.expiry) {
    delete otpStore[email];
    return res.status(400).json({ success: false, message: 'OTP has expired. Please request a new one.' });
  }

  if (record.otp !== otp.trim()) {
    return res.status(400).json({ success: false, message: 'Incorrect OTP. Please try again.' });
  }

  // OTP verified — clear from store
  const userData = record.userData;
  delete otpStore[email];

  return res.json({ success: true, message: 'OTP verified successfully.', user: userData });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
