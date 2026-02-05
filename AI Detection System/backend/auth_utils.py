
import random
import time

# Simple in-memory store: {email: {"code": "1234", "expires": timestamp}}
otp_store = {}

def generate_otp(email):
    """Generates a 4-digit OTP and stores it."""
    code = f"{random.randint(1000, 9999)}"
    expires = time.time() + 300 # 5 minutes
    otp_store[email] = {"code": code, "expires": expires}
    return code

import smtplib
from email.message import EmailMessage

# Credentials
SMTP_EMAIL = "p.vadivel043@gmail.com"
SMTP_PASSWORD = "hliy ciid xhgt fuxo"

def send_email(to_email, code):
    """
    Sends the verification code via Gmail SMTP using a rich HTML template.
    """
    msg = EmailMessage()
    msg['Subject'] = 'Metadata Verification Code - AI Image Detection System'
    msg['From'] = SMTP_EMAIL
    msg['To'] = to_email

    # Plain text fallback
    plain_content = f"""Dear User,
Greetings from AI Image Detection System!
Your OTP Code is: {code}
It is valid for 5 minutes.
"""
    msg.set_content(plain_content)

    # HTML Template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <p>Dear User,</p>
        <p>Greetings from <strong>AI Image Detection System</strong>!</p>
        
        <p>We’re delighted to welcome you to our advanced AI-powered platform designed to help you accurately identify and analyze <strong>AI-generated and real images</strong> using cutting-edge forensic and machine learning technologies.</p>
        
        <p>To ensure the security of your account, please verify your email address using the <strong>One-Time Password (OTP)</strong> below:</p>
        
        <div style="margin: 20px 0; padding: 15px; background-color: #f0f7ff; border-left: 5px solid #2563eb; width: fit-content;">
            <p style="margin: 0;"><strong>🔐 Your OTP Code:</strong></p>
            <p style="font-size: 24px; font-weight: bold; color: #2563eb; margin: 5px 0 0 0;">{code}</p>
        </div>
        
        <blockquote style="border-left: 3px solid #ccc; margin: 10px 0; padding-left: 10px; color: #555;">
            This OTP is valid for a limited time. Please do not share it with anyone for security reasons.
        </blockquote>
        
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
        
        <h3>What Our System Offers</h3>
        <ul>
            <li>Advanced <strong>AI vs Real Image Detection</strong></li>
            <li>Deep <strong>Metadata & C2PA Analysis</strong></li>
            <li>Image <strong>Forensic Techniques</strong> (ELA, Noise, Frequency Analysis)</li>
            <li>Secure and privacy-focused verification</li>
            <li>Continuous updates with latest AI detection models</li>
        </ul>
        
        <p>Once verified, you’ll gain full access to all features and future enhancements.</p>
        
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
        
        <h3>🌸 தமிழில் உங்களை வரவேற்கிறோம் 🌸</h3>
        
        <p><strong>AI Image Detection System-க்கு உங்களை அன்புடன் வரவேற்கிறோம்!</strong><br>
        உங்கள் பயணம் பாதுகாப்பாகவும், நம்பிக்கையுடனும், பயனுள்ளதாகவும் இருக்க எங்கள் குழு எப்போதும் உங்களுடன் இருக்கும்.<br>
        உங்கள் ஆதரவுக்கும் நம்பிக்கைக்கும் நன்றி.</p>
        
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
        
        <p>Warm regards,<br>
        <strong>Vadivel P,B.Sc.(CS), M.Sc.(IT)</strong><br>
        Creator & Developer<br>
        <strong>AI Image Detection System</strong></p>
        
        <p>We’re happy to have you with us! 😊</p>
    </body>
    </html>
    """
    
    msg.add_alternative(html_content, subtype='html')

    try:
        # Connect to Gmail SMTP
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[SMTP] Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[SMTP ERROR] Failed to send email: {e}")
        return False

def verify_otp(email, code):
    """Verifies the provided OTP."""
    if email not in otp_store:
        return False
    
    data = otp_store[email]
    
    if time.time() > data["expires"]:
        del otp_store[email]
        return False # Expired
        
    if data["code"] == code:
        del otp_store[email] # Consume code
        return True
        
    return False
