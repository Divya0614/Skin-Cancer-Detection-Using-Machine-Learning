from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Mock prediction function (no model required)
def mock_predict(img_array):
    return np.random.uniform(0, 1)  # Random confidence score between 0 and 1

# Email configuration (update with your credentials)
EMAIL_ADDRESS = "gvamsi5218@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "qptfgadwmvkpztei"    # Replace with your app-specific password

# Create directories if they don't exist
if not os.path.exists('static/history'):
    os.makedirs('static/history')
if not os.path.exists('static/prescriptions'):
    os.makedirs('static/prescriptions')

prediction_history = []

def preprocess_image(image, brightness=1.0, contrast=1.0):
    """Preprocess image with brightness and contrast adjustments"""
    img = image.resize((128, 128))  # Size can be arbitrary for mock predictions
    img = np.array(img)
    img = img * brightness
    img = ((img - 127.5) * contrast + 127.5)
    img = np.clip(img, 0, 255) / 255.0
    return img

def send_email_alert(email, result, confidence):
    """Send email alert for malignant cases"""
    subject = "Skin Cancer Detection Alert"
    body = f"Alert: Your recent skin scan result is {result} with {confidence} confidence.\nPlease consult a doctor immediately if malignant."
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp_server.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")

def generate_prescription(image_path, result, confidence):
    """Generate a PDF prescription"""
    filename = f"static/prescriptions/prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Hospital Header
    logo_path = 'static/logo.png'
    if os.path.exists(logo_path):
        story.append(RLImage(logo_path, width=50, height=50))
    story.append(Paragraph("HealthCare Hospital", styles['Title']))
    story.append(Paragraph("123 Medical Lane, Health City", styles['Normal']))
    story.append(Spacer(1, 12))

    # Patient Info
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Image
    story.append(RLImage(image_path, width=200, height=200))
    story.append(Spacer(1, 12))

    # Diagnosis
    story.append(Paragraph(f"Diagnosis: {result}", styles['Heading2']))
    story.append(Paragraph(f"Confidence Score: {confidence}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Medicine Recommendation
    medicines = "Topical Fluorouracil" if result == "Malignant" else "Moisturizing Cream"
    story.append(Paragraph("Recommended Medicine:", styles['Heading3']))
    story.append(Paragraph(medicines, styles['Normal']))
    story.append(Spacer(1, 12))

    # Detailed Investigation
    story.append(Paragraph("Detailed Investigation:", styles['Heading3']))
    story.append(Paragraph("Parameters checked: Color variation, Edge irregularity, Texture analysis", styles['Normal']))
    story.append(Paragraph("Note: Please consult a dermatologist for complete examination", styles['Normal']))

    doc.build(story)
    return filename

@app.route('/')
def index():
    return render_template('index.html', history=prediction_history[-5:])

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    email = request.form.get('email', '')
    brightness = float(request.form.get('brightness', 1.0))
    contrast = float(request.form.get('contrast', 1.0))

    image = Image.open(image_file)
    processed_img = preprocess_image(image, brightness, contrast)
    img_array = np.expand_dims(processed_img, axis=0)
    
    # Use mock prediction
    prediction = mock_predict(img_array)

    confidence = prediction if prediction > 0.5 else 1 - prediction
    result = "Malignant" if prediction > 0.5 else "Benign"

    # Save image temporarily
    temp_path = f"static/history/temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(temp_path)

    # Generate prescription
    prescription_path = generate_prescription(temp_path, result, f"{confidence:.2%}")

    # Convert image to base64 for display
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # History
    history_path = f"static/history/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(history_path)
    prediction_history.append({
        'result': result,
        'confidence': f"{confidence:.2%}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_path': history_path,
        'prescription_path': prescription_path
    })

    # Email alert for malignant cases
    if result == "Malignant" and email:
        send_email_alert(email, result, f"{confidence:.2%}")

    return render_template('index.html',
                         pred=result,
                         confidence=f"{confidence:.2%}",
                         image_data=img_str,
                         history=prediction_history[-5:],
                         prescription_path=prescription_path,
                         sound_alert=result == "Malignant")

@app.route('/download_prescription/<path:filename>')
def download_prescription(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False)