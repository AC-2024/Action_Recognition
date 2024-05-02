from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response
import os
import cv2  # Added the import statement for cv2

from ultralytics import YOLO
from werkzeug.utils import secure_filename
# from detect_falls import detect_and_save_falls
from detect_falls import detect_save_demo

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

app = Flask(__name__)

# Define the upload folder and allowed extensions for uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the result folder for storing the resulted video
RESULT_FOLDER = os.path.join('static', 'result')
app.config['RESULT_FOLDER'] = RESULT_FOLDER


# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Email Configuration
SENDER_EMAIL = "actionrecognition2024@gmail.com"  # Enter your email address
SENDER_PASSWORD = "inmf akze tvxg vihp"      # Enter your email password


def send_email(receiver_email, image_path):
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = "Emergency: Fall Detected"

    # Add message body
    body = "A fall has been detected. Please find the attached image for reference."
    msg.attach(MIMEText(body, 'plain'))

    # Attach image
    with open(image_path, 'rb') as fp:
        img = MIMEImage(fp.read())
    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
    msg.attach(img)

    # Send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())

def generate_frames(camera_url, receiver_email):
    # Initialize YOLO model
    model = YOLO("fall_det_1.pt")

    # Open the video stream from the IP camera
    cap = cv2.VideoCapture()
    cap.open(camera_url)

    # Flag to capture the first frame when fall is detected
    fall_detected = False

    # Flag to indicate if the image has been captured after fall detection
    image_captured = False

    # Frame counter to track the number of frames processed after fall detection
    frame_counter = 0

    # Frame number to capture after fall detection
    capture_frame_number = 10  # Adjust this number as needed

    while True:
        # Read a frame from the video stream
        success, frame = cap.read()

        if not success:
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.5)

        # Check if fall is detected
        if not fall_detected:
            for result in results:
                if len(result.boxes) > 0:  # Assuming at least one object is detected
                    fall_detected = True
                    break

        if fall_detected and not image_captured:
            frame_counter += 1

            # If enough frames have been processed after fall detection, capture the frame
            if frame_counter == capture_frame_number:
                # Save the annotated frame with bounding boxes
                annotated_frame = results[0].plot() if results else frame
                cv2.imwrite('fall_detected_frame.jpg', annotated_frame)

                # Send emergency email with the captured image
                send_email(receiver_email, 'fall_detected_frame.jpg')

                # Set the flag to indicate that the image has been captured
                image_captured = True

        # If fall is no longer detected, reset the flags and frame counter
        if not fall_detected:
            image_captured = False
            frame_counter = 0

        # Reset fall detection flag if the person stands up again
        if fall_detected and image_captured:
            for result in results:
                if len(result.boxes) == 0:  # Assuming no objects are detected
                    fall_detected = False
                    break

        # Visualize the results on the frame
        annotated_frame = results[0].plot() if results else frame

        # Convert the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the video capture object
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/about-us')
def about():
    return render_template('about-us.html')


@app.route('/contacts')
def contacts():
    return render_template('contacts.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        receiver_email = request.form['receiver_email']  # Retrieve receiver's email from the form
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Ensure 'uploads' directory exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Perform fall detection on the uploaded video
            detect_save_demo(file_path, receiver_email)
            return redirect(url_for('result'))


@app.route('/video_feed', methods=['POST'])
def video_feed():
    # Get the IP camera URL from the form input
    camera_url = request.form['camera_url']
    receiver_email = request.form['receiver_email']

    # Return the response for video streaming
    return Response(generate_frames(camera_url, receiver_email), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    # Get the path of the resulted video
    result_video_path = os.path.join(app.config['RESULT_FOLDER'], 'output_video.mp4')
    return render_template('result.html', result_video_path=result_video_path)


@app.route('/video')
def video():
    # Get the path of the resulted video
    result_video_path = os.path.join(app.config['RESULT_FOLDER'], 'output_video.mp4')
    return send_file(result_video_path, mimetype='video/mp4')


if __name__ == '__main__':
    app.run(debug=True)
