import cv2
import os
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def detect_and_save_falls(input_video_path):
    # Initialize YOLO model
    model = YOLO("fall_det_1.pt")

    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the desired display window size
    display_width = 1280
    display_height = 720

    # Calculate the aspect ratio of the video
    aspect_ratio = width / height

    # Calculate the height based on the desired width and aspect ratio
    new_height = int(display_width / aspect_ratio)

    # Check if the calculated height fits within the desired display height
    if new_height > display_height:
        new_height = display_height
        new_width = int(display_height * aspect_ratio)
    else:
        new_width = display_width

    # Define the codec and create VideoWriter object
    output_folder = "./static/result"
    output_filename = "output_video.mp4"
    output_path = os.path.join(output_folder, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Resize the frame to fit the display window
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

            # Write the frame to the output video
            out.write(resized_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

def send_email_demo(receiver_email, attachment_path):
    # Email configuration
    sender_email = "actionrecognition2024@gmail.com"  # Update with your email
    sender_password = "inmf akze tvxg vihp"  # Update with your password

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Fall Detected"

    # Add body to email
    body = "Fall has been detected. Please find the attached video."
    message.attach(MIMEText(body, "plain"))

    # Open the file to be sent
    with open(attachment_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {attachment_path}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, text)


def detect_save_demo(input_video_path, receiver_email):
    # Initialize YOLO model
    model = YOLO("fall_det_1.pt")

    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the desired display window size
    display_width = 1280
    display_height = 720

    # Calculate the aspect ratio of the video
    aspect_ratio = width / height

    # Calculate the height based on the desired width and aspect ratio
    new_height = int(display_width / aspect_ratio)

    # Check if the calculated height fits within the desired display height
    if new_height > display_height:
        new_height = display_height
        new_width = int(display_height * aspect_ratio)
    else:
        new_width = display_width

    # Define the codec and create VideoWriter object
    output_folder = "./static/result"
    output_filename = "output_video.mp4"
    output_path = os.path.join(output_folder, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.5)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Resize the frame to fit the display window
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

            # Write the frame to the output video
            out.write(resized_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, close the output video writer, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Detected output video saved at: {output_path}")

    # Send email with the detected video as attachment
    send_email_demo(receiver_email, output_path)
