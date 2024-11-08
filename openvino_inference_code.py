import cv2
import numpy as np
from openvino.runtime import Core, Tensor
import supervision as sv

# Paths to model files (OpenVINO IR format)
model_xml = '/home/cognitica-i7-13thgen/NPS/OpenVino_model/py_person_0.2/model1.xml'  # Adjust to your model path
model_bin = '/home/cognitica-i7-13thgen/NPS/OpenVino_model/py_person_0.2/model1.bin'  # Adjust to your model path

# Initialize OpenVINO runtime
core = Core()

# Read the OpenVINO model
model = core.read_model(model=model_xml, weights=model_bin)

# Compile the model for the device (e.g., CPU, GPU)
compiled_model = core.compile_model(model=model, device_name='GPU')  # Change to 'CPU' if needed

# Prepare input and output layers
input_layer = compiled_model.input(0)  # Get the first input layer
output_layer_count = len(compiled_model.outputs)  # Get the number of output layers

# Create an inference request
infer_request = compiled_model.create_infer_request()

# Preprocessing Parameters for DETR
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
input_size = (640, 640)  # Input size the model was trained with

# Load video
video_path = 0  # Path to your video (0 for webcam)
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize Supervision BoxAnnotator for bounding box drawing
box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame (Resize, Normalize, Convert to CHW)
    image_resized = cv2.resize(frame, input_size)
    image = image_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = (image - mean) / std  # Standardize with model's mean and std
    image = image.transpose((2, 0, 1))  # HWC to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 3, 640, 640)

    # Create OpenVINO tensor from the preprocessed image
    input_tensor = Tensor(image)
    infer_request.set_input_tensor(input_tensor)

    # Perform inference
    infer_request.infer()

    # Get the output tensors
    outputs = [infer_request.get_output_tensor(i).data for i in range(output_layer_count)]

    # Process detections (assuming first output is the detection scores, second output is bounding boxes)
    detection_scores = outputs[0]  # Detected bounding boxes and confidences
    bounding_boxes = outputs[1]  # Bounding box coordinates

    # Apply sigmoid to the detection scores (convert logits to probabilities)
    detection_scores = 1 / (1 + np.exp(-detection_scores))

    # Postprocessing: Iterate through detections
    height, width = frame.shape[:2]  # Get original frame dimensions

    # Visualize detections on the frame
    for i in range(detection_scores.shape[1]):  # Iterate through all detections
        # Extract confidence score (detections[i, 0]) and bounding box coordinates (boxes[i])
        confidence = detection_scores[0, i, 0]  # Confidence score (index 0 is assumed for confidence)
        xmin, ymin, xmax, ymax = bounding_boxes[0, i]  # Get bounding box coordinates (xmin, ymin, xmax, ymax)

        # Convert normalized coordinates to pixel values
        xmin = int(xmin * width)
        ymin = int(ymin * height)
        xmax = int(xmax * width)
        ymax = int(ymax * height)
        print(f'conf: {confidence}, coords: {(xmin, ymin, xmax, ymax)}')

        # Apply a threshold to filter low-confidence detections (e.g., 0.08)
        if confidence > 0.03:
            # Ensure bounding box coordinates are within image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)

            # Convert bounding box to a numpy array (instead of tuple)
            bbox = np.array([xmin, ymin, xmax, ymax])

            # Create a label with confidence score
            label = f"Confidence: {confidence:.2f}"
            class_id = 0  # Replace with actual class_id (e.g., for "person")

            # Prepare detections in the format needed by Supervision library
            detections = [(bbox, confidence, class_id, None)]

            # Annotate the frame with bounding boxes
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=[label])

    # Show the frame with detections
    cv2.imshow('Video Inference', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
