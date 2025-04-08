import time
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import csv
import os
import rawpy
import imageio
from flask import Flask, jsonify, request
from supabase import create_client, Client
from threading import Thread
import logging
from datetime import datetime
import struct

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase setup
url = "https://wrktsadpbrocmuyuztxl.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indya3RzYWRwYnJvY211eXV6dHhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDEzNjcwMzMsImV4cCI6MjA1Njk0MzAzM30.EDH66keNiSQB6wvsYJ-STKPQgIBRNj9QG7sAH0Xvm3s"
supabase: Client = create_client(url, key)

# Files to ignore
IGNORE_FILES = ['.emptyFolderPlaceholder', '.DS_Store', 'Thumbs.db']

# Supported RAW image formats (professional cameras)
CAMERA_RAW_EXTENSIONS = ['.cr2', '.nef', '.arw', '.orf', '.rw2', '.dng']

# ESP32CAM RAW format
ESP32_RAW_EXTENSION = '.raw'

# Supported image formats
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'] + CAMERA_RAW_EXTENSIONS + [ESP32_RAW_EXTENSION]

# Temp directory for converted images
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_converted")
os.makedirs(TEMP_DIR, exist_ok=True)

# Conversion tracking
converted_files = {}  # Maps original RAW filename to converted JPG filename

# ESP32CAM RAW to JPG conversion function
def convert_esp32cam_raw_to_jpg(raw_data, filename):
    """
    Convert ESP32CAM RAW image data to JPG format
    
    ESP32CAM RAW files could be in different formats depending on configuration:
    1. Standard camera formats (processed by the general raw converter)
    2. Grayscale format (typically 320x240 or 640x480)
    3. YUV format (requires specific conversion)
    4. RGB565 format (common with ESP32CAM)
    5. JPEG format (already in JPEG format just with .raw extension)
    
    Args:
        raw_data (bytes): RAW image data
        filename (str): Original filename for logging purposes
        
    Returns:
        PIL.Image: Image object or None if conversion failed
    """
    try:
        # First try to open as if it's already a JPEG (common for ESP32CAM)
        try:
            img = Image.open(BytesIO(raw_data))
            logger.info(f"ESP32CAM file {filename} was already in a readable format")
            return img
        except Exception:
            pass  # Not a standard format, continue with conversion attempts
        
        # Try to determine dimensions from filename
        # ESP32CAM files often have format like ESP32CAM_date_N.raw
        # If no size info in filename, default to 320x240 (common ESP32CAM resolution)
        width, height = 320, 240

        # Create BytesIO object for output
        buffer = BytesIO()
        
        # Try to interpret as RGB565 format (common for ESP32CAM)
        try:
            # Calculate expected size for RGB565 (16 bits per pixel)
            expected_size = width * height * 2
            
            if len(raw_data) == expected_size:
                # Convert RGB565 to RGB888
                rgb_data = bytearray(width * height * 3)
                
                for i in range(0, len(raw_data), 2):
                    if i + 1 < len(raw_data):
                        pixel = struct.unpack('<H', raw_data[i:i+2])[0]
                        
                        # Extract RGB components (5 bits R, 6 bits G, 5 bits B)
                        r = (pixel >> 11) & 0x1F
                        g = (pixel >> 5) & 0x3F
                        b = pixel & 0x1F
                        
                        # Scale to 8 bits per channel
                        r = (r * 255) // 31
                        g = (g * 255) // 63
                        b = (b * 255) // 31
                        
                        idx = (i // 2) * 3
                        rgb_data[idx] = r
                        rgb_data[idx + 1] = g
                        rgb_data[idx + 2] = b
                
                # Create image from RGB data
                img = Image.frombytes('RGB', (width, height), bytes(rgb_data))
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                return Image.open(buffer)
        except Exception as e:
            logger.warning(f"RGB565 conversion failed for {filename}: {str(e)}")
        
        # Try to interpret as grayscale (8-bit)
        try:
            expected_size = width * height
            
            if len(raw_data) == expected_size:
                img = Image.frombytes('L', (width, height), raw_data)
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                return Image.open(buffer)
        except Exception as e:
            logger.warning(f"Grayscale conversion failed for {filename}: {str(e)}")
        
        # If all attempts fail, save the raw data to a file for debugging
        debug_path = os.path.join(TEMP_DIR, f"debug_{os.path.basename(filename)}")
        with open(debug_path, 'wb') as f:
            f.write(raw_data)
        
        logger.error(f"Could not convert ESP32CAM RAW file {filename}. Saved debug file to {debug_path}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to convert ESP32CAM RAW file {filename}: {str(e)}")
        return None

# Camera RAW to JPG conversion function
def convert_camera_raw_to_jpg(raw_data, filename, quality=95):
    """
    Convert professional camera RAW image data to JPG format
    
    Args:
        raw_data (bytes): RAW image data
        filename (str): Original filename for logging purposes
        quality (int): JPG quality (1-100)
        
    Returns:
        PIL.Image: Image object or None if conversion failed
    """
    try:
        # Create a temporary file to save the RAW data
        temp_raw_path = os.path.join(TEMP_DIR, f"temp_{os.path.basename(filename)}")
        
        # Write raw data to the temporary file
        with open(temp_raw_path, 'wb') as f:
            f.write(raw_data)
        
        # Process the RAW file
        with rawpy.imread(temp_raw_path) as raw:
            rgb = raw.postprocess()
        
        # Clean up temporary file
        os.remove(temp_raw_path)
        
        # Convert to JPG and store in memory
        jpg_buffer = BytesIO()
        imageio.imwrite(jpg_buffer, rgb, format='jpg', quality=quality)
        jpg_buffer.seek(0)
        
        # Create a PIL Image from the JPG data
        img = Image.open(jpg_buffer)
        
        logger.info(f"Successfully converted camera RAW file {filename} to JPG")
        return img
        
    except Exception as e:
        logger.error(f"Failed to convert camera RAW file {filename}: {str(e)}")
        return None

# Load the TFLite model
model_path = 'C:/Users/abhij/OneDrive/Desktop/project/model.tflite'  # Update with your TFLite model path
try:
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input shape
    input_shape = input_details[0]['shape']
    logger.info(f"Model loaded successfully. Input shape: {input_shape}")
except Exception as e:
    logger.error(f"Failed to load the TFLite model: {e}")
    raise

# Set up a CSV file to store predictions
csv_file = 'predictions.csv'

# Create CSV file with headers if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'prediction', 'timestamp', 'original_file'])

# Track processed files
processed_files = set()

# Helper function to load processed files from CSV
def load_processed_files():
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) > 0:
                    processed_files.add(row[0])
                    # If there's an original filename (for converted RAW files)
                    if len(row) >= 4 and row[3]:
                        processed_files.add(row[3])
    logger.info(f"Loaded {len(processed_files)} already processed files")

# Helper function to check if file should be processed
def should_process_file(file_name):
    # Skip system files and placeholders
    if any(ignore_file in file_name for ignore_file in IGNORE_FILES):
        return False
    
    # Skip already processed files
    if file_name in processed_files:
        return False
        
    # Check file extension (only process supported image files)
    file_ext = os.path.splitext(file_name.lower())[1]
    if file_ext not in VALID_IMAGE_EXTENSIONS:
        logger.info(f"Skipping unsupported file type: {file_name}")
        return False
        
    return True

# Helper function to fetch image from Supabase
def fetch_image_from_supabase(file_name: str):
    try:
        response = supabase.storage.from_('img').download(file_name)
        
        # Check if this is a RAW image that needs conversion
        file_ext = os.path.splitext(file_name.lower())[1]
        
        if file_ext == ESP32_RAW_EXTENSION:
            # This is an ESP32CAM RAW file
            logger.info(f"Fetched ESP32CAM RAW image {file_name}, converting to JPG")
            return convert_esp32cam_raw_to_jpg(response, file_name)
        elif file_ext in CAMERA_RAW_EXTENSIONS:
            # This is a professional camera RAW file
            logger.info(f"Fetched camera RAW image {file_name}, converting to JPG")
            return convert_camera_raw_to_jpg(response, file_name)
        else:
            # Regular image, open directly
            try:
                img = Image.open(BytesIO(response))
                return img
            except Exception as e:
                logger.error(f"Failed to open image {file_name}: {str(e)}")
                return None
    except Exception as e:
        logger.error(f"Failed to fetch image {file_name}: {str(e)}")
        return None

# Helper function to preprocess the image for TFLite
def preprocess_image(image):
    try:
        # Get expected height and width from input shape
        height = input_shape[1] if len(input_shape) == 4 else 224
        width = input_shape[2] if len(input_shape) == 4 else 224
        
        img = image.resize((width, height))  # Resize to match the model's input shape
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize the image
        
        # Handle different input requirements
        if len(input_shape) == 4:
            # Add batch dimension if needed
            if input_shape[0] == 1 or input_shape[0] is None:
                img = np.expand_dims(img, axis=0)
        
        # Ensure data type matches the model's expected input
        img = img.astype(input_details[0]['dtype'])
        return img
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
        return None

# Class labels for plant disease detection
class_labels = {
    0: "Apple - Apple Scab", 1: "Apple - Black Rot", 2: "Apple - Cedar Apple Rust", 3: "Apple - Healthy",
    4: "Blueberry - Healthy", 5: "Cherry - Powdery Mildew", 6: "Cherry - Healthy",
    7: "Corn - Cercospora Leaf Spot (Gray Leaf Spot)", 8: "Corn - Common Rust", 9: "Corn - Northern Leaf Blight",
    10: "Corn - Healthy", 11: "Grape - Black Rot", 12: "Grape - Esca (Black Measles)",
    13: "Grape - Leaf Blight (Isariopsis Leaf Spot)", 14: "Grape - Healthy",
    15: "Orange - Haunglongbing (Citrus Greening)", 16: "Peach - Bacterial Spot", 17: "Peach - Healthy",
    18: "Pepper Bell - Bacterial Spot", 19: "Pepper Bell - Healthy", 20: "Potato - Early Blight",
    21: "Potato - Late Blight", 22: "Potato - Healthy", 23: "Raspberry - Healthy", 24: "Soybean - Healthy",
    25: "Squash - Powdery Mildew", 26: "Strawberry - Leaf Scorch", 27: "Strawberry - Healthy",
    28: "Tomato - Bacterial Spot", 29: "Tomato - Early Blight", 30: "Tomato - Late Blight",
    31: "Tomato - Leaf Mold", 32: "Tomato - Septoria Leaf Spot",
    33: "Tomato - Spider Mites (Two-spotted Spider Mite)", 34: "Tomato - Target Spot",
    35: "Tomato - Tomato Yellow Leaf Curl Virus", 36: "Tomato - Tomato Mosaic Virus", 37: "Tomato - Healthy"
}
# Helper function to predict using TFLite model
def predict(image):
    try:
        img = preprocess_image(image)
        if img is None:
            return None
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process the output based on your model type
        # For classification models, get the predicted class
        predicted_class_index = np.argmax(output_data)
        
        # Map the class index to the label
        prediction_label = class_labels.get(predicted_class_index, "Unknown Class")
        
        return {"prediction": prediction_label}
    except Exception as e:
        logger.error(f"Failed to predict image: {str(e)}")
        return None

# Function to log the output into CSV
def log_to_csv(file_name, result, original_file=None):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([file_name, result['prediction'], timestamp, original_file or ''])
            
        # Add to processed files set
        processed_files.add(file_name)
        if original_file:
            processed_files.add(original_file)
            
        logger.info(f"Logged prediction for {file_name}: {result['prediction']}")
    except Exception as e:
        logger.error(f"Failed to log to CSV: {str(e)}")

# Function to process and log new images
def process_new_images():
    # Load already processed files
    load_processed_files()
    
    while True:
        try:
            # List all files in the Supabase bucket
            files = supabase.storage.from_('img').list()
            
            # Filter files that should be processed
            new_files = [file['name'] for file in files if should_process_file(file['name'])]
            
            if new_files:
                logger.info(f"Found {len(new_files)} new files to process")
                
                for file_name in new_files:
                    logger.info(f"Processing new file: {file_name}")

                    # Fetch and predict the image
                    image = fetch_image_from_supabase(file_name)
                    if image is None:
                        logger.error(f"Could not fetch image: {file_name}")
                        # Mark as processed to avoid repeated attempts
                        processed_files.add(file_name)
                        continue
                        
                    result = predict(image)
                    if result is None:
                        logger.error(f"Could not predict image: {file_name}")
                        # Mark as processed to avoid repeated attempts
                        processed_files.add(file_name)
                        continue
                    
                    # Determine if this is a converted RAW file
                    original_file = None
                    file_ext = os.path.splitext(file_name.lower())[1]
                    if file_ext in CAMERA_RAW_EXTENSIONS or file_ext == ESP32_RAW_EXTENSION:
                        # This is the original RAW file
                        original_file = file_name
                    
                    # Log the result to CSV
                    log_to_csv(file_name, result, original_file)
            else:
                logger.info("No new files to process")
                
        except Exception as e:
            logger.error(f"Failed to process new images: {str(e)}")
        
        # Wait for some time before checking for new files
        time.sleep(5)  # Check every 30 seconds

# Start the background image processing thread
def start_background_task():
    thread = Thread(target=process_new_images)
    thread.daemon = True
    thread.start()

# Create a Flask app to handle predictions if requested
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_image():
    file_name = request.args.get('image_name')
    if not file_name:
        return jsonify({'error': 'No image name provided'}), 400

    try:
        # Skip system files
        if any(ignore_file in file_name for ignore_file in IGNORE_FILES):
            return jsonify({'error': 'Cannot process this file type'}), 400
            
        # Check if already processed
        if file_name in processed_files:
            # Read the prediction from CSV
            with open(csv_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and len(row) > 0 and row[0] == file_name:
                        return jsonify({"prediction": row[1], "cached": True})
            
        # If not found in CSV or not processed, fetch and predict
        image = fetch_image_from_supabase(file_name)
        if image is None:
            return jsonify({'error': 'Failed to fetch image'}), 500
            
        result = predict(image)
        if result is None:
            return jsonify({'error': 'Failed to predict image'}), 500
        
        # Determine if this is a converted RAW file
        original_file = None
        file_ext = os.path.splitext(file_name.lower())[1]
        if file_ext in CAMERA_RAW_EXTENSIONS or file_ext == ESP32_RAW_EXTENSION:
            # This is the original RAW file
            original_file = file_name
            
        # Log the result to CSV
        log_to_csv(file_name, result, original_file)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to predict image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    esp32_raw_count = sum(1 for file in processed_files if file.lower().endswith(ESP32_RAW_EXTENSION))
    camera_raw_count = sum(1 for file in processed_files if any(file.lower().endswith(ext) for ext in CAMERA_RAW_EXTENSIONS))
    
    return jsonify({
        'processed_files': len(processed_files),
        'esp32cam_raw_files': esp32_raw_count,
        'camera_raw_files': camera_raw_count,
        'model_loaded': True,
        'input_shape': input_shape.tolist() if hasattr(input_shape, 'tolist') else input_shape,
        'supported_formats': VALID_IMAGE_EXTENSIONS
    })

@app.route('/supported_formats', methods=['GET'])
def get_supported_formats():
    return jsonify({
        'regular_images': [ext for ext in VALID_IMAGE_EXTENSIONS if ext not in CAMERA_RAW_EXTENSIONS and ext != ESP32_RAW_EXTENSION],
        'camera_raw_formats': CAMERA_RAW_EXTENSIONS,
        'esp32cam_raw_format': ESP32_RAW_EXTENSION,
        'all_supported': VALID_IMAGE_EXTENSIONS
    })

if __name__ == '__main__':  
    # Create temp directory if it doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    # Start the background task that monitors new images
    start_background_task()
    
    # Run the local server
    app.run(host='0.0.0.0', port=5000)