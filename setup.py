import os

# Define directories
directories = [
    "cctv_system/app/api",
    "cctv_system/app/core",
    "cctv_system/app/services/level_1_detection",
    "cctv_system/app/services/level_2_tracking",
    "cctv_system/app/services/level_3_recognition",
    "cctv_system/app/models",
    "cctv_system/app/utils",
    "cctv_system/data/known_faces",
    "cctv_system/data/logs",
    "cctv_system/static"
]

# Define files
files = [
    "cctv_system/requirements.txt",
    "cctv_system/.env",
    "cctv_system/app/main.py"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Create files
for file in files:
    with open(file, 'a'):
        os.utime(file, None)

print("Folder structure created successfully.")