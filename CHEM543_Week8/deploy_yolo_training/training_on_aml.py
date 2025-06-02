# import necessary packages
from azureml.core import Workspace, Dataset, Run
import os, tempfile, tarfile, yaml

# Make a temporary directory and mount molecule image dataset
print("Create temporary directory...")
mounted_path = './tmp'
print('Temporary directory made at' + mounted_path)

# Get the molecule dataset from the current workspace, and download it
print("Fetching dataset")
ws = Run.get_context().experiment.workspace
dataset = Dataset.get_by_name(ws, name='molecule_image_yolov5')
print("Download dataset")
dataset.download(mounted_path,overwrite=True)
print("Check that the tar file is there:")
print(os.listdir(mounted_path))
print("molecule_images dataset download done")

# untar all files under this directory, 
for file in os.listdir(mounted_path):
    if file.endswith('.tar'):
        print(f"Found tar file: {file}")
        tar = tarfile.open(os.path.join(mounted_path, file))
        tar.extractall()
        tar.close()
    
print("")
print("Content of the molecule_images folder:")
molecule_images_folder = os.path.join(".","molecule_images")
print(os.listdir(molecule_images_folder))

# this is needed for container
os.system('apt-get install -y python3-opencv')
    
print("Current Directory:")
print(os.getcwd())
print()
    
print("Cloning yolov5")
os.system('git clone https://github.com/ultralytics/yolov5')
print("Check content of '.' folder:")
print(os.listdir('.'))

# Let's check that pytorch recognizes the GPU
import torch
print(f"yolov5 enviroment setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Generate yaml config file for run on Azure GPU
yolo_yaml = os.path.join('.', 'molecule_detection_yolov5.yaml')

tag = 'molecule' 
tags = [tag]
with open(yolo_yaml, 'w') as yamlout:
    yaml.dump(
        {'train': os.path.join('../molecule_images','train'),
        'val': os.path.join('../molecule_images','val'),
        'nc': len(tags),
        'names': tags},
        yamlout,
        default_flow_style=None,
        sort_keys=False
    )

# Let's copy the yaml file to the "./outputs" folder we well so we can find it in the logs of the experiment once it's complete
os.system('cp ./molecule_detection_yolov5.yaml ./outputs/molecule_detection_yolov5.yaml')

os.system('python yolov5/train.py --img 640 --batch 16 --epochs 100 --data ./molecule_detection_yolov5.yaml --weights yolov5s.pt')

os.system('python yolov5/detect.py --weights ./yolov5/runs/train/exp/weights/best.pt --iou 0.05 --save-txt --source ./molecule_images/test/images/')

# Copy to the outputs folder so that the results get saved as part of the AML run
os.system('cp -r ./yolov5/runs ./outputs/')
