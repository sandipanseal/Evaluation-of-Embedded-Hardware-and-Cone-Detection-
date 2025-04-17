# #  Evaluation of Embedded Hardware and Cone Detection

This project implements a real-time, GPU-accelerated computer vision pipeline for cone detection in autonomous driving scenarios. It runs on the AMD Kria KV260 board using OpenCV and OpenCL, and is optimized to detect and label cones with high accuracy under resource-constrained embedded environments.

## Project Overview

Cone detection is critical in applications such as autonomous racing and navigation. Traditional CPU or FPGA-based systems often fall short in terms of flexibility, ease of development, and real-time performance. This project addresses those limitations by:
- Leveraging the embedded **ARM Mali GPU** on the **AMD Kria KV260** board
- Accelerating key steps using **OpenCL** with **OpenCV** integration
- Implementing a full pipeline from color segmentation to bounding box labeling

##  Objectives

- Offload CPU and use embedded GPU for real-time processing
- Accurately detect colored cones (blue, orange, yellow)
- Compare performance across CPU and GPU platforms
- Reduce false positives using robust filtering techniques

##  Hardware & Software Stack

### Hardware
-  **AMD Kria KV260** (Zynq UltraScale+ MPSoC with ARM Cortex-A53, Mali-400 GPU)
-  Local Machine: Intel Core i5-7300HQ, NVIDIA GTX 1050

### Software
- Python 3.12
- OpenCV 4.11.0
- OpenCL 2.0
- Visual Studio Code

##  Features

- GPU-accelerated processing using OpenCL
- HSV color segmentation for blue, orange, and yellow cones
- Morphological filtering to reduce noise
- Region of Interest (ROI) optimization
- Template matching with multi-scale cone templates
- Non-Maximum Suppression (NMS)
- False Positive Filtering based on aspect ratio and size
- Color labeling and coverage threshold filtering
- Real-time visualization of detection with bounding boxes

##  Sample Workflow

1. Load image and cone template
2. Convert to HSV and segment cone colors
3. Apply morphological operations
4. Extract ROI and perform template matching
5. Apply NMS and false positive filtering
6. Assign color labels
7. Visualize and save results

---

##  Folder Structure
-- cone_hsv.py # Main script for cone detection 
-- HOW_TO_RUN.txt # Run instructions 
-- AMS_LAB_Project_Report.pdf 
-- Embedded_Hardware_and_Cone_Detection.pptx 
-- README.md



##  How to Run

Ensure OpenCV and OpenCL are installed and configured correctly.

```bash
# For GPU acceleration
python cone_hsv.py data --use_gpu

# For CPU execution
python cone_hsv.py data


## Results Summary
* 90%+ detection accuracy
* All cone colors labeled correctly
* Strong scalability potential for parallel workloads

## Applications
*  Formula Student Driverless
*  Autonomous navigation systems
*  Robotics & automation

## Future Enhancements
* Deep learning-based detection
* FPGA-GPU hybrid acceleration
* Parallel image stream processing
* Broader object detection beyond cones

## Contributors
* Abhishek Pannase
* Sandipan Seal
* Rizwan Mohsin
Supervised by: Prof. Benjamin Noack, Moritz Pötzsch
Otto von Guericke University Magdeburg

