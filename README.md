# 🛸 Embedded Drone Object Detection (OpenCV DNN & MobileNet-SSD)

This repository implements a lightweight, high-performance object detection subsystem written in modern cross-platform C++ (C++17/20). Designed for autonomous edge drones, the software processes flight logs (pre-recorded video feeds) through a Deep Neural Network (DNN) to identify, classify, and track 21 categories of everyday objects (vehicles, pedestrians, obstacles) in real time without any cloud dependencies.

---

## 🎯 Architectural Engineering Choices

*   **100% Native Portability:** Zero external OS-specific graphical dependencies (such as Windows API or complex GUI frameworks like Qt). The application uses standard CLI entry parameters (`argc`/`argv`), making it completely portable and ready to be compiled natively on a Windows station or cross-compiled for an embedded Linux flight computer (e.g., Raspberry Pi, NVIDIA Jetson).
*   **Encapsulated Logic (OOP):** All deep learning operations—blob normalization, forward propagation, and matrix output parsing—are fully self-contained inside a robust, reusable `ObjectDetector` class.
*   **Hardware Efficiency:** Uses Google’s **MobileNet-SSD** network architecture. This model utilizes depthwise separable convolutions to achieve high frame rates (FPS) on host CPUs, heavily minimizing the power consumption of the drone's onboard battery.

---

## 🗂️ Project Structure

*   `ObjectDetector.hpp` : Class interface defining the DNN network structure, object classes, and parameters.
*   `main.cpp` : Orchestrates the video frame decoding loop, calls the detector, and manages standard I/O arguments.
*   `model/` : Directory reserved for the neural network assets.
    *   `MobileNetSSD_deploy.prototxt` : Caffe network architecture configuration file.
    *   *(Ignored)* `MobileNetSSD_deploy.caffemodel` : Trained network weights (binary format, excluded via `.gitignore` due to file size constraints).

---

## ⚙️ Dependencies & Local Compilation

### 1. Requirements
*   **Compiler:** GCC / MinGW (32-bit `i686` or 64-bit `x86_64` aligned with your environment).
*   **Library:** OpenCV 4.x (Compiled with the `-DBUILD_opencv_python=OFF` optimization flag to focus strictly on the native C++ API).

### 2. IDE Layout (Code::Blocks 25.03 Configuration)
To properly compile the project, link your local OpenCV installation inside Code::Blocks:
1. Go to **Project** -> **Build options** -> **Search directories**.
2. **Compiler:** Add the path to `.../install/include/`
3. **Linker:** Add the path to `.../install/[arch]/mingw/lib/`
4. Go to **Linker settings** -> **Link libraries** and add the static archive files in this **strict hierarchical order**:
   * `libopencv_highgui4120.a`
   * `libopencv_imgcodecs4120.a` *(required for video/image decoding)*
   * `libopencv_imgproc4120.a`
   * `libopencv_core4120.a` *(must remain at the bottom as the foundational dependency)*

---

## 🚀 Execution & Command-Line Usage

Since the system targets headless robotic platforms, you pass the video log path directly to the binary at execution:

```bash
# General terminal syntax
./drone_object_detection.exe path/to/your/flight_log.mp4
```

### Testing inside Code::Blocks without a terminal:
1. Go to the top menu: **Project** -> **Set programs' arguments...**
2. In the **Program arguments** input field, type your local test file name (e.g., `drone_flight.mp4`).
3. Press **F9** to automatically compile and feed the video path straight into the application.

---

## 🛡️ Asset Management & Deployment Notes

The binary model weights (`.caffemodel`) and test video feeds (`.mp4`) are excluded from the online Git tree tracking system via `.gitignore` to maintain a light footprint. Developers cloning this repository must download the corresponding architecture files manually from the official MobileNet-SSD repositories before executing the pipeline.
