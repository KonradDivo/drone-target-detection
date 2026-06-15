#include "ObjectDetector.hpp"

// Implémentation du Constructeur de la classe
ObjectDetector::ObjectDetector(const std::string& modelTxt, const std::string& modelBin) {
    m_confidenceThreshold = 0.5f; // On n'affiche que les objets sûrs à plus de 50%

    // Liste officielle des 21 classes gérées par MobileNet-SSD
    m_classes = {"background", "aeroplane", "bicycle", "bird", "boat",
                 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                 "dog", "horse", "motorbike", "person", "pottedplant",
                 "sheep", "sofa", "train", "tvmonitor"};

    // Chargement du réseau de neurones via le module DNN d'OpenCV
    m_net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    
    // Optimisation pour le processeur (CPU)
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// Implémentation de la détection et du dessin
void ObjectDetector::detectAndDraw(cv::Mat& frame) {
    // 1. Transformer l'image brute en "Blob" (format d'entrée requis par l'IA)
    // MobileNet attend une image redimensionnée en 300x300 pixels
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);
    
    m_net.setInput(inputBlob);
    
    // 2. Lancer la propagation avant (L'IA effectue ses milliards de calculs matriciels)
    cv::Mat detection = m_net.forward();

    // 3. Analyser la matrice de sortie retournée par l'IA
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);

        // Si l'IA est sûre d'elle, on extrait les coordonnées de l'objet
        if (confidence > m_confidenceThreshold) {
            size_t classId = static_cast<size_t>(detectionMat.at<float>(i, 1));
            
            // Calculer les vraies dimensions du rectangle par rapport à la taille de l'écran
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop   = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop   = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            cv::Rect objectRect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);

            // 4. Dessiner le rectangle et le texte sur l'image du drone
            cv::rectangle(frame, objectRect, cv::Scalar(0, 255, 0), 2);
            std::string label = m_classes[classId] + ": " + std::to_string(static_cast<int>(confidence * 100)) + "%";
            cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
    }
}
