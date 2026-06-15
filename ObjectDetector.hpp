#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class ObjectDetector {
private:
    cv::dnn::Net m_net;               // Le réseau de neurones OpenCV DNN
    std::vector<std::string> m_classes; // La liste des noms d'objets connus
    float m_confidenceThreshold;       // Le seuil de détection (ex: 50%)

public:
    // Constructeur : charge les fichiers de l'IA et initialise les classes
    ObjectDetector(const std::string& modelTxt, const std::string& modelBin);

    // Fonction principale : analyse une image et dessine les rectangles des objets détectés
    void detectAndDraw(cv::Mat& frame);
};

#endif
