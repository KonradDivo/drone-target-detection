#include "ObjectDetector.hpp"
#include <iostream>
#include <string>

// int argc : le nombre d'arguments reçus
// char* argv[] : le tableau contenant les textes des arguments
int main(int argc, char* argv[]) {
    std::cout << "=== INITIALISATION DE L'IA DU DRONE (OPENCV DNN) ===" << std::endl;

    // 1. Vérification portable de l'argument
    // Si l'utilisateur n'a pas donné de fichier, on affiche le mode d'emploi
    std::string videoPath = "";
    if (argc < 2) {
        std::cout << "\n[USAGE ERROR] Vous devez spécifier le chemin d'une vidéo." << std::endl;
        std::cout << "Exemple d'exécution dans le terminal :" << std::endl;
        std::cout << "  ./drone_object_detection.exe chemin/vers/votre_video.mp4\n" << std::endl;
        videoPath = "sample_video/14958996_1920_1080_30fps.mp4";
    }
    else{
        videoPath = argv[1];
    }
    // Récupération du premier argument passé au programme
    std::cout << "Chargement du log vidéo : " << videoPath << std::endl;

    std::string protoxtPath = "model/MobileNetSSD_deploy.prototxt";
    std::string caffeModelPath = "model/MobileNetSSD_deploy.caffemodel";

    try {
        ObjectDetector detector(protoxtPath, caffeModelPath);
        
        // 2. Ouverture dynamique et portable du fichier passé en paramètre
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Erreur : Impossible de lire le fichier vidéo sélectionné : " << videoPath << std::endl;
            return -1;
        }

        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            detector.detectAndDraw(frame);

            cv::imshow("Vision Ordinateur Drone - IA", frame);

            if (cv::waitKey(10) == 27) break; // Échap pour quitter
        }
    } 
    catch (const cv::Exception& e) {
        std::cerr << "Erreur critique OpenCV : " << e.what() << std::endl;
    }

    return 0;
}
