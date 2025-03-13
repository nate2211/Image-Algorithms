#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

cv::Scalar getSmartColor(const cv::Mat& image, int x, int y) {
    std::map<int, int> colorHistogram;
    int totalPixels = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
                int color = static_cast<int>(image.at<uchar>(ny, nx));
                colorHistogram[color]++;
                totalPixels++;
            }
        }
    }

    int weightedSum = 0, count = 0;
    for (auto& it : colorHistogram) {
        weightedSum += it.first * it.second;
        count += it.second;
    }

    int chosenColor = (count > 0) ? weightedSum / count : 0;
    return cv::Scalar(chosenColor);
}

void applyEdgeDetection(const std::string& filename) {
    int lowThreshold, highThreshold;
    std::cout << "Enter low threshold for Canny Edge Detection: ";
    std::cin >> lowThreshold;
    std::cout << "Enter high threshold for Canny Edge Detection: ";
    std::cin >> highThreshold;

    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    cv::Mat edges;
    cv::Canny(image, edges, lowThreshold, highThreshold);
    cv::imwrite("edges.png", edges);
    cv::imshow("Edge Detection", edges);
    cv::waitKey(0);
}
void applyKMeansSegmentation(const std::string& filename) {
    int k;
    std::cout << "Enter the number of clusters (k) for k-means: ";
    std::cin >> k;

    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    cv::Mat samples = image.reshape(1, image.rows * image.cols);
    samples.convertTo(samples, CV_32F);

    cv::Mat labels, centers;
    cv::kmeans(samples, k, labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
        3, cv::KMEANS_PP_CENTERS, centers);

    centers = centers.reshape(3, centers.rows);
    cv::Mat segmented(image.size(), image.type());

    for (int i = 0; i < samples.rows; i++) {
        int clusterIdx = labels.at<int>(i, 0);
        segmented.at<cv::Vec3b>(i / image.cols, i % image.cols) = centers.at<cv::Vec3b>(clusterIdx, 0);
    }

    cv::imwrite("segmented.png", segmented);
    cv::imshow("K-Means Segmentation", segmented);
    cv::waitKey(0);
}

void warpAndConvertToBW(const std::string& filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    int width = image.cols, height = image.rows;
    std::cout << "Enter width of processed image: ";
    std::cin >> width;
    std::cout << "Enter height of processed image: ";
    std::cin >> height;
    double blurSigma;
    std::cout << "Enter Gaussian blur sigma value: ";
    std::cin >> blurSigma;
    int lowThreshold, highThreshold;
    std::cout << "Enter low threshold for Canny Edge Detection: ";
    std::cin >> lowThreshold;
    std::cout << "Enter high threshold for Canny Edge Detection: ";
    std::cin >> highThreshold;
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));

    cv::Mat edges;
    cv::GaussianBlur(resizedImage, resizedImage, cv::Size(5, 5), blurSigma);
    cv::Canny(resizedImage, edges, lowThreshold, highThreshold);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> warpMultiplier(1.5, 5.5);

    cv::Mat warpedImage(height, width, CV_8UC1, cv::Scalar(255));

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                double warpStrength = warpMultiplier(gen);
                int newX = x + static_cast<int>(warpStrength);
                int newY = y + static_cast<int>(warpStrength);

                if (newX >= 1 && newX < width - 1 && newY >= 1 && newY < height - 1) {
                    cv::line(warpedImage, cv::Point(x, y), cv::Point(newX, newY), getSmartColor(resizedImage, newX, newY), 1);
                }
            }
        }
    }

    cv::imwrite("warped_lines.png", warpedImage);
    cv::imshow("Warped Image", warpedImage);
    cv::waitKey(0);
}

void applyRegionGrowing(const std::string& filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    int seedX, seedY, threshold;
    std::cout << "Enter seed point (x y): ";
    std::cin >> seedX >> seedY;
    std::cout << "Enter intensity threshold: ";
    std::cin >> threshold;

    cv::Mat segmented = cv::Mat::zeros(image.size(), CV_8UC1);
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(cv::Point(seedX, seedY));
    int seedValue = image.at<uchar>(seedY, seedX);

    while (!pixelQueue.empty()) {
        cv::Point p = pixelQueue.front();
        pixelQueue.pop();

        if (segmented.at<uchar>(p.y, p.x) == 0 && abs(image.at<uchar>(p.y, p.x) - seedValue) <= threshold) {
            segmented.at<uchar>(p.y, p.x) = 255;

            if (p.x > 0) pixelQueue.push(cv::Point(p.x - 1, p.y));
            if (p.x < image.cols - 1) pixelQueue.push(cv::Point(p.x + 1, p.y));
            if (p.y > 0) pixelQueue.push(cv::Point(p.x, p.y - 1));
            if (p.y < image.rows - 1) pixelQueue.push(cv::Point(p.x, p.y + 1));
        }
    }

    cv::imwrite("region_growing.png", segmented);
    cv::imshow("Region Growing Segmentation", segmented);
    cv::waitKey(0);
}

void applyGraphCutSegmentation(const std::string& filename) {
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    cv::Rect rect;
    std::cout << "Enter ROI for Graph Cut (x y width height): ";
    std::cin >> rect.x >> rect.y >> rect.width >> rect.height;

    cv::Mat mask, bgModel, fgModel;
    mask.create(image.size(), CV_8UC1);
    mask.setTo(cv::GC_PR_BGD);
    mask(rect).setTo(cv::GC_PR_FGD);

    cv::grabCut(image, mask, rect, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);
    cv::Mat result;
    cv::compare(mask, cv::GC_PR_FGD, result, cv::CMP_EQ);
    cv::Mat segmented(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    image.copyTo(segmented, result);

    cv::imwrite("graph_cut_segmented.png", segmented);
    cv::imshow("Graph Cut Segmentation", segmented);
    cv::waitKey(0);
}


void applyAdaptiveThresholding(const std::string& filename) {
    int blockSize, C;
    std::cout << "Enter block size for adaptive thresholding (odd number > 1): ";
    std::cin >> blockSize;
    std::cout << "Enter constant C to subtract from mean: ";
    std::cin >> C;

    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    cv::Mat thresholded;
    cv::adaptiveThreshold(image, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, C);

    cv::imwrite("adaptive_thresholded.png", thresholded);
    cv::imshow("Adaptive Thresholding", thresholded);
    cv::waitKey(0);
}


void applyWatershedSegmentation(const std::string& filename) {
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return;
    }

    cv::Mat gray, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    cv::Mat distTransform;
    cv::distanceTransform(binary, distTransform, cv::DIST_L2, 5);
    cv::normalize(distTransform, distTransform, 0, 1.0, cv::NORM_MINMAX);
    cv::threshold(distTransform, distTransform, 0.5, 1.0, cv::THRESH_BINARY);

    cv::Mat sureForeground;
    distTransform.convertTo(sureForeground, CV_8U, 255);
    cv::Mat sureBackground;
    cv::dilate(binary, sureBackground, cv::Mat(), cv::Point(-1, -1), 3);
    sureBackground = 255 - sureBackground;

    cv::Mat markers;
    cv::connectedComponents(sureForeground, markers);
    markers += 1;
    markers.setTo(0, sureBackground);

    cv::watershed(image, markers);
    cv::Mat segmented = cv::Mat::zeros(image.size(), CV_8UC3);

    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) == -1) {
                segmented.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
            else {
                segmented.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j);
            }
        }
    }

    cv::imwrite("watershed_segmented.png", segmented);
    cv::imshow("Watershed Segmentation", segmented);
    cv::waitKey(0);
}


int main() {
    std::string filename;
    while (true) {
        std::cout << "Enter the path to an image file: ";
        std::getline(std::cin, filename);
        if (!cv::imread(filename, cv::IMREAD_GRAYSCALE).empty()) break;
        std::cerr << "Error: Could not load image! Please try again." << std::endl;
    }

    int choice;
    while (true) {
        std::cout << "Select processing method:\n";
        std::cout << "1. Edge Detection\n";
        std::cout << "2. Warped Edge Effect\n";
        std::cout << "3. K-Means Segmentation\n";
        std::cout << "4. Region growing Segmentation\n";
        std::cout << "5. Graph Cut\n";
        std::cout << "6. Adaptive Threshold\n";
        std::cout << "7. Exit\n";
        std::cout << "Enter your choice: ";
        std::cin >> choice;

        switch (choice) {
        case 1:
            applyEdgeDetection(filename);
            break;
        case 2:
            warpAndConvertToBW(filename);
            break;
        case 3:
            applyKMeansSegmentation(filename);
            break;
        case 4:
            applyRegionGrowing(filename);
            break;
        case 5:
            applyGraphCutSegmentation(filename);
            break;
        case 6:
            applyAdaptiveThresholding(filename);
            break;
        case 7:
            return 0;

        default:
            std::cerr << "Invalid choice! Please select again." << std::endl;
        }
    }
}
