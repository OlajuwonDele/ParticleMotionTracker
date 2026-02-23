#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp> 

struct detObject {
    cv::Point2d centre;
    double area;
    std::string colour;
    cv::Rect boundingBox;
    int classID;
    float confidence;
};

struct Bead {
    int ParticleID;
    std::string ParticleColour;
    int FrameNumberOnEnter;
    int FrameNumberOnExit;
    int xPosAtEnter;
    int xPosAtExit;
    cv::Point2d centroid;
    double area;
    bool active = true;
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect trackerBox;       
    int framesLost = 0; 
};

const std::string DEBUG_WINDOW_NAME = "DEBUG WINDOW";
const std::string videoFilePath = "video.avi";
const std::string saveDataFilepath = "outputData.csv";

const std::string yoloModelPath = "custom_dataset_YOLO.onnx"; 

const int yoloInputWidth = 640; 
const int yoloInputHeight = 640;
const float confThreshold = 0.45; 
const float nmsThreshold = 0.2;  

int currentParticleID = 0;
int particleNumber = 0;
std::vector<Bead> inFrameBeads;
std::vector<Bead> outFrameBeads;

const std::array<std::string, 2> classNames = {"black", "white"};

std::vector<std::string> csvData = std::vector<std::string> {
	"ParticleID, ParticleColour, FrameNumberOnEnter, FrameNumberOnExit, xPosAtEnter, xPosAtExit, "
};

bool LoadVideo(cv::VideoCapture& capture);
bool loadYOLO(cv::dnn::Net& net); 
void processVideo(cv::VideoCapture cap);
void detectBeadsYOLO(const cv::Mat &frame, cv::dnn::Net &net, std::vector<detObject> &detectedObjects);
void objectTracker(std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<detObject> &detectedObjects, std::vector<bool> &matches);
void createBeads(int frameNumber, std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<bool> &matches, const std::vector<detObject> &detectedObjects);
void exitBeads(int frameNumber, std::vector<Bead> &inFrameBeads, std::vector<Bead> &outFrameBeads);
void drawDetection(cv::Mat &frame, const std::vector<Bead> &inFrameBeads, int &frameNumber);
void updateData(const std::vector<Bead> &outFrameBeads, std::vector<std::string> &csvData);
bool saveDataAsCSV();

int main() {
    cv::VideoCapture cap;
    if (!LoadVideo(cap)) return -1;
    processVideo(cap);
    saveDataAsCSV();
    return 1;
}


void detectBeadsYOLO(const cv::Mat &frame, cv::dnn::Net &net, std::vector<detObject> &detectedObjects) {
    detectedObjects.clear();
    
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
    net.setInput(blob, "images");

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat res = outputs[0];

    if (res.dims > 2) {
        res = res.reshape(1, res.size[1]); 
    }

    if (res.cols > res.rows) {
        cv::transpose(res, res);
    }

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = (float)frame.cols / yoloInputWidth;
    float y_factor = (float)frame.rows / yoloInputHeight;

    float* data = (float*)res.data;
    int rows = res.rows;       
    int dimensions = res.cols; 

    for (int i = 0; i < rows; ++i) {
        

        cv::Mat scores = res.row(i).colRange(4, dimensions);
        
        cv::Point classIdPoint;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

        if (max_class_score > confThreshold) {
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back((float)max_class_score);
            classIds.push_back(classIdPoint.x);
        }
        
        data += dimensions;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        detObject obj;
        obj.boundingBox = boxes[idx] & cv::Rect(0, 0, frame.cols, frame.rows);
        obj.confidence = confidences[idx];
        obj.classID = classIds[idx];
        
        if (obj.classID < (int)classNames.size()) {
            obj.colour = classNames[obj.classID];
        } else {
            obj.colour = "unknown";
        }

        obj.centre = cv::Point2d(obj.boundingBox.x + obj.boundingBox.width / 2.0, 
                                 obj.boundingBox.y + obj.boundingBox.height / 2.0);
        obj.area = (double)obj.boundingBox.area();
        detectedObjects.push_back(obj);
    }
}


bool loadYOLO(cv::dnn::Net& net) {
    try {
        std::cout << "Loading ONNX model from: " << yoloModelPath << std::endl;
        net = cv::dnn::readNetFromONNX(yoloModelPath);
        
        
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); 
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ONNX Load Error: " << e.what() << std::endl;
        return false;
    }
}

void objectTracker(std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<detObject> &detectedObjects, std::vector<bool> &matches) {
    for (auto& particle : inFrameBeads) {
        particle.active = false;
        bool ok = particle.tracker->update(frame, particle.trackerBox);

        
        if (ok) {
            cv::Point2d trackerCentre(particle.trackerBox.x + particle.trackerBox.width / 2.0,
                                     particle.trackerBox.y + particle.trackerBox.height / 2.0);
            int closestID = -1;
            double closestDist = 10.0; 

            for (size_t i = 0; i < detectedObjects.size(); i++) {
                if (matches[i]) continue;
                if (particle.ParticleColour != detectedObjects[i].colour) continue;

                double dist = cv::norm(trackerCentre - detectedObjects[i].centre);
                if (dist < closestDist) {
                    closestDist = dist;
                    closestID = i;
                }
            }
            if (closestID != -1) {
                particle.centroid = detectedObjects[closestID].centre;
                particle.trackerBox = detectedObjects[closestID].boundingBox;
                particle.tracker->init(frame, particle.trackerBox);
                particle.framesLost = 0;
                particle.active = true;
                matches[closestID] = true;
            } 
            
            else {
                bool insideFrame = (particle.centroid.x > 0 && particle.centroid.x < frame.cols &&
                                    particle.centroid.y > 0 && particle.centroid.y < frame.rows);
    

                particle.centroid = trackerCentre;
                particle.framesLost++;
                particle.active = (insideFrame && particle.framesLost < 5);
            }
        }
    }
}

void createBeads(int frameNumber, std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<bool> &matches, const std::vector<detObject> &detectedObjects){
    int maxHeight = 146;
    for (size_t i = 0; i < detectedObjects.size(); i++) {
        if (!matches[i]) {
            if (detectedObjects[i].centre.y > maxHeight) continue;
            Bead newBead;
            newBead.ParticleID = currentParticleID++;
            newBead.centroid = detectedObjects[i].centre;
            newBead.area = detectedObjects[i].area;
            newBead.FrameNumberOnEnter = frameNumber;
            newBead.xPosAtEnter = static_cast<int>(detectedObjects[i].centre.x);
            newBead.ParticleColour = detectedObjects[i].colour;
            newBead.trackerBox = detectedObjects[i].boundingBox;
            cv::TrackerCSRT::Params params;
            params.psr_threshold = 0.06;
            newBead.tracker = cv::TrackerCSRT::create(params);
            newBead.tracker->init(frame, newBead.trackerBox);
            inFrameBeads.push_back(newBead);
        }
    }
}

void exitBeads(int frameNumber, std::vector<Bead> &inFrameBeads, std::vector<Bead> &outFrameBeads){
    for (auto it = inFrameBeads.begin(); it != inFrameBeads.end();) {
        if (!it->active) {
            it->FrameNumberOnExit = frameNumber;
            it->xPosAtExit = static_cast<int>(it->centroid.x);
            outFrameBeads.push_back(*it);
            it = inFrameBeads.erase(it);
        } else it++;
    }
}

void processVideo(cv::VideoCapture cap){
    cv::Mat frame;
    int frameNumber = 0;
    
    int desiredFPS = 30;
    int delay = 1000 / desiredFPS;
    
    cv::dnn::Net net;
    if (!loadYOLO(net)) return;

    while (true) {
        cap.read(frame);
        if (frame.empty()) break;
        
        std::vector<detObject> detectedObjects;
        
        detectBeadsYOLO(frame, net, detectedObjects);
        std::vector<bool> matches(detectedObjects.size(), false);
        objectTracker(inFrameBeads, frame, detectedObjects, matches);
        createBeads(frameNumber, inFrameBeads, frame, matches, detectedObjects);
        exitBeads(frameNumber, inFrameBeads, outFrameBeads);
        drawDetection(frame, inFrameBeads, frameNumber);
        
        frameNumber++;
        cv::imshow(DEBUG_WINDOW_NAME, frame);
        if (cv::waitKey(delay) == 27) break;
    }
    updateData(outFrameBeads, csvData);
}

bool LoadVideo(cv::VideoCapture& capture) {
    capture = cv::VideoCapture(videoFilePath);
    if (!capture.isOpened()) {
        std::cerr << "Could not load file " << videoFilePath << std::endl;
        return false;
    }
    return true;
}

void updateData(const std::vector<Bead> &outFrameBeads, std::vector<std::string> &csvData){
	for(auto &bead : outFrameBeads){
        if (bead.FrameNumberOnExit - bead.FrameNumberOnEnter > 5) {
            csvData.push_back(
                std::to_string(particleNumber++) + ", " + 
                bead.ParticleColour + "," + 
                std::to_string(bead.FrameNumberOnEnter) + ", " + 
                std::to_string(bead.FrameNumberOnExit) + ", " + 
                std::to_string(bead.xPosAtEnter) + ", " + 
                std::to_string(bead.xPosAtExit));
            }
	}
}

void drawDetection(cv::Mat &frame, const std::vector<Bead> &inFrameBeads, int &frameNumber){
    for (const auto& bead : inFrameBeads) {
        cv::Scalar color = (bead.ParticleColour == "black") ? cv::Scalar(0,0,0) : cv::Scalar(255,255,255);
        cv::rectangle(frame, bead.trackerBox, color, 2);
        std::string label = "ID:" + std::to_string(bead.ParticleID);
        cv::putText(frame, label, bead.centroid + cv::Point2d(0, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 2);
        cv::circle(frame, bead.centroid, 3, cv::Scalar(0,0,255), -1);
    }
    cv::putText(frame, "Frame: " + std::to_string(frameNumber), cv::Point2d(20, 40) , cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
}

bool saveDataAsCSV()
{
	try {
		std::ofstream outputFile;
		outputFile.open(saveDataFilepath);
		for (const std::string& line : csvData) {
			outputFile << line << "\n";
		}

		outputFile.close();

		return true;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		return false;
	}

	return false;
}