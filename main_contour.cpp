#include <iostream>
#include <fstream>

#include "opencv2/tracking.hpp"
#include <opencv2/core.hpp>
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

struct detContour {
    cv::Point2d centre;
    double area;
	std::string colour;
	cv::Rect boundingBox;
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
const std::string saveDataFilepath = "outputDataTrad.csv";


double minContourArea = 65;
double maxContourArea = 900;
int currentParticleID = -2;
std::vector<Bead> inFrameBeads;
std::vector<Bead> outFrameBeads;

std::vector<std::string> csvData = std::vector<std::string> {
	"ParticleID, ParticleColour, FrameNumberOnEnter, FrameNumberOnExit, xPosAtEnter, xPosAtExit, "
};

bool LoadVideo(cv::VideoCapture& capture);
void processVideo(cv::VideoCapture cap);
void contourDetection(const cv::Mat &frame, std::vector<detContour> &detectedContours, std::vector<cv::Mat> &contourMasks);
void contourTracker(std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<detContour> &detectedContours, std::vector<bool> &matches);
void createBeads(int frameNumber, std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<bool> &matches, const std::vector<detContour> &detectedContours, const std::vector<cv::Mat> &contourMasks);
void exitBeads(int frameNumber, std::vector<Bead> &inFrameBeads, std::vector<Bead> &outFrameBeads);
void drawDetection(cv::Mat &frame, const std::vector<Bead> &inFrameBeads, int &frameNumber);
void updateData(const std::vector<Bead> &outFrameBeads, std::vector<std::string> &csvData);
bool saveDataAsCSV();

int main()
{
	cv::VideoCapture cap;
	
	if (!LoadVideo(cap))
		return -1;

	processVideo(cap);


	if (!saveDataAsCSV()) {
		return -1;
	}

	return 1;
}


void contourDetection(const cv::Mat &frame, std::vector<detContour> &detectedContours, std::vector<cv::Mat> &contourMasks){
	cv::Mat grey, maskWhite, maskBlack, maskBackground;
    cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grey, grey, cv::Size(5, 5), 0);
    cv::inRange(grey, 182, 255, maskWhite);
	cv::inRange(grey, 39, 43, maskBlack);

	cv::Mat whiteDilated;
	cv::dilate(maskWhite, whiteDilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(50,50)));
	cv::subtract(maskBlack, whiteDilated, maskBlack);

	cv::Mat openKernelBlack  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1,1));
    cv::Mat closeKernelBlack = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1,1));

	cv::Mat openKernelWhite = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));

	cv::morphologyEx(maskWhite, maskWhite, cv::MORPH_OPEN,  openKernelWhite,  cv::Point(-1,-1), 1);

    cv::imshow("maskWhite", maskWhite);
    cv::imshow("maskBlack", maskBlack);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::pair<cv::Mat, std::string>> maskColours = {
        {maskBlack, "black"},
        {maskWhite, "white"}
    };

	for (auto &currentMask : maskColours){
		cv::Mat mask = currentMask.first;
		std::string colour = currentMask.second;

		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (auto &contour: contours){
			double area = cv::contourArea(contour);	 
			if (area < minContourArea || area > maxContourArea) continue;

			cv::Moments contour_moment = cv::moments(contour);
			cv::Point2d centroid(contour_moment.m10/contour_moment.m00, contour_moment.m01/contour_moment.m00);
			detContour currentContour;
			currentContour.centre = centroid;
			currentContour.area = area;
			currentContour.colour = colour;
			
			cv::Rect boundingBox = cv::boundingRect(contour);
			currentContour.boundingBox = boundingBox;
			detectedContours.push_back(currentContour);

			cv::Mat cMask = cv::Mat::zeros(mask.size(), CV_8UC1);
			cv::drawContours(cMask, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
			contourMasks.push_back(cMask);
		}
	}
}

void contourTracker(std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<detContour> &detectedContours, std::vector<bool> &matches) {
    for (auto& particle : inFrameBeads) {
        particle.active = false;

        bool ok = particle.tracker->update(frame, particle.trackerBox);

        if (ok) {
            cv::Point2d trackerCentre(
                particle.trackerBox.x + particle.trackerBox.width / 2.0,
                particle.trackerBox.y + particle.trackerBox.height / 2.0
            );

            int closestID = -1;
            double closestDist = 60.0; 

            for (int i = 0; i < detectedContours.size(); i++) {
                if (matches[i]) continue;
                if (particle.ParticleColour != detectedContours[i].colour) continue;

                double dist = cv::norm(trackerCentre - detectedContours[i].centre);
                if (dist < closestDist) {
                    closestDist = dist;
                    closestID = i;
                }
            }

            if (closestID != -1) {
                particle.centroid = detectedContours[closestID].centre;
                particle.trackerBox = detectedContours[closestID].boundingBox;
                particle.tracker->init(frame, particle.trackerBox);
                particle.framesLost = 0;
                particle.active = true;
                matches[closestID] = true;
            } else {
				bool insideFrame = (particle.centroid.x > 0 && 
                        particle.centroid.x < frame.cols &&
                        particle.centroid.y > 0 && 
                        particle.centroid.y < frame.rows);

                particle.centroid = trackerCentre;
                particle.framesLost++;
                particle.active = (insideFrame && particle.framesLost < 10);
            }
        }
    }
}

void createBeads(int frameNumber, std::vector<Bead> &inFrameBeads, const cv::Mat &frame, const std::vector<bool> &matches, const std::vector<detContour> &detectedContours, const std::vector<cv::Mat> &contourMasks){
	int maxHeight = 160;
	for (int i = 0; i < detectedContours.size(); i++) {
		if (!matches[i]) {
			if (detectedContours[i].centre.y > maxHeight) continue;
			Bead newBead;
			newBead.ParticleID = currentParticleID++;
			newBead.centroid = detectedContours[i].centre;
			newBead.area = detectedContours[i].area;
			newBead.FrameNumberOnEnter = frameNumber;
			newBead.xPosAtEnter = static_cast<int>(detectedContours[i].centre.x);
			newBead.ParticleColour = detectedContours[i].colour;
			newBead.trackerBox = detectedContours[i].boundingBox;
			newBead.tracker = cv::TrackerCSRT::create();
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
		}
		else it++;
    }
}

void updateData(const std::vector<Bead> &outFrameBeads, std::vector<std::string> &csvData){
	for(auto &bead : outFrameBeads){
		if(bead.FrameNumberOnExit - bead.FrameNumberOnEnter > 5){
			csvData.push_back(
				std::to_string(bead.ParticleID) + ", " + 
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
		if (bead.ParticleID < 0) continue;
		std::string label = "ID: " + std::to_string(bead.ParticleID);
		cv::putText(frame, label, bead.centroid + cv::Point2d(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.6, bead.ParticleColour == "black" ? cv::Scalar(0,0,0) : cv::Scalar(255,255,255));
		cv::circle(frame, bead.centroid, 3, cv::Scalar(0,0,255), -1);
    }
	std::string frame_label = "Frame: " + std::to_string(frameNumber);
	cv::putText(frame, frame_label, cv::Point2d(20, 40) , cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255),2);
}

void processVideo(cv::VideoCapture cap){

	cv::Mat frame;
	int frameNumber = 0;
	int desiredFPS = 60;
	int delay = 1000 / desiredFPS;  
	
	while (cv::waitKey(delay) != 27) {
		cap.read(frame);   
		if (!frame.empty()){
			std::vector<cv::Mat> contourMasks;
			std::vector<detContour> detectedContours;

			contourDetection(frame, detectedContours, contourMasks);
			std::vector<bool> matches(detectedContours.size(), false);
			contourTracker(inFrameBeads, frame, detectedContours, matches);
			createBeads(frameNumber, inFrameBeads, frame, matches, detectedContours, contourMasks);
			exitBeads(frameNumber, inFrameBeads, outFrameBeads);
			drawDetection(frame, inFrameBeads, frameNumber);
			
			frameNumber++;
			cv::imshow(DEBUG_WINDOW_NAME, frame);
		}  
		else break;
		
	}
	updateData(outFrameBeads, csvData);

}

bool LoadVideo(cv::VideoCapture& capture)
{
	capture = cv::VideoCapture(videoFilePath);

	if (!capture.isOpened()) {
		std::cerr << "Could not load file " << videoFilePath;
		return false;
	}

	return true;
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