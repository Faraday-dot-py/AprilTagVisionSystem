import copy
import time

import cv2
from pupil_apriltags import Detector
import numpy as np
from PIL import Image, ImageEnhance
from time import process_time as proc_time

logTimes = True

class AprilTag:
    """
    The AprilTag class detects april tags and will draw parimeters
    around each tag, if requested
    
    String  famlies:            What family of tag to detect                                             - defualt "16h5"
    Integer nthreads:           How many threads to use                                                  - default 1
    Float   quad_decimate:      How much to reduce image resolution (Does not change FOV)                - default 1.0
    Float   quad_sigma:         Intensity of the low-pass blur (average out rapid chanegs in intensity)  - default 0.0
    Integer refine_edges:       How "loose" the edge detection is                                        - default 1
    Float   decode_sharpening:  How much sharpening is done                                              - default 0.25
    
    More information can be found here (uses similar terms):
    https://docs.photonvision.org/en/latest/docs/getting-started/pipeline-tuning/apriltag-tuning.html#target-family 
    """

    def __init__(self,
                 families="tag16h5",
                 nThreads=6,
                 quadDecimate=0.0,
                 quadSigma=0.0,
                 refineEdges=1,
                 decodeSharpening=1.0,
                 cameraChannel = 0):

        self.detector = Detector(families=families,
                                 nthreads=nThreads,
                                 quad_decimate=quadDecimate,
                                 quad_sigma=quadSigma,
                                 refine_edges=refineEdges,
                                 decode_sharpening=decodeSharpening)

        self.cap = cv2.VideoCapture(cameraChannel)

    def createText(self, img, text, x ,y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def detectTags(self, img, maxTags = -1):
        return self.detector.detect(img)

    def draw_tags(self, image, tags):
        for tag in tags:
            tag_family = tag.tag_family
            tag_id = tag.tag_id
            center = tag.center
            corners = tag.corners

            center = (int(center[0]), int(center[1]))
            corner_01 = (int(corners[0][0]), int(corners[0][1]))
            corner_02 = (int(corners[1][0]), int(corners[1][1]))
            corner_03 = (int(corners[2][0]), int(corners[2][1]))
            corner_04 = (int(corners[3][0]), int(corners[3][1]))

            cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

            cv2.line(image, (corner_01[0], corner_01[1]),
                    (corner_02[0], corner_02[1]), (255, 0, 0), 2)
            cv2.line(image, (corner_02[0], corner_02[1]),
                    (corner_03[0], corner_03[1]), (255, 0, 0), 2)
            cv2.line(image, (corner_03[0], corner_03[1]),
                    (corner_04[0], corner_04[1]), (0, 255, 0), 2)
            cv2.line(image, (corner_04[0], corner_04[1]),
                    (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        return image

    def getImage(self):
        # Read the captured image
        ret, image = self.cap.read()

        # Make sure that the image was captured successfully
        if not ret:
            raise SystemExit("Ret not found [Im on line 55 :)]")

        return image;

    def getProcessedImage(self, enhancementFactor = 1):
        
        if logTimes: print("\n-------------------------------------------------------\n")
        
        startTime = time.time()
        image = self.getImage()
        endTime = time.time()
        if logTimes: print(f"Getting the image: {endTime-startTime}")
        
        # Convert to grayscale
        startTime = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        endTime = time.time()
        if logTimes: print(f"Grayscale conversion: {endTime-startTime}")

        # Convert cv2 image to a format pillow can understand
        startTime = time.time()
        im_pil = Image.fromarray(gray)
        endTime = time.time()
        if logTimes: print(f"Pillow conversion: {endTime-startTime}")


        # Create an instance of the image enhancer
        startTime = time.time()
        enhancer = ImageEnhance.Contrast(im_pil)
        endTime = time.time()
        if logTimes: print(f"Image enhancer init: {endTime-startTime}")


        # Enhance image with the wonderful PIL library <3
        startTime = time.time()
        enhanced = enhancer.enhance(enhancementFactor)
        endTime = time.time()
        if logTimes: print(f"Image enhancement: {endTime-startTime}")


        # Convert back to cv2 format
        startTime = time.time()
        im_np = np.asarray(enhanced)
        endTime = time.time()
        if logTimes: print(f"Cv2 conversion {endTime-startTime}")
	
        if logTimes: print("\n-------------------------------------------------------\n")

        return im_np

    def main(self):
        while True:
            sTime = time.time()
            total = 0

            startTime1 = time.time()
            img = self.getProcessedImage()
            endTime1 = time.time()
            total += (endTime1-startTime1)
            if logTimes: print(f"Total of getProcessedImage --> {endTime1-startTime1}")

            startTime1 = time.time()
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            endTime1 = time.time()
            total += (endTime1-startTime1)
            if logTimes: print(f"Total of waitkey --> {endTime1-startTime1}")

            startTime1 = time.time()
            self.createText(img, "FPS: " + str(round(1.0 / (time.time() - sTime), 2)), 10, 30)
            endTime1 = time.time()
            total += (endTime1-startTime1)
            if logTimes: print(f"Total of createText --> {endTime1-startTime1}")

            startTime1 = time.time()
            tags = self.detectTags(img)
            endTime1 = time.time()
            total += (endTime1-startTime1)
            if logTimes: print(f"Total of detectTags --> {endTime1-startTime1}")

            startTime1 = time.time()
            rendered = self.draw_tags(img, tags)
            endTime1 = time.time()
            total += (endTime1-startTime1)
            if logTimes: print(f"Total of draw_tags --> {endTime1-startTime1}") 
            
            startTime1 = time.time()
            cv2.imshow("April Tag Detection", rendered)
            endTime1 = time.time()  
            total += (endTime1-startTime1)          
            if logTimes: print(f"Total of image show --> {endTime1-startTime1}")

            if logTimes: print(f"Total process time --> {time.time() - sTime}, adds to --> {total}")
        self.cap.release(0)
        cv2.destroyAllWindows()

        

detector = AprilTag()    

if __name__ == "__main__":
    detector.main()
