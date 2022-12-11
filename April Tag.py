import copy
import time

import cv2
from pupil_apriltags import Detector

at_detector = Detector(
    families="tag16h5",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25
)


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
    """

    def __init__(self,
                 families="tag16h5",
                 nThreads=1,
                 quadDecimate=1.0,
                 quadSigma=0.0,
                 refineEdges=1,
                 decodeSharpening=0.25):
        self.detector = Detector(families=families,
                                 nthreads=nThreads,
                                 quad_decimate=quadDecimate,
                                 quad_sigma=quadSigma,
                                 refine_edges=refineEdges,
                                 decode_sharpening=decodeSharpening)

    # def detectTags(img, maxTags = -1)


# OMG ITS SO MESSY

def main():
    cap = cv2.VideoCapture(0)

    elapsed_time = 0

    while True:
        start_time = time.time()

        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        debug_image = draw_tags(debug_image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        cv2.imshow('AprilTag Detect Demo', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def draw_tags(
        image,
        tags,
        elapsed_time,
):
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

        cv2.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(image,
                "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                cv2.LINE_AA)

    return image


if __name__ == '__main__':
    main()
