from pupil_apriltags import Detector

at_detector = Detector(
    famlies = "16h5",
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
            famlies = "16h5",
            nThreads=1,
            quadDecimate=1.0,
            quadSigma=0.0,
            refineEdges=1,
            decodeSharpening=0.25):
        
        self.detector = Detector(famlies=famlies,
            nthreads=nThreads,
            quad_decimate=quadDecimate,
            quad_sigma=quadSigma,
            refine_edges=refineEdges,
            decode_sharpening=decodeSharpening)

    def detectTags(img, maxTags = -1):
        #Do an assertion here to check if the image provided is indeed an image
        assert img.type == 