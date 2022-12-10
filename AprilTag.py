from pupil_apriltags import Detector

at_detector = Detector(
    famlies = "16h5",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25
)

