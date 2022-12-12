

at_detector = Detector(
    families="tag16h5",
    nthreads=6,
    quad_decimate=1.0,
    quad_sigma=1.0,
    refine_edges=10,
    decode_sharpening=0.25
)

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

        rendered_image = draw_tags(image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        cv2.imshow('AprilTag Detect Demo', rendered_image)

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
