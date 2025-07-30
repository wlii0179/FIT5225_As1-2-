from ultralytics import YOLO
import cv2
import logging

log = logging.getLogger(__name__)

def predict(model, src_img, dst_img):
    log.info(f"Predicting with source image: {src_img}, output to {dst_img}")

    img = cv2.imread(src_img)
    if img is None:
        log.error(f"Error: Could not read image at {src_img}")
        exit(1)

    results = model(src_img)
    print(results)
    # Process the results
    for result in results:
        keypoints = result.keypoints  # Keypoints object
        #print(keypoints)

        if keypoints is not None and len(keypoints.xy) > 0: #Check if keypoints are detected
            # Get keypoints coordinates (x, y) - Shape: (num_people, num_keypoints, 2)
            keypoints_xy = keypoints.xy[0]  # Assuming only one person detected for simplicity. Adapt if multiple people are expected.
            #print(keypoints_xy)
            # Get confidence scores for each keypoint - Shape: (num_people, num_keypoints)
            keypoints_conf = keypoints.conf[0]
            #print(keypoints_conf)

            # Plot keypoints on the image
            for k, (x, y) in enumerate(keypoints_xy):
                if keypoints_conf[k] > 0.5: # Only plot if confidence is above a threshold (e.g., 0.5)
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles
                    cv2.putText(img, str(k), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red labels

            # Draw lines connecting the keypoints (optional - customize as needed)
            # Example: Connect shoulders and elbows
            # You'll need to define the connections based on the model's keypoint indices
            # Example connections for COCO format (adjust if using a different dataset):
            connections = [[5, 6], [5, 11], [6, 12], [11, 12]] # Example connections (left shoulder - right shoulder, left shoulder - left hip, right shoulder - right hip, left hip - right hip)

            for connection in connections:
                p1 = (int(keypoints_xy[connection[0]][0]), int(keypoints_xy[connection[0]][1]))
                p2 = (int(keypoints_xy[connection[1]][0]), int(keypoints_xy[connection[1]][1]))

                if keypoints_conf[connection[0]] > 0.5 and keypoints_conf[connection[1]] > 0.5: #only draw the line if the keypoints are above the confidence level
                    cv2.line(img, p1, p2, (0, 0, 255), 2)  # Red lines

            # Save the image
            cv2.imwrite(dst_img, img)
        else:
            print("No keypoints were detected in the image.")
    return results

def main():
    try:
        log.info("Loading YOLO pose detection model...")
        model = YOLO('./yolo11l-pose.pt')
        # Load a pretrained YOLOv11x pose model
        log.info("Model is loaded.")
        # Predict on an image
        image_path = './test.jpg'  # Replace with your image path
        output_image = 'test_with_keypoints.jpg'
        result = predict(model,image_path,output_image)
        log.info(result)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()