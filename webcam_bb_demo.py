import tensorflow as tf
import cv2
import time
import argparse
import random

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=300)
parser.add_argument('--cam_height', type=int, default=300)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        i = 0
        start = time.time()
        frame_count = 0
        left_center_cor = (0,0)
        right_center_cor = (0,0)
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            overlay_image = posenet.draw_keypoints(
                display_image, pose_scores, keypoint_scores[:, 9:11], keypoint_coords[:, 9:11, :],
                0.15, 0.1)
            
            if i == 100:
                i = 0
                offset = random.randint(-50, 50)
                left_center_cor = (int(keypoint_coords[:, 0, :][0][1])+100, int(keypoint_coords[:, 0, :][0][0])+offset)
                right_center_cor = (int(keypoint_coords[:, 1, :][0][1]-100), int(keypoint_coords[:, 1, :][0][0])-offset)
            out_img = cv2.circle(overlay_image, left_center_cor, 10, (0, 0, 255), -1) 
            out_img = cv2.circle(out_img, right_center_cor, 10, (0, 255, 255), -1) 
            i += 1
            cv2.imshow('Basket ball workout app', out_img)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()