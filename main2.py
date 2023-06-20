import cv2
import numpy as np
import streamlit as st
from iris_tracking import process_frame
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av


# class IrisTrackingTransformer(VideoTransformerBase):
#     def __init__(self, threshold):
#         self.threshold = threshold

#     def detect(self, frame):
#         iris_pos, ratio = process_frame(frame, self.threshold)

#         frame_with_info = np.copy(frame)

#         cv2.putText(frame_with_info, f"Iris position: {iris_pos}", (30, 30),
#                     cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(frame_with_info, f"Iris ratio: {ratio:.2f}", (30, 60),
#                     cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)

#         #cv2.putText(frame_with_info,"Iris position and ratio: {}, {}".format(iris_pos, ratio), (30, 30),
#         #           cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)

#         return frame_with_info
st.title("Iris Tracking app")
threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, step=0.01)

def callback(frame):
    img = frame.to_ndarray(format = "bgr24")
    iris_pos, ratio = process_frame(img,threshold)

    cv2.putText(img, f"Iris position: {iris_pos}", (30, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"Iris ratio: {ratio:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)

        #cv2.putText(frame_with_info,"Iris position and ratio: {}, {}".format(iris_pos, ratio), (30, 30),
        #           cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format = "bgr24")

webrtc_streamer(key = "example", video_frame_callback=callback)


# def main():
#     st.title("Iris Tracking")

#     threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, step=0.01)

#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         video_transformer_factory=lambda: IrisTrackingTransformer.detect(threshold)
#     )

#     if not webrtc_ctx.state.playing:
#         st.warning("Waiting for video to start...")

#     if st.button("Stop"):
#         webrtc_ctx.video_receiver.stop()

#     if webrtc_ctx.video_receiver and webrtc_ctx.video_receiver.stopped:
#         st.error("Video receiver has stopped.")


# if __name__ == "__main__":
#     main()
