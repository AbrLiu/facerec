import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph


FRGraph = FaceRecGraph();
face_detect = MTCNNDetect(FRGraph, scale_factor=2); 
aligner = AlignCustom();
frame=cv2.imread('srcImg.png')
rects, landmarks = face_detect.detect_face(frame, 80);
for (i, rect) in enumerate(rects):
   aligned_frame, pos = aligner.align(160, frame, landmarks[i]);
   cv2.imshow(' ',aligned_frame)
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
   exit()
