import cv2
print("cv2:", cv2.__version__)
print("ximgproc exists?", hasattr(cv2, "ximgproc"))
print("createSuperpixelSLIC?", hasattr(cv2.ximgproc, "createSuperpixelSLIC"))