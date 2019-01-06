
# coding: utf-8

# In[21]:


UBIT = 'nramanat'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2


# In[22]:


Tsucuba_Left_RGB = cv2.imread("tsucuba_left.png")
Tsucuba_Right_RGB = cv2.imread("tsucuba_right.png")
Tsucuba_Left = cv2.imread("tsucuba_left.png",0)
Tsucuba_Right = cv2.imread("tsucuba_right.png",0)
SIFT = cv2.xfeatures2d.SIFT_create()
KeyPoints_Tsucuba_Left, Descriptor_Tsucuba_Left = SIFT.detectAndCompute(Tsucuba_Left,None)
KeyPoints_Tsucuba_Right, Descriptor_Tsucuba_Right = SIFT.detectAndCompute(Tsucuba_Right,None)


# In[23]:


cv2.imwrite("task2_sift1.jpg",cv2.drawKeypoints(Tsucuba_Left_RGB,KeyPoints_Tsucuba_Left,None))
cv2.imwrite("task2_sift2.jpg",cv2.drawKeypoints(Tsucuba_Right_RGB,KeyPoints_Tsucuba_Right,None))


# In[24]:


BruteForceMatcher = cv2.BFMatcher()
KeyPointMatches = BruteForceMatcher.knnMatch(Descriptor_Tsucuba_Left,Descriptor_Tsucuba_Right,k=2)
GoodMatchesList = [m for m,n in KeyPointMatches if m.distance < (0.75*n.distance) ]
Matches = cv2.drawMatches(Tsucuba_Left_RGB,KeyPoints_Tsucuba_Left,Tsucuba_Right_RGB,KeyPoints_Tsucuba_Right,GoodMatchesList,None,flags=2)
cv2.imwrite("task2_matches_knn.jpg",Matches)


# In[25]:


SourcePoints = np.float32([KeyPoints_Tsucuba_Left[m.queryIdx].pt for m in GoodMatchesList]).reshape(-1,1,2)
DestinationPoints = np.float32([KeyPoints_Tsucuba_Right[m.trainIdx].pt for m in GoodMatchesList]).reshape(-1,1,2)
SourcePoints = np.int32(np.round(SourcePoints))
DestinationPoints = np.int32(np.round(DestinationPoints))
F, Mask = cv2.findFundamentalMat(SourcePoints,DestinationPoints,cv2.RANSAC)
print(F)


# In[26]:


GoodMatches_Inliers = []
Mask_Inliers = []
for i in range(0,len(Mask)-1):
    if(Mask[i][0] == 1):
        GoodMatches_Inliers.append(GoodMatchesList[i])
        Mask_Inliers.append(Mask[i])

GoodMatches_Inliers = np.random.choice(GoodMatches_Inliers, size=10, replace=False)
InlierSourcePoints = np.float32([KeyPoints_Tsucuba_Left[m.queryIdx].pt for m in GoodMatches_Inliers]).reshape(-1,1,2)
InlierDestinationPoints = np.float32([KeyPoints_Tsucuba_Right[m.trainIdx].pt for m in GoodMatches_Inliers]).reshape(-1,1,2)
InlierSourcePoints = np.int32(np.round(InlierSourcePoints))
InlierDestinationPoints = np.int32(np.round(InlierDestinationPoints))
Matches = cv2.drawMatches(Tsucuba_Left_RGB,KeyPoints_Tsucuba_Left,Tsucuba_Right_RGB,KeyPoints_Tsucuba_Right,GoodMatches_Inliers,None,flags=2)


# In[27]:


DestinationLines = cv2.computeCorrespondEpilines(InlierSourcePoints, 1, F)
DestinationLines = DestinationLines.reshape(-1,3)

SourceLines = cv2.computeCorrespondEpilines(InlierDestinationPoints, 2, F)
SourceLines = SourceLines.reshape(-1,3)

height,width = Tsucuba_Left.shape

for SourceLine, DestinationLine,sourcePoint,destinationPoint in zip(SourceLines, DestinationLines, InlierSourcePoints, InlierDestinationPoints):
    color = tuple(np.random.randint(0,255,3).tolist())
    x0, y0 = map(int, [0, -DestinationLine[2]/DestinationLine[1]])
    x1, y1 = map(int, [width, -(DestinationLine[2]+DestinationLine[0]*width)/DestinationLine[1]])
    Tsucuba_Right_RGB = cv2.line(Tsucuba_Right_RGB, (x0, y0), (x1, y1), tuple([0,255,0]))
    Tsucuba_Right_RGB = cv2.circle(Tsucuba_Right_RGB, tuple(destinationPoint[0]), 5, color, -1)
    Tsucuba_Left_RGB = cv2.circle(Tsucuba_Left_RGB, tuple(sourcePoint[0]), 5, color, -1)

    x0, y0 = map(int, [0, -SourceLine[2]/SourceLine[1]])
    x1, y1 = map(int, [width, -(SourceLine[2]+SourceLine[0]*width)/SourceLine[1]])
    Tsucuba_Left_RGB = cv2.line(Tsucuba_Left_RGB, (x0, y0), (x1, y1), tuple([0,255,0]))
    Tsucuba_Left_RGB = cv2.circle(Tsucuba_Left_RGB, tuple(sourcePoint[0]), 5, color, -1)
    Tsucuba_Right_RGB = cv2.circle(Tsucuba_Right_RGB, tuple(destinationPoint[0]), 5, color, -1)


# In[28]:


cv2.imwrite("task2_epi_right.jpg",Tsucuba_Right_RGB)


# In[29]:


cv2.imwrite("task2_epi_left.jpg",Tsucuba_Left_RGB)


# In[30]:


num_disp = 48
window_size = 17
stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)
stereo.setMinDisparity(16)
stereo.setNumDisparities(num_disp)
stereo.setBlockSize(window_size)
stereo.setDisp12MaxDiff(2)
stereo.setUniquenessRatio(6)
stereo.setSpeckleRange(25)
stereo.setSpeckleWindowSize(150)
Disparity = np.float32((stereo.compute(Tsucuba_Left, Tsucuba_Right)))*(1.0 / 16.0)
NormalizedDisparity = (Disparity - Disparity.min()) / (Disparity.max() - Disparity.min())*255

cv2.imwrite("task2_disparity.jpg",NormalizedDisparity)

