# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

# Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

# Features:
1. Detects the face in an image.
2. Places a stylish sunglass overlay perfectly on the face.
3. Works seamlessly with individual passport-size photos.
4. Customizable for different sunglasses styles or photo types.
# Technologies Used:
Python
1. OpenCV for image processing
2. Numpy for array manipulations
# How to Use:
Clone this repository.
1. Add your passport-sized photo to the images folder.
2. Run the script to see your "cool" transformation!
# Applications:
1. Learning basic image processing techniques.
2. Adding flair to your photos for fun.
3. Practicing computer vision workflows.
4. Feel free to fork, contribute, or customize this project for your creative needs!

# Procedure :

1. Open Jupyter Notebook and create a new Python notebook for the experiment.
2. Import the required Python libraries such as NumPy, Pandas, Matplotlib, and OpenCV as required.
3. Load the input image/data into the Python environment using the appropriate function.
4. Read and inspect the input image to identify its size, dimensions, and other basic properties.
5. Perform the required image processing operations using the suitable OpenCV/Python functions.
6. Display the original and processed images using Matplotlib or OpenCV for comparison.
7. Observe and analyze the obtained output and verify that the required image processing operation has been performed correctly.
8. Save the final output and record the observations and results in the experiment record.

# Program :

### DEVELOPED BY : Janani saraswathi S
### REGISTER NUMBER : 212225230110


#### IMPORT LIBRARIES AND READ PASSPORT IMAGE
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

faceImage = cv2.imread('passport.png')

plt.imshow(faceImage[:, :, ::-1])
plt.title("Passport Image")
plt.show()
```


#### DISPLAY IMAGE DIMENSIONS
```
faceImage.shape
```


#### READ AND DISPLAY SUNGLASS PNG IMAGE
```
glassPNG = cv2.imread('sunglass.png', -1)

plt.imshow(glassPNG[:, :, ::-1])
plt.title("Sunglass Image")
plt.show()
```


#### RESIZE SUNGLASS IMAGE
```
glassPNG = cv2.resize(glassPNG, (190, 50))

print("image Dimension ={}".format(glassPNG.shape))

```
#### SEPARATE COLOR AND ALPHA CHANNELS
```
glassBGR = glassPNG[:, :, 0:3]

glassMask1 = glassPNG[:, :, 3]

plt.figure(figsize=[15, 15])

plt.subplot(121)
plt.imshow(glassBGR[:, :, ::-1])
plt.title('Sunglass Color channels')

plt.subplot(122)
plt.imshow(glassMask1, cmap='gray')
plt.title('Sunglass Alpha channel')

plt.show()
```

#### NAIVE SUNGLASS OVERLAY
```
faceWithGlassesNaive = faceImage.copy()

x, y = 460, 660

target_w = 800
target_h = 400

glassResized = cv2.resize(
    glassBGR,
    (target_w, target_h)
)

faceWithGlassesNaive[
    y:y+target_h,
    x:x+target_w
] = glassResized

plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.axis("off")
plt.show()

```
#### ALPHA BLENDING AND MASKING
```
glassAlpha = glassPNG[..., 3] / 255.0

glassBGR = glassPNG[..., :3]

glassBGR = cv2.resize(
    glassBGR,
    (target_w, target_h)
)

glassAlpha = cv2.resize(
    glassAlpha,
    (target_w, target_h)
)

eyeRoi = faceImage[
    y:y+target_h,
    x:x+target_w
].copy()

maskedEye = eyeRoi * (
    1 - glassAlpha[..., np.newaxis]
)

maskedGlass = glassBGR * (
    glassAlpha[..., np.newaxis]
)

eyeRoiFinal = maskedEye + maskedGlass

```
#### DISPLAY MASKED EYE AND SUNGLASS REGIONS
```
maskedEye_disp = np.clip(
    maskedEye.astype(np.uint8),
    0,
    255
)

maskedGlass_disp = np.clip(
    maskedGlass.astype(np.uint8),
    0,
    255
)

eyeRoiFinal_disp = np.clip(
    eyeRoiFinal.astype(np.uint8),
    0,
    255
)

plt.figure(figsize=[20, 20])

plt.subplot(131)
plt.imshow(maskedEye_disp[..., ::-1])
plt.title("Masked Eye Region")
plt.axis("off")

plt.subplot(132)
plt.imshow(maskedGlass_disp[..., ::-1])
plt.title("Masked Sunglass Region")
plt.axis("off")

plt.subplot(133)
plt.imshow(eyeRoiFinal_disp[..., ::-1])
plt.title("Augmented Eye and Sunglass")
plt.axis("off")

plt.show()

```
#### FINAL SUNGLASS AUGMENTATION
```
faceWithGlasses = faceImage.copy()

x, y = 460, 660

target_w, target_h = 800, 400

glassBGR = cv2.resize(
    glassBGR,
    (target_w, target_h)
)

glassAlpha = cv2.resize(
    glassPNG[..., 3] / 255.0,
    (target_w, target_h)
)

roi = faceWithGlasses[
    y:y+target_h,
    x:x+target_w
]

for c in range(3):

    roi[..., c] = (
        roi[..., c] * (1 - glassAlpha)
        + glassBGR[..., c] * glassAlpha
    )

faceWithGlasses[
    y:y+target_h,
    x:x+target_w
] = roi

```
#### DISPLAY ORIGINAL AND FINAL IMAGE
```
plt.figure(figsize=[15, 8])

plt.subplot(1, 2, 1)

plt.imshow(faceImage[..., ::-1])
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(faceWithGlasses[..., ::-1])
plt.title("Face with Sunglasses")
plt.axis("off")

plt.tight_layout()
plt.show()
```
# Output :
<img width="455" height="544" alt="image" src="https://github.com/user-attachments/assets/ab2a5790-83c6-48f3-ad2d-9c7632cf45fd" />
<img width="685" height="342" alt="image" src="https://github.com/user-attachments/assets/6327f0e3-1c80-4508-8204-249471cd323d" />
<img width="788" height="122" alt="image" src="https://github.com/user-attachments/assets/774c6218-c283-4c16-9ed9-b6b276b4f556" />
<img width="788" height="122" alt="image" src="https://github.com/user-attachments/assets/2e68274c-40ed-4cfa-9704-7aaf22ad2b21" />
<img width="377" height="467" alt="image" src="https://github.com/user-attachments/assets/a561de8a-651d-4d9a-ab30-d073ab9b16cd" />
<img width="788" height="110" alt="image" src="https://github.com/user-attachments/assets/089d7f77-56cf-4f55-ab47-8bc8fb97f57b" />
<img width="783" height="457" alt="image" src="https://github.com/user-attachments/assets/a06f6fa6-f6d0-4ae6-9cf9-8fa88d023136" />



# Result :
Sunglasses are successfully blended onto the face using alpha blending.
