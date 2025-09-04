# Computer Vision Workshop: Bike Image Manipulation and 3D Visualization

## Dataset and File Structure

For this workshop, we will use a dataset of bike images stored on a fast SSD drive provided to you. This ensures quick access to the images and smooth processing during the hands-on exercises.

### Where to Find the Data
- The SSD will be connected to your workstation and mounted as a drive (e.g., `D:/` or similar).
- The dataset is located in the following folder structure:

```
NMBS-Bikes-dataset/
└── wonka/
    └── input_image/
        ├── original/
        │   ├── IMG20230104225221_01.jpg
        │   ├── IMG20230104225246_01.jpg
        │   └── ...
        └── side/
            ├── IMG20230104225221_01-side.jpg
            ├── IMG20230104225246_01-side.jpg
            └── ...
```

### File Naming and Pairing
- Each bike is photographed from multiple angles.
- The `original` folder contains the main images, e.g., `IMG20230104225221_01.jpg`.
- The `side` folder contains images of the same bike from a slightly offset viewpoint, with filenames matching the original but with `-side` appended before `.jpg`, e.g., `IMG20230104225221_01-side.jpg`.
- To pair images, simply match the base filename (everything before `.jpg`) between the two folders.

### What You'll Do
- You will write code to automatically pair each original image with its corresponding side image.
- You will use these pairs for all subsequent computer vision tasks in the workshop.

**Tip:** If you are unsure about the SSD path on your machine, ask your instructor or check the drive list in your file explorer.

---

Welcome to the Computer Vision Workshop! In this workshop, you'll learn how to work with real-world bike images, perform image manipulations, extract objects, and even create a simple 3D visualization using Python and popular libraries like OpenCV, PyTorch, and Matplotlib.

## Workshop Structure
This workshop is organized as a Jupyter notebook (`workshop.ipynb`). Each code cell introduces a new concept or task. Below, you'll find a step-by-step guide explaining what each cell does, the math behind it, and code snippets you can use or adapt.

---

### 1. **Setup and Data Pairing**
**Purpose:** Load the dataset, pair each original image with its corresponding side image, and prepare for further processing.

**Key OpenCV/Python functions:**
- `os.listdir`, `os.path.join`

**Example code:**
```python
originals = sorted([f for f in os.listdir(dir_original) if f.endswith('.jpg')])
sides = sorted([f for f in os.listdir(dir_side) if f.endswith('.jpg')])
pairs = []
for orig in originals:
    base = orig[:-4]
    side_name = f"{base}-side.jpg"
    if side_name in sides:
        pairs.append((os.path.join(dir_original, orig), os.path.join(dir_side, side_name)))
```

---

### 2. **Display Image Pairs**
**Purpose:** Visualize the paired images side by side.

**Key OpenCV/Matplotlib functions:**
- `cv2.imread`, `cv2.cvtColor`, `matplotlib.pyplot.imshow`

**Example code:**
```python
def show_image_pair(orig_path, side_path):
    img_orig = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
    img_side = cv2.cvtColor(cv2.imread(side_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_orig)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img_side)
    plt.title('Side')
    plt.axis('off')
    plt.show()
```

---

### 3. **OpenCV Image Manipulations**
**Purpose:** Apply basic image processing techniques to a sample image.

**Key OpenCV functions:**
- `cv2.cvtColor` (for grayscale)
- `cv2.GaussianBlur`
- `cv2.Canny`

**Example code:**
```python
img_gray = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(sample_img, (11, 11), 0)
img_edges = cv2.Canny(img_gray, 100, 200)
```

---

### 4. **Finding 'Other Face' Pairs Using Image Embeddings**
**Purpose:** Use a pre-trained deep learning model to find the most similar images (the other face of the same bike) based on visual similarity.

**Key PyTorch/Torchvision functions:**
- `torchvision.models.resnet18`
- `torch.nn.Sequential`
- `torchvision.transforms`
- `scipy.spatial.distance.cdist`

**Example code:**
```python
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
emb = model(x).squeeze().cpu().numpy().flatten()
dists = cdist([emb], np.array(embeddings)[mask])[0]
```

---

### 5. **Feature Matching Between Best-Matched Faces**
**Purpose:** Compare the best-matched faces (from the previous step) using ORB feature matching.

**Key OpenCV functions:**
- `cv2.ORB_create`
- `cv2.BFMatcher`
- `cv2.drawMatches`

**Example code:**
```python
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
```

---

### 6. **Bike Extraction Using Semantic Segmentation**
**Purpose:** Extract the bike from the image, removing the background, person, and ground using a pre-trained DeepLabV3 segmentation model.

**Key PyTorch/Torchvision functions:**
- `torchvision.models.segmentation.deeplabv3_resnet50`
- `torchvision.transforms`

**Example code:**
```python
seg_model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").to(device)
out = seg_model(x)['out'][0]
mask = out.argmax(0).cpu().numpy() == BIKE_CLASS
```

---

### 7. **3D Gaussian Splat Visualization**
**Purpose:** Create a simple 3D visualization of the bike using four different views.

**Key math and OpenCV/numpy functions:**
- 3D projection math (see below)
- `np.stack`, `np.concatenate`, `matplotlib.pyplot.scatter`

**Projection math:**
Given a pixel (x, y) in the image, focal length f, and image center (cx, cy):

```
Z = assumed depth (e.g., 2 meters)
X_cam = (x - cx) * Z / f
Y_cam = (y - cy) * Z / f
```
To transform to world coordinates:
```
[ X_world ]   [ R | t ]   [ X_cam ]
[ Y_world ] = [       ] * [ Y_cam ]
[ Z_world ]   [ 0 | 1 ]   [ Z    ]
```
Where R is a 3x3 rotation matrix and t is a 3D translation vector for each camera/view.

**Example code:**
```python
ys, xs = np.where(mask)
zs = np.ones_like(xs) * 2  # Assume all points are 2m from camera
xs_cam = (xs - cx) * zs / focal_length
ys_cam = (ys - cy) * zs / focal_length
pts_cam = np.stack([xs_cam, ys_cam, zs], axis=1)
pts_world = pts_cam @ pose['R'].T + pose['t']
```

**Visualization:**
```python
ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, s=sizes, alpha=0.5)
```

---

## Learning Objectives
- Understand how to pair and manipulate real-world images.
- Apply basic and advanced computer vision techniques using OpenCV and PyTorch.
- Use pre-trained deep learning models for image similarity and segmentation.
- Visualize multi-view data in 3D.

## Tips for Interns
- Read the comments in each code cell carefully—they explain what each part does.
- Try changing parameters (e.g., blur size, edge thresholds, camera poses) and observe the effects.
- Experiment with your own ideas: can you improve the segmentation? Try more images for the 3D splat?
- If you get stuck, ask questions and collaborate with your peers!

## OpenCV Functions Used
- `cv2.imread`, `cv2.cvtColor`, `cv2.GaussianBlur`, `cv2.Canny`, `cv2.ORB_create`, `cv2.BFMatcher`, `cv2.drawMatches`

Happy coding and exploring computer vision!
