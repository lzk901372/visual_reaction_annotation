### 1. Emotion Detection

Use FER in `./fer_emotion` to detect frame-wise emotions (7-class emotion scores) resulting in `.csv` files. See `./fer_emotion/README.md` for details.

### 2. Landmarks and Motion Detection

Use Mediapipe in `./nod_headshake/face_detection` to extract landmarks and motion information resulting in `.pkl` files. See `./nod_headshake/face_detection/README.md` for details.

### 3. Fine Valid Results

Use `./generate_usable.py` to fine usable video-result pairs and save it into a txt file. See `./README.md` for details.

### 4. Reaction Detection

For six different reaction classes, use different detectors in `./reaction_detector` for frame-wise detection. See `./reaction_detector/README.md` for details.

### 5. HDF5 Injection

Use `./build_reaction_pruned_h5.py` to inject reaction scores into a HDF5 file. See `./RAEDME.md` for details.