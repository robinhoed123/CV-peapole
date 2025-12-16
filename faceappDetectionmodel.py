import cv2
import os
import numpy as np
from collections import Counter
from pathlib import Path
from insightface.app import FaceAnalysis
from ultralytics import YOLO

#Settings
embed_dir = r"finalProjeckt/embeddings_buffalo_l"
input_dir = Path(r"kagelData/testset")
output_dir = r"Facemodel/testdata\output"
yolo_model_path = r"finalProjeckt/best.pt"
yolo_model = YOLO(yolo_model_path)



YOLO_LABELS = {
    0: "ennis", 1: "robin", 2: "akif", 3: "seppe", 4: "alper", 5: "lorenzo",
    6: "tj", 7: "rayen", 8: "eh", 9: "arno", 10: "thomas", 11: "daiane"
}

imglist = sorted(p for p in input_dir.iterdir() if p.is_file())
#Init InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

#Functions

def detectface(img):
    # gets faces from insightface and sorts them left to right
    faces = app.get(img)
    facecord = []

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        facecord.append((img[y1:y2, x1:x2], x1, y1, x2, y2, f.embedding)) 
    facecord.sort(key=lambda f: f[1])
    return facecord

def drawOnimg(data, imgpath: str,img,predicton):
    # draws bounding boxes, names and confidence on the image
    for idx, face in enumerate(data):
        x1, y1, x2, y2 = face[1:5]
        # draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # write name
        cv2.putText(img, predicton[idx][0], (x1 + 20, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # write confidence
        cv2.putText(img, str(predicton[idx][1]), (x1 + 20, y2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(imgpath)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)
    Log(f"Image saved to {output_path}")

def getHightscore(data, name):
    """
    Filters the data list for tuples matching the given name and returns the tuple with the highest score.
    
    Args:
        data: List of tuples (name, score)
        name: The name to filter by
    
    Returns:
        The tuple (name, score) with the highest score for the given name, or None if no matches.
    """
    filtered = [t for t in data if t[0] == name]
    if not filtered:
        Log(f"getHightscore function error: cannot filter on name {name}" )
        Log(f"input into fuction was {data} , {name}")
        return None
    return max(filtered, key=lambda x: x[1])

def getmostcomonname(list):
    """
    Finds the most common name in a list of tuples.
    Each tuple has name as first element and score as second.
    Uses Counter to count occurrences.
    If there is only one most common, returns that.
    If there is a tie, checks the highest average score via highest_average.
    
    Args:
        list: List of tuples, where list[i][0] is name and list[i][1] is confidence score.
    
    Returns:
        Most common name as string. If tie, name with highest average.
    """
    namelist = [i[0] for i in list]
    mostCommonlist = Counter(namelist).most_common()
    if len(mostCommonlist) == 1:
        return getHightscore(list, mostCommonlist[0][0])
        # return mostCommonlist[0][0]
    else:
        realmostcomenlist = []
        most = mostCommonlist[0][1]
        for i in mostCommonlist:
            if i[1] == most:
                realmostcomenlist.append(i[0])
    if len(realmostcomenlist) == 1:
        return getHightscore(list, mostCommonlist[0][0])
        # return realmostcomenlist[0]
    else:
        return highest_average(list, realmostcomenlist)

def highest_average(data, names):
    # filter data to keep only names present in the list
    filtered = [t for t in data if t[0] in names]
    # if nothing filtered, return None
    if not filtered:
        Log(f"highest_average function error" )
        Log(f"data:{data}, name:{names}") # fixed variable name in log
        return None
    # create dict to group values per name
    values_per_name = {}
    # loop through filtered data and add values to name
    for name, value in filtered:
        values_per_name.setdefault(name, []).append(value)
    # calculate average per name
    averages = [(name, sum(vals)/len(vals)) for name, vals in values_per_name.items()]
    # find name with highest average
    highest = max(averages, key=lambda x: x[1])
    return highest
    # return highest[0]

def l2_norm(x):
    # normalizes the vector
    return x / np.linalg.norm(x)

def find_person(embedding, embed_dir=embed_dir, threshold=0.35, aantal=1200):
    # matches embedding against database files
    emb = l2_norm(embedding)
    sims = []

    for f in os.listdir(embed_dir):
        if not f.endswith(".npy"):
            continue
        ref = np.load(os.path.join(embed_dir, f))
        ref = l2_norm(ref)
        sim = np.dot(emb, ref)
        sims.append((f, sim))

    if not sims:
        return "No embeddings found in folder."

    sims.sort(key=lambda x: x[1], reverse=True)
    top = [(name, score) for name, score in sims[:aantal] if score >= threshold]

    if not top:
        Log("no face found")
        return ("none",1)
        # no match found for this face
    """
    nameData contains all found embedding sims, score is always higher than threshold X "0.35"
    and is sorted by score. example content:
     [(robin,0.6),(robin,0.5),(enis,0.42),(eh,0.38),(arno,0.35)...]
    """
    namedata = [((i[0].split('_'))[0].lower(), i[1]) for i in top]
    prodiction = getmostcomonname(namedata)
    if (prodiction == None):
        return ("none",1)
    else:
        return prodiction

def getMaxfaces(img):
    # tries rotating and flipping the image to find more faces
    steps=4
    maxrange=60 #this has to be between 1 and 90 deg
    deg=round(maxrange/steps)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = detectface(img)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    for x in range(3):
        if (x==2):
            img = cv2.flip(img, 1)
        for i in range(steps):
            rotation_matrix = cv2.getRotationMatrix2D(center, (i+1)*(deg), 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height))
            tempdata = detectface(img)
            if(len(tempdata)>len(data)):
                data=tempdata
                Log(f"more faces found when image rotated {(i+1)*(-deg)} deg")
                if (x>=2):
                    Log("img is mirrored")
        deg=deg*(-1)
    return data

def scaleimg(img, factor, upscale=False):
    # resizes the image by a factor
    height, width = img.shape[:2]
    if upscale:
        scale = factor
    else:
        scale = 1.0 / factor
    scaled_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return scaled_img

def yolo_prediction(img):
    """
    Runs YOLO prediction on the image and returns formatted results.
    
    Args:
        img: The input image.
        
    Returns:
        A list of predictions: [[label, confidence, [x1, y1, x2, y2]], ...]
    """
    results = yolo_model(img, verbose=False)
    preds = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            
            label = YOLO_LABELS.get(cls, "unknown")
            label = label.lower()
            preds.append([label, conf, [x1, y1, x2, y2]])
    return preds

def merge_predictions(methods_list, threshold=0.5):
    """
    Merges predictions from multiple methods.
    1. Filters out predictions with confidence < threshold.
    2. Resolves overlaps: 
       - Calculates overlap distance (x2_current - x1_next).
       - If overlap_distance > width of smallest box, they are merged.
    3. Final check: Removes duplicates if adjacent predictions have the same label. 
    iknow this is not a good solution but its 3:42 in the morning i need to go to bed 
    Args:
        methods_list: A list of lists, where each inner list contains predictions.
                      Each prediction is [label, confidence, [x1, y1, x2, y2]].
        threshold: Minimum confidence score required to keep a prediction.
        
    Returns:
        A sorted list of merged predictions.
    """
    # Flatten the list of lists into a single list of predictions
    all_preds = []
    for method in methods_list:
        for pred in method:
            # Filter by threshold
            if pred[1] >= threshold:
                all_preds.append(pred)
            
    # Sort predictions by x1 coordinate
    all_preds.sort(key=lambda x: x[2][0])
    
    if not all_preds:
        return []
        
    merged = []
    current_best = all_preds[0]
    
    for i in range(1, len(all_preds)):
        next_pred = all_preds[i]
        
        # Calculate widths
        # current_best coords: [x1, y1, x2, y2]
        cb_coords = current_best[2]
        cb_width = cb_coords[2] - cb_coords[0]
        
        # next_pred coords
        np_coords = next_pred[2]
        np_width = np_coords[2] - np_coords[0]
        
        # Calculate overlap distance: box1_x2 - box2_x1
        # Since list is sorted by x1, current_best is box1, next_pred is box2
        overlap_distance = cb_coords[2] - np_coords[0]
        
        smallest_width = min(cb_width, np_width)
        
        # Check condition: overlap > width of smallest bounding box
        if overlap_distance > smallest_width:
            # They overlap significantly (one likely inside the other), pick the one with higher confidence
            if next_pred[1] > current_best[1]:
                current_best = next_pred
            # If current_best is higher, we keep it and ignore next_pred
        else:
            # No significant overlap, push current_best to merged and start a new cluster
            merged.append(current_best)
            current_best = next_pred
            
    # Append the last one
    merged.append(current_best)
    
    #Final Check: Remove adjacent duplicates based on label
    # "if the next prediction is the same as the previous prediction remove the previous prediction"
    final_result = []
    i = 0
    while i < len(merged) - 1:
        current_pred = merged[i]
        next_pred = merged[i+1]
        
        # Check if labels are the same
        if current_pred[0] == next_pred[0]:
            # Skip the current one (effectively removing it)
            pass 
        else:
            final_result.append(current_pred)
        i += 1
        
    # Always add the last element if the list wasn't empty
    if merged:
        final_result.append(merged[-1])
    
    return final_result

def append_kaggle_csv(text):
    # appends text to the kaggle csv file
    text = text.lower()
    with open(KagglePath, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")

def Log(text):
    # prints text and writes it to the log file
    text=str(text)
    print(text)
    with open(logPath, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
#Main loop

# checks existing logs in folder, new log file = log_n+1
# ensures a new log for every run
logfolder=r"finalProjeckt/logs"
os.makedirs(logfolder, exist_ok=True)
log_files = [f for f in os.listdir(logfolder) if os.path.isfile(os.path.join(logfolder, f))]
count = len(log_files)
logPath = os.path.join(logfolder, f"log_{count}.txt")

# same for kaggle files
kagglefolder=r"finalProjeckt/kagle"
os.makedirs(kagglefolder, exist_ok=True)
kaggle_files = [f for f in os.listdir(logfolder) if os.path.isfile(os.path.join(logfolder, f))]
count = len(kaggle_files)
KagglePath = os.path.join(kagglefolder, f"Robin_faceprodictions_{count}.csv")

append_kaggle_csv("image,label_name")

for p, img_path in enumerate(imglist):
    Log("=======================")
    img = cv2.imread(str(img_path))

    #data = getMaxfaces(img) unused
    scaled_img=scaleimg(img,2,True)
    
    #Method 1: Normal Detection
    data = detectface(scaled_img)
    method1_preds = []
    for face in data:
        pred_name, pred_score = find_person(face[5])
        # face[1:5] is x1, y1, x2, y2
        coords = [face[1], face[2], face[3], face[4]]
        method1_preds.append([pred_name, pred_score, coords])

    #Method 2: Flipped Detection
    flipped_img = cv2.flip(scaled_img, 1)
    data_flipped = detectface(flipped_img)
    method2_preds = []
    h, w = scaled_img.shape[:2]
    
    for face in data_flipped:
        pred_name, pred_score = find_person(face[5])
        x1, y1, x2, y2 = face[1], face[2], face[3], face[4]
        
        # Flip coordinates back
        # x1 in flipped image corresponds to w - x2 in original
        # x2 in flipped image corresponds to w - x1 in original
        new_x1 = w - x2
        new_x2 = w - x1
        
        # Ensure x1 < x2 (though the math above should guarantee it if x1 < x2 originally)
        if new_x1 > new_x2:
            new_x1, new_x2 = new_x2, new_x1
            
        coords = [new_x1, y1, new_x2, y2]
        method2_preds.append([pred_name, pred_score, coords])
        
    # Reverse the list so it goes from left to right (since we flipped the image)
    method2_preds.reverse()
    
    #Method 3: YOLO Detection
    method3_preds = yolo_prediction(scaled_img)

    #Merge Predictions
    # The 3D list structure: [method1_preds, method2_preds, method3_preds]
    final_list = merge_predictions([method1_preds, method2_preds, method3_preds])
    #final_list = merge_predictions(method1_preds)

    #Reconstruct data for drawing and logging
    data = []
    predictions = []
    
    for item in final_list:
        name, score, coords = item
        # Reconstruct tuple for drawOnimg: (crop, x1, y1, x2, y2, embedding)
        # We pass None for crop and embedding as they are not used in drawOnimg
        data.append((None, coords[0], coords[1], coords[2], coords[3], None))
        predictions.append((name, score))

    names=[i[0] for i in predictions]
    namesCSV=[i for i in names if i !="none"]# remove Face=none prodictions
    drawOnimg(data, str(img_path),scaled_img, predictions)
    Log(names)
    Log(f"{len(names)} peapole found in {img_path} {p+1}/{len(imglist)} done")
    strname = ";".join(namesCSV)
    if(len(namesCSV)>0):
        append_kaggle_csv(f"{p+1},{strname}")
    else:
        append_kaggle_csv(f"{p+1},none")