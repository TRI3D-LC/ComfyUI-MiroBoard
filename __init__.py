import numpy as np
import os
import cv2
import torch
import json
import requests
import io


def addImage(boardId, fileObj, x, y, width):
    import json
    from os import path
    import requests
    import dotenv
    import os
    dotenv.load_dotenv()
    oAuthToken = os.getenv("MIRO_OAUTH_TOKEN")
    files = [
        (
            "resource",
            ("tmp.jpg", fileObj, "image/png"),
        ),
        (
            "data",
            (
                None,
                json.dumps(
                    {
                        "position": {
                            "x": x,
                            "y": y,
                        },
                        "geometry": {
                            "width": width,
                            "rotation": 0,
                        },
                    }
                ),
                "application/json",
            ),
        ),
    ]
    headers = {
        "Authorization": f"Bearer {oAuthToken}",
    }
    url = f"https://api.miro.com/v2/boards/{boardId}/images"

    response = requests.post(url, headers=headers, data={}, files=files)
    print(response.text)

def addNote(boardId, content, x,y,width):
    import json
    from os import path
    import requests
    import dotenv
    import os
    dotenv.load_dotenv()
    oAuthToken = os.getenv("MIRO_OAUTH_TOKEN")
    import requests

    url = f"https://api.miro.com/v2/boards/{boardId}/sticky_notes"

    payload = {
        "data": { "content": content },
        "position": {
            "x": x,
            "y": y,
        },
        "geometry": {
            "width": width,
        },
    }
    headers = {
        "Authorization": f"Bearer {oAuthToken}",
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
#get current file path and directory 
def getDBLocalPath():
    import os
    currentDir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(currentDir, "db.local")

def writeToDB(x_offset):
    #there is a db.local file in the same directory as this file, we will write x_offset value to that text file
    db_local_path = getDBLocalPath()
    with open(db_local_path, "w") as f:
        f.write(str(x_offset))

def readFromDB():
    import os
    import time
    #read the x_offset value from the db.local file
    db_local_path = getDBLocalPath()
    if not os.path.exists(db_local_path):
        writeToDB(0)
        time.sleep(1)
    with open(db_local_path, "r") as f:
        x_offset = f.read()
    return int(x_offset)
    


class AddImageMiroBoard:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "y_start": ("INT", {
                    "default": -1,
                }),
                "x_offset": ("INT", {
                    "default": 0,
                }),
                "board_id": ("STRING", {
                    "default": "BOARD_ID",
                }),
                "input_image_1": ("IMAGE", ),
                "input_image_2": ("IMAGE", ),
                "input_image_3": ("IMAGE", ),
                "input_image_4": ("IMAGE", ),
                "input_image_5": ("IMAGE", ),
                "notes" : ("STRING", {
                    "default": "Notes",
                }),
            },
        }
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "MiroBoard"
    # def run(self, input_image_1):
    def run(self, board_id, y_start, x_offset, input_image_1, input_image_2, input_image_3, input_image_4, input_image_5, notes):

        if board_id == "BOARD_ID":
            return { "ui": { "images": list() } }

        import base64
        # from .utils import addImage, writeToDB, readFromDB
        def tensor_to_cv2_img(tensor, remove_alpha=False):
            i = 255. * tensor.cpu().numpy()  # This will give us (H, W, C)
            img = np.clip(i, 0, 255).astype(np.uint8)
            return img
        def cv2_img_to_tensor(img):
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)[
                None,
            ]
            return img
        
        input_images = [input_image_1, input_image_2, input_image_3, input_image_4, input_image_5]
        # input_images = [input_image_1]

        if y_start >= 0:
            writeToDB(y_start)
        else:
            y_start = readFromDB()
        width = 400
        gap_x = int(0.2*width)
        # board_id = "uXjVK75fvYY="

        cnt = -1

        addNote(board_id, notes, x_offset + cnt*(width + gap_x), y_start, width)
        cnt+=1
        for input_image in input_images:
            if input_image is None:
                continue

            input_image = tensor_to_cv2_img(input_image[0])
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            file_obj = io.BytesIO()
            is_success, buffer = cv2.imencode(".jpg", input_image)
            file_obj.write(buffer)
            file_obj.seek(0)

            
            addImage(board_id, file_obj, x=x_offset + cnt*(width + gap_x), y=y_start, width=width)
            cnt+=1 
        
        writeToDB(y_start + 2*width)

        return { "ui": { "images": list() } }
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "add-image-miro-board": AddImageMiroBoard,
}
VERSION = "0.1.1"
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "add-image-miro-board": "Add Image Miro Board" + " v" + VERSION,
}


