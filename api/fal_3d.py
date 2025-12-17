import fal_client
from PIL import Image
from api.BaseModel3D import BaseModel3D
from ultralytics import YOLO
import torch
import numpy as np
import cv2

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])


class FalAPI3D(BaseModel3D):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.human_detector_model = YOLO("assets/yolo11l-seg.pt")
        self.human_detector_model.to(self.device)
        self.human_conf = 0.4

    def deal_with_one_image(self, image_str):
        padding = 5
        image = self.read_image(image_str, format="local")
        detected_objects = self.human_detector_model.predict(
            source=image,
            conf=self.human_conf,
            iou=0.45,
            max_det=10,
            classes=[0],  # Class 0 corresponds to 'person' in COCO dataset
            device=self.device
        )
        output_data = []
        result = detected_objects[0]
        for index in range(len(result.boxes)):
            box = result.boxes[index]
            if result.masks is not None:
                x1 = int(box.xyxy[0][0].item())
                y1 = int(box.xyxy[0][1].item())
                x2 = int(box.xyxy[0][2].item())
                y2 = int(box.xyxy[0][3].item())
                
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.width, x2 + padding)
                y2 = min(image.height, y2 + padding)
                
                mask = result.masks.data[index]
                mask = mask.cpu().numpy().astype(np.uint8) * 255
                mask = cv2.resize(mask, (image.width, image.height))
                mask_cropped = mask[y1:y2, x1:x2]
                mask_pil = Image.fromarray(mask_cropped)
                image_cropped = image.crop((x1, y1, x2, y2))
                
                output_data.append({
                    "image": image_cropped,
                    "mask": mask_pil,
                    "image_height": image.height,
                    "image_width": image.width
                })
                # cv2.imwrite(f"tmps/image_{index}.png", cv2.cvtColor(np.array(image_cropped), cv2.COLOR_RGB2BGR))
                # cv2.imwrite(f"tmps/mask_{index}.png", mask_cropped)

        return output_data
        

    def predict(self, image_str):
        detected_datas = self.deal_with_one_image(image_str)
        if len(detected_datas) == 0:
            print("No human detected.")
            return None
        
        # TODO: support multiple detected humans
        detected_data = detected_datas[0]
        image = detected_data["image"]
        mask = detected_data["mask"]
        image_url = fal_client.upload_image(image)
        mask_url = fal_client.upload_image(mask)

        try:
            result = fal_client.subscribe(
                "fal-ai/sam-3/3d-objects",
                arguments={
                    "image_url": image_url,
                    "prompt": "human",
                    "point_prompts": [],
                    "box_prompts": [],
                    "mask_urls": [mask_url]
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
        except Exception as e:
            print("Error during 3D reconstruction:", str(e))
            return None

        print("3D Reconstruction result:", result)
        ply_url = result["gaussian_splat"]["url"]
        print("3D Model PLY URL:", ply_url)
        
        detected_data["ply_url"] = ply_url
        return detected_data
        
if __name__ == "__main__":
    fal_api_3d = FalAPI3D()
    fal_api_3d.predict("https://v3b.fal.media/files/b/0a8439f8/E8gEXWsl2C-Euo4dGayzi_An_zyCCnSaytVklh_99sSYt4Z4Hh5e3s7VnNlx5JfN5KuC0j_bnq1AP9JfRoAmOQz5TP0DdCYMk4796Gloe5no1vvpoqhD-p3kE.jpeg")
    # print("Detected objects:", len(result))
