from typing import Tuple

import cv2 as cv
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes

from PIL import Image, ImageOps

import torch

from src.model.detr import DETR
from src.data.transforms import ValidationTransform
from src.visualize.inference import extract_boxes, extract_boxes_with_attn
from src.data import (
    box_to_img_xywh_and_scale,
    get_xywh,
    CLASSES, 
    CLASS_IDX_TO_NAME
)


class DETRVisualizer:
    """Really simple visualizer class for DETR.""" 
    def __init__(
            self, 
            detr: DETR, 
            img_size: Tuple[int, int], 
            device: str="cpu"
    ) -> None:
        """
        Args:
        ----- 
        detr : The DETR model
        img_size : Image size for DETR
        device : Computing device (i.e. cpu or cuda or mps) 
        """ 
        self.device = torch.device(device)
        self.detr = detr
        self.detr.eval() 
        self.detr.to(device) 

        # Prepare DETR for attention map visualization 
        for decoder in self.detr.transformer.decoder.layers:
            decoder.save_attention_weight_ = True
        img_h, img_w = img_size     # Same (height, width) convention as v2.Resize.
        self.attention_size = ((img_h + 31) // 32, (img_w + 31) // 32)

        # Pillow image to torch.Tensor transform
        self.transform = ValidationTransform(img_size)
        self.empty_tgt = {
            "boxes": torch.zeros(0, 4, dtype=torch.float32), 
            "labels": torch.zeros(0, dtype=torch.int64)
        }
        self.img_size = img_size
        
        # Prepare colors
        color_palette = sns.color_palette("hls", len(CLASSES))
        self.color_palette = color_palette

    @torch.no_grad()
    def detect_webcam(
        self, 
        score_thresh: float=0.5, 
        nms_iou_thresh: float | None = None
    ) -> None:
        """Opens webcam using openCV and detects objects in video stream.

        Args:
        -----
        score_thresh : Threshold for foreground objects 
        nms_iou_thresh : nms threshold (actually not needed if trained properly)
        """ 
        cap = cv.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert to torch.Tensor (with correct device)
            img_tensor, img_size = self._convert_from_numpy(frame)

            # DETR inference
            logits, boxes_pred = self.detr(img_tensor)
            scores, labels, boxes = extract_boxes(
                logits, boxes_pred, score_thresh, nms_iou_thresh
            )

            # Visualize bounding boxes
            self._visualize_boxes_cv(frame, labels, scores, boxes, img_size)

            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv.destroyAllWindows()

    @torch.no_grad()
    def detect_video(
            self, 
            video_file: str, 
            score_thresh: float=0.5, 
            nms_iou_thresh: float | None = None,
            detect_every: int=2
    ) -> None:
        """Opens video file using openCV and detects objects in video stream.

        Args:
        -----
        score_thresh : Threshold for foreground objects 
        nms_iou_thresh : nms threshold (actually not needed if trained properly)
        """ 
        cap = cv.VideoCapture(video_file)
        t = 0

        while cap.isOpened():
            # Capture frame
            ret, frame = cap.read()
        
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
                
            # Convert to torch.Tensor (with correct device)
            img_tensor, img_size = self._convert_from_numpy(frame)
            
            # DETR inference
            if t % detect_every == 0: 
                logits, boxes_pred = self.detr(img_tensor)
                scores, labels, boxes = extract_boxes(
                    logits, boxes_pred, score_thresh, nms_iou_thresh
                )

            # Visualize
            self._visualize_boxes_cv(frame, labels, scores, boxes, img_size)
            cv.imshow('frame', frame)
            t += 1

            if cv.waitKey(1) == ord('q'):
                break


        cap.release()
        cv.destroyAllWindows()

    @torch.no_grad()
    def detect_img(
            self, 
            img_file: str, 
            score_thresh: float = 0.5, 
            nms_iou_thresh: float | None = None
    ) -> None:
        """Opens image file using pyplot and detects objects in image.

        Args:
        -----
        score_thresh : Threshold for foreground objects 
        nms_iou_thresh : nms threshold (actually not needed if trained properly)
        """ 
        img, img_tensor, img_size = self._load_image_from_file(img_file) 

        logits, boxes_pred = self.detr(img_tensor)
        scores, labels, boxes = extract_boxes(logits, boxes_pred, score_thresh, nms_iou_thresh)

        w, h = img.size
        fig_width= 10
        fig_height = fig_width * h / w
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))       

        # fig, ax = plt.subplots()        
        ax.axis("off") 

        ax.imshow(img)

        self._visualize_boxes_pyplot(ax, labels, scores, boxes, img_size) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #plt.tight_layout() 
        plt.show() 

    @torch.no_grad()
    def attention_img(
            self, 
            img_file: str, 
            score_thresh: float = 0.5, 
            nms_iou_thresh: float | None = None
    ) -> None:
        """Opens image file using pyplot and detects objects in image (+attention map).

        Args:
        -----
        score_thresh : Threshold for foreground objects 
        nms_iou_thresh : nms threshold (ctually not needed if trained properly)
        """ 
        
        img, img_tensor, img_size = self._load_image_from_file(img_file) 

        # DETR inference
        logits, boxes_pred = self.detr(img_tensor)
        
        # Use the actual encoder grid, including rectangular inputs.
        self.attention_size = self.detr.feature_grid_size
        # Cache attention weights
        attention_weights = self.detr.transformer.decoder.attention_weights_
        attention_weights = torch.stack(attention_weights).mean(0).squeeze(0)     # [num_queries, H * W]

        # Extract valid objects 
        scores, labels, boxes, attention_weights = extract_boxes_with_attn(
            logits, boxes_pred, attention_weights, score_thresh, nms_iou_thresh
        ) 

        w, h = img.size
        fig_width= 10
        fig_height = fig_width * h / w
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))        

        # fig, ax = plt.subplots()        
        ax.axis("off") 

        ax.imshow(img)

        for i, box in enumerate(boxes):
            box = box_to_img_xywh_and_scale(box, img_size) 
            x, y, w, h = get_xywh(box) 

            _, obj_prob, obj_name, obj_color = self._get_detection_metadata(labels, scores, i)

            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(img_size[0], int(round(x + w)))
            y2 = min(img_size[1], int(round(y + h))) 
            
            # Draw attention weight
            weight = self._upsample_attention_map(attention_weights[i], img_size)
            masked = np.ma.masked_all_like(weight) 
            masked[y1:y2, x1:x2] = weight[y1:y2, x1:x2]
            ax.imshow(masked, alpha=0.2, cmap="jet")

            # Draw rectangle
            obj_rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor=obj_color, facecolor="none"
            )
            ax.add_patch(obj_rect)

            # Draw text 
            self._draw_text_pyplot(ax, x, y, obj_name, obj_prob, obj_color)
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show() 

    def _visualize_boxes_cv(
            self, 
            frame: np.ndarray, 
            labels: torch.Tensor, 
            scores: torch.Tensor, 
            boxes: torch.Tensor, 
            img_size: Tuple[int, int]
    ) -> None:
        """Visualizes bounding boxes of objects in one image using openCV.

        Args:
        ---- 
        frame : Image frame as np.ndarray
        labels : Labels of shape [n_objects]
        scores : Scores of shape [n_objects]
        boxes : Boxes of shape [n_objects, 4]
        img_size : Real image size (probably differs from self.img_size)
        """ 
        for i, box in enumerate(boxes):
            box = box_to_img_xywh_and_scale(box, img_size) 
            x, y, w, h = get_xywh(box)

            # Get detection meta information (i.e. prob, name, color, ...)
            _, obj_prob, obj_name, obj_color = self._get_detection_metadata(labels, scores, i)
            obj_color = tuple(obj_color[::-1])
            obj_color_cv = tuple((np.array(obj_color[::-1]) * 255).round().astype(np.uint8).tolist())
            
            font = cv.FONT_HERSHEY_SIMPLEX
            text = f"{obj_name} {obj_prob}%"
            (text_w, text_h), baseline = cv.getTextSize(text, font, 1, 1) 

            # Draw object rectangle 
            cv.rectangle(frame, (x, y), (x + w, y + h), obj_color_cv, 3)

            # Draw filled rectangle around text and text
            cv.rectangle(frame, (x, y - text_h - baseline), (x + text_w, y), obj_color_cv, -1)
            cv.putText(frame, text, (x, y - baseline // 2), font, 1, (0, 0, 0), 1, cv.LINE_AA) 

    def _visualize_boxes_pyplot(
            self,
            ax: Axes,
            labels: torch.Tensor,
            scores: torch.Tensor,
            boxes: torch.Tensor,
            img_size: Tuple[int, int]
    ) -> None:
        """Visualizes bounding boxes of objects in one image using pyplot.

        Args:
        ---- 
        frame : Image frame as np.ndarray
        labels : Labels of shape [n_objects]
        scores : Scores of shape [n_objects]
        boxes : Boxes of shape [n_objects, 4]
        img_size : Real image size (probably differs from self.img_size)
        """ 
        for i, box in enumerate(boxes):
            box = box_to_img_xywh_and_scale(box, img_size) 
            x, y, w, h = get_xywh(box)

            # Get obj probability, name and color
            _, obj_prob, obj_name, obj_color = self._get_detection_metadata(labels, scores, i)

            # Draw rectangle 
            obj_rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor=obj_color, facecolor="none"
            )
            ax.add_patch(obj_rect)

            self._draw_text_pyplot(ax, x, y, obj_name, obj_prob, obj_color)

    def _draw_text_pyplot(
            self, 
            ax: Axes, 
            x: int, 
            y: int, 
            obj_name: str, 
            obj_prob: float, 
            obj_color: Tuple[float, float, float]
    ) -> None:
        """Draws object metainformation on an Axes object.

        Args:
        -----
        ax : Current Axes
        x : x-start position of text (top-left) 
        y : y-start position of text (top-left) 
        obj_name : Name of the object
        obj_prob : Probability of the object (from prediction)
        obj_color : Color of the object
        """
        # Draw text
        text = f"{obj_name} {obj_prob}%"
        ax.text(
            x, y - 2, text,
            color="black",
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                facecolor=obj_color,
                edgecolor=obj_color,
                boxstyle="square,pad=0.2",
                linewidth=0
            ),
            fontsize=12
        )

    def _get_detection_metadata(
            self, 
            labels: torch.Tensor, 
            scores: torch.Tensor, 
            index: int
    ) -> Tuple[int, float, str, Tuple[float, float, float]]:
        """Extract object meta information (i.e. label, prob, name, color)."""
 
        assert index < labels.size(0), (
            f"Invalid index for tensor acces, got: {index}, "
            f"expects something between 0 - {labels.size(0) - 1}" 
        )
        
        obj_label = int(labels[index].item()) 
        obj_prob = round(float(100 * scores[index].item()), 2)
        obj_name = CLASS_IDX_TO_NAME[obj_label]
        obj_color = self.color_palette[obj_label] 

        return obj_label, obj_prob, obj_name, obj_color

    def _upsample_attention_map(
            self, 
            attention_map: torch.Tensor, 
            img_size: Tuple[int, int]
    ) -> np.ndarray:
        """Expects attention map of shape [1, W * H].""" 
        h_attn, w_attn = self.attention_size
        weight = attention_map.reshape(1, 1, h_attn, w_attn)            # [1, H * W] -> [1, 1, H, W]
        
        w_img, h_img = img_size     # Original image size
        weight = torch.nn.functional.interpolate(
            weight, (h_img, w_img), mode="bilinear"
        )                                                               # [1, 1, H * W] -> [1, 1, H_0, W_0]
        weight = weight.squeeze(0).squeeze(0).detach().cpu().numpy()    # [H_0, W_0]
        return weight

    def _load_image_from_file(
            self, 
            img_file: str
    ) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
        """Loads image as PIL.Image and torch.Tensor.""" 
        # Pillow image 
        img = Image.open(img_file)
        # print(img.size) 
        if img_file.lower().endswith(".jpeg"): 
            img = ImageOps.exif_transpose(img) 
        # print(img.size) 
        # img = img.rotate(-90) 
        img_size = img.size 

        # To torch.Tensor 
        img_tensor, _ = self.transform(img, self.empty_tgt)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img, img_tensor, img_size

    def _convert_from_numpy(
            self, 
            frame: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Converts np.ndarray (in BGR) to torch.Tensor (in RGB).""" 
        # BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
        
        # Frame to PIL image and torch.Tensor
        img = Image.fromarray(frame_rgb)
        img_size = img.size 

        # To torch.Tensor 
        img_tensor, _ = self.transform(img, self.empty_tgt)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, img_size