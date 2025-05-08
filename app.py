import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr
import os
import io
import uuid

# Load Faster R-CNN model with proper weight assignment
frcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, progress=True)
state_dict = torch.hub.load_state_dict_from_url(frcnn_weights.url, progress=True, map_location=torch.device('cpu'))
frcnn_model.load_state_dict(state_dict, strict=False)
frcnn_model.eval()

# Load DETR model and processor
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load Mask R-CNN model
maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn_model.eval()

# Load Mask2Former model and processor
mask2former_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")
mask2former_model.eval()

# COCO class names for Faster R-CNN and Mask R-CNN
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Mask2Former label map
MASK2FORMER_COCO_NAMES = mask2former_model.config.id2label if hasattr(mask2former_model.config, "id2label") else {str(i): str(i) for i in range(133)}

def detect_objects_frcnn(image, threshold=0.5):
    """Run Faster R-CNN detection."""
    if image is None:
        blank_img = Image.new('RGB', (400, 400), color='white')
        plt.figure(figsize=(10, 10))
        plt.imshow(blank_img)
        plt.text(0.5, 0.5, "No image provided", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=20)
        plt.axis('off')
        output_path = f"frcnn_blank_output_{uuid.uuid4()}.png"
        plt.savefig(output_path)
        plt.close()
        return output_path, 0

    try:
        threshold = float(threshold) if threshold is not None else 0.5
        image = image.convert('RGB')
        img_array = np.array(image).astype(np.float32) / 255.0
        transform = frcnn_weights.transforms()
        image_tensor = transform(Image.fromarray((img_array * 255).astype(np.uint8))).unsqueeze(0)

        with torch.no_grad():
            prediction = frcnn_model(image_tensor)[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        valid_detections = sum(1 for score in scores if score >= threshold)

        image_np = np.array(image)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        ax = plt.gca()

        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                x1, y1, x2, y2 = box
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
                class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
                ax.text(x1, y1, f'{class_name}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

        plt.axis('off')
        plt.tight_layout()
        output_path = f"frcnn_output_{uuid.uuid4()}.png"
        plt.savefig(output_path)
        plt.close()
        return output_path, valid_detections
    except Exception as e:
        error_img = Image.new('RGB', (400, 400), color='white')
        plt.figure(figsize=(10, 10))
        plt.imshow(error_img)
        plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12, wrap=True)
        plt.axis('off')
        error_path = f"frcnn_error_output_{uuid.uuid4()}.png"
        plt.savefig(error_path)
        plt.close()
        return error_path, 0

def detect_objects_detr(image, threshold=0.9):
    """Run DETR detection."""
    if image is None:
        blank_img = Image.new('RGB', (400, 400), color='white')
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(blank_img)
        ax.text(0.5, 0.5, "No image provided", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=20)
        plt.axis('off')
        output_path = f"detr_blank_output_{uuid.uuid4()}.png"
        plt.savefig(output_path)
        plt.close(fig)
        return output_path, 0

    try:
        image = image.convert('RGB')
        inputs = detr_processor(images=image, return_tensors="pt")
        outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        valid_detections = len(results["scores"])

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none'))
            ax.text(xmin, ymin, f"{detr_model.config.id2label[label.item()]}: {round(score.item(), 2)}",
                    bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8)

        plt.axis('off')
        output_path = f"detr_output_{uuid.uuid4()}.png"
        plt.savefig(output_path)
        plt.close(fig)
        return output_path, valid_detections
    except Exception as e:
        error_img = Image.new('RGB', (400, 400), color='white')
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(error_img)
        ax.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, wrap=True)
        plt.axis('off')
        error_path = f"detr_error_output_{uuid.uuid4()}.png"
        plt.savefig(error_path)
        plt.close(fig)
        return error_path, 0

def detect_objects_maskrcnn(image, threshold=0.5):
    """Run Mask R-CNN detection and segmentation."""
    if image is None:
        blank_img = Image.new('RGB', (400, 400), color='white')
        plt.figure(figsize=(10, 10))
        plt.imshow(blank_img)
        plt.text(0.5, 0.5, "No image provided", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=20)
        plt.axis('off')
        output_path = f"maskrcnn_blank_output_{uuid.uuid4()}.png"
        plt.savefig(output_path)
        plt.close()
        return output_path, 0

    try:
        image = image.convert('RGB')
        transform = torchvision.transforms.ToTensor()
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = maskrcnn_model(img_tensor)[0]

        masks = output['masks']
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        valid_detections = sum(1 for score in scores if score >= threshold)

        image_np = np.array(image).copy()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)

        for i in range(len(masks)):
            if scores[i] >= threshold:
                mask = masks[i, 0].cpu().numpy()
                mask = mask > 0.5
                color = np.random.rand(3)
                colored_mask = np.zeros_like(image_np, dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = mask * int(color[c] * 255)
                image_np = np.where(mask[:, :, None], 0.5 * image_np + 0.5 * colored_mask, image_np).astype(np.uint8)

                x1, y1, x2, y2 = boxes[i]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2))
                label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
                ax.text(x1, y1, f"{label}: {scores[i]:.2f}", bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)

        ax.imshow(image_np)
        ax.axis('off')
        output_path = f"maskrcnn_output_{uuid.uuid4()}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return output_path, valid_detections
    except Exception as e:
        error_img = Image.new('RGB', (400, 400), color='white')
        plt.figure(figsize=(10, 10))
        plt.imshow(error_img)
        plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12, wrap=True)
        plt.axis('off')
        error_path = f"maskrcnn_error_output_{uuid.uuid4()}.png"
        plt.savefig(error_path)
        plt.close()
        return error_path, 0

def detect_objects_mask2former(image, threshold=0.5):
    """Run Mask2Former detection and segmentation."""
    if image is None:
        blank_img = Image.new('RGB', (400, 400), color='white')
        plt.figure(figsize=(10, 10))
        plt.imshow(blank_img)
        plt.text(0.5, 0.5, "No image provided", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=20)
        plt.axis('off')
        output_path = f"mask2former_blank_output_{uuid.uuid4()}.png"
        plt.savefig(output_path)
        plt.close()
        return output_path, 0

    try:
        image = image.convert('RGB')
        inputs = mask2former_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = mask2former_model(**inputs)

        results = mask2former_processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        segmentation_map = results["segmentation"].cpu().numpy()
        segments_info = results["segments_info"]

        valid_detections = sum(1 for segment in segments_info if segment.get("score", 1.0) >= threshold)

        image_np = np.array(image).copy()
        overlay = image_np.copy()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)

        for segment in segments_info:
            score = segment.get("score", 1.0)
            if score < threshold:
                continue
            segment_id = segment["id"]
            label_id = segment["label_id"]
            mask = segmentation_map == segment_id
            color = np.random.rand(3)
            overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 255 * 0.5).astype(np.uint8)

            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x1, x2 = x_indices.min(), x_indices.max()
            y1, y2 = y_indices.min(), y_indices.max()

            label_name = MASK2FORMER_COCO_NAMES.get(str(label_id), str(label_id))
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2))
            ax.text(x1, y1, f"{label_name}: {score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)

        ax.imshow(overlay)
        ax.axis('off')
        output_path = f"mask2former_output_{uuid.uuid4()}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return output_path, valid_detections
    except Exception as e:
        error_img = Image.new('RGB', (400, 400), color='white')
        plt.figure(figsize=(10, 10))
        plt.imshow(error_img)
        plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12, wrap=True)
        plt.axis('off')
        error_path = f"mask2former_error_output_{uuid.uuid4()}.png"
        plt.savefig(error_path)
        plt.close()
        return error_path, 0

def update_model_choices(category):
    """Update model choices for prediction radio buttons based on selected category."""
    if category == "Object Detection":
        return gr.update(choices=["ConvNet (Faster R-CNN)", "Transformer (DETR)"], value=None, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif category == "Object Segmentation":
        return gr.update(choices=["ConvNet (Mask R-CNN)", "Transformer (Mask2Former)"], value=None, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    return gr.update(choices=[], visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def analyze_performance(image, category, user_opinion, frcnn_threshold=0.5, detr_threshold=0.9, maskrcnn_threshold=0.5, mask2former_threshold=0.5):
    """Analyze and compare model performance for all models in the selected category."""
    if image is None:
        return "Please upload an image first.", None, None, None, None, "No analysis available."

    frcnn_result = None
    detr_result = None
    maskrcnn_result = None
    mask2former_result = None
    frcnn_count = 0
    detr_count = 0
    maskrcnn_count = 0
    mask2former_count = 0

    if category == "Object Detection":
        frcnn_result, frcnn_count = detect_objects_frcnn(image, frcnn_threshold)
        detr_result, detr_count = detect_objects_detr(image, detr_threshold)
    elif category == "Object Segmentation":
        maskrcnn_result, maskrcnn_count = detect_objects_maskrcnn(image, maskrcnn_threshold)
        mask2former_result, mask2former_count = detect_objects_mask2former(image, mask2former_threshold)

    # Analyze performance
    counts = {}
    model_mapping = {
        "ConvNet (Faster R-CNN)": "ConvNet (Faster R-CNN)",
        "Transformer (DETR)": "Transformer (DETR)",
        "ConvNet (Mask R-CNN)": "ConvNet (Mask R-CNN)",
        "Transformer (Mask2Former)": "Transformer (Mask2Former)"
    }
    if category == "Object Detection":
        counts = {
            "ConvNet (Faster R-CNN)": frcnn_count,
            "Transformer (DETR)": detr_count
        }
    elif category == "Object Segmentation":
        counts = {
            "ConvNet (Mask R-CNN)": maskrcnn_count,
            "Transformer (Mask2Former)": mask2former_count
        }

    max_count = max(counts.values())
    max_models = [model for model, count in counts.items() if count == max_count]

    if len(max_models) == 1:
        analysis = f"Result: {max_models[0]} performed best, identifying {max_count} objects.\n\n"
    else:
        analysis = f"Result: {', '.join(max_models)} performed equally well, each identifying {max_count} objects.\n\n"

    if user_opinion:
        analysis += f"You predicted that {user_opinion} would perform best.\n"
        if user_opinion in max_models:
            analysis += f"Congratulations, your prediction was correct!\n"
        else:
            analysis += f"Your prediction was not correct. {user_opinion} identified {counts[user_opinion]} objects, while {', '.join(max_models)} performed best with {max_count} objects. Please try again with a new image.\n"

    if category == "Object Detection":
        analysis += "\nConvNet (Faster R-CNN) is efficient and reliable for general object identification tasks. Transformer (DETR) excels in complex scenes by leveraging advanced context understanding."
    elif category == "Object Segmentation":
        analysis += "\nConvNet (Mask R-CNN) provides precise object outlines for detailed analysis. Transformer (Mask2Former) often outperforms in complex scenes due to its advanced architecture."

    # Image-specific recommendation
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    pixel_variance = np.var(img_array)

    if height * width > 1000 * 1000:
        analysis += f"\n\nThis high-resolution image benefits from Transformer models, which excel in detailed and complex scenes."
    if pixel_variance > 1000:
        analysis += f"\n\nThis image has high complexity. Transformer models often provide superior results in such cases."
    if height * width < 500 * 500:
        analysis += f"\n\nFor smaller images, ConvNet models often deliver reliable results with lower computational demands."
    if category == "Object Segmentation" and max_count > 0:
        analysis += "\n\nFor detailed outlining tasks, Transformer (Mask2Former) may be preferable for complex scenes due to its advanced design."

    # Enhanced result formatting
    if user_opinion and user_opinion in max_models:
        celebration = "🎉✨"
        analysis = analysis.replace("Congratulations", f"{celebration} EPIC WIN! {celebration}")
        analysis = analysis.replace("!\n", "! 🥳\n")
        analysis += "\n\n🌟 You've mastered the AI showdown! 🌟"
    elif user_opinion:
        analysis = analysis.replace("try again", "try again 💪")

    # Convert to HTML with styling
    html_analysis = f"""
    <div class="{'celebrate' if user_opinion in max_models else ''}" style="margin: 15px 0;">
        <h3 style='color: {"#4CAF50" if user_opinion in max_models else "#f44336"}; margin-bottom: 15px;'>
            {"🏆 " + max_models[0] + " Dominates!" if len(max_models) == 1 else "⚔️ Tie Battle!"}
        </h3>
        <div style="background: var(--background-fill-primary); padding: 20px; border-radius: 10px; 
                    white-space: pre-wrap; overflow-wrap: break-word; color: var(--text-color);">
            {analysis}
        </div>
    </div>
    """
    return "Analysis complete!", frcnn_result, detr_result, maskrcnn_result, mask2former_result, html_analysis

# Create Gradio interface with enhanced design
with gr.Blocks(title="AI Vision Showdown", theme=gr.themes.Default(primary_hue="emerald", secondary_hue="blue")) as app:
    gr.Markdown("""
    # 🎯 AI Vision Showdown: ConvNets vs Transformers
    ### 🤖 Battle of the algorithms! Upload an image and predict which AI will dominate!
    """)
    
    # Enhanced CSS
    gr.HTML("""
    <style>
        @keyframes celebrate {
            0% { transform: rotate(0deg); }
            25% { transform: rotate(5deg); }
            50% { transform: rotate(-5deg); }
            75% { transform: rotate(5deg); }
            100% { transform: rotate(0deg); }
        }
        .celebrate { animation: celebrate 0.5s ease-in-out; }
        .battle-card {
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            background: var(--background-fill-primary);
            border: 1px solid var(--border-color-primary);
        }
        .analysis-box {
            background: var(--background-fill-secondary) !important;
            color: var(--text-color) !important;
            padding: 20px;
            border-radius: 10px;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        .loading-status {
            padding: 15px;
            background: var(--background-fill-secondary);
            border-radius: 8px;
            margin: 10px 0;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """)

    # State variables
    image_state = gr.State(None)
    category_state = gr.State(None)
    loading_status = gr.HTML(visible=False)

    # Top Section: Inputs
    with gr.Row(variant="battle-card"):
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📤 Image Upload Zone")
            image_input = gr.Image(type="pil", label="Drag & Drop Your Challenge Image")
            upload_button = gr.Button("🔼 Upload Challenge Image", variant="primary")

        with gr.Column(scale=1, min_width=300):
            with gr.Group(visible=False) as prediction_selection:
                gr.Markdown("## 🔮 Prediction Arena")
                category_choice = gr.Radio(
                    choices=["Object Detection", "Object Segmentation"],
                    label="⚔️ Select Battle Ground",
                    value=None,
                    elem_classes="battle-card"
                )
                user_opinion = gr.Radio(
                    choices=[],
                    label="🏹 Predict the Victor",
                    value=None,
                    visible=False,
                    elem_classes="battle-card"
                )
                
                # Enhanced threshold controls
                with gr.Accordion("🎚️ Advanced Battle Parameters", open=False):
                    frcnn_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Faster R-CNN Confidence (Speed Demon 🏎️)",
                        visible=False
                    )
                    detr_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                        label="DETR Confidence (Attention Master 🔍)",
                        visible=False
                    )
                    maskrcnn_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Mask R-CNN Confidence (Precision Expert ✂️)",
                        visible=False
                    )
                    mask2former_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                        label="Mask2Former Confidence (Transformer Champ 🤖)",
                        visible=False
                    )

                detect_button = gr.Button("⚔️ Start Showdown", variant="primary")

    # Results Section
    with gr.Group(visible=False) as outputs_panel:
        gr.Markdown("## 📊 Battle Results")
        with gr.Tabs():
            with gr.TabItem("Object Detection Warriors", visible=False) as detection_tab:
                with gr.Row():
                    frcnn_result = gr.Image(type="filepath", label="🚀 Faster R-CNN (ConvNet Champion)", elem_classes="battle-card")
                    detr_result = gr.Image(type="filepath", label="🧠 DETR (Transformer Visionary)", elem_classes="battle-card")
            
            with gr.TabItem("Segmentation Gladiators", visible=False) as segmentation_tab:
                with gr.Row():
                    maskrcnn_result = gr.Image(type="filepath", label="⚔️ Mask R-CNN (Pixel Perfect)", elem_classes="battle-card")
                    mask2former_result = gr.Image(type="filepath", label="🛡️ Mask2Former (Segmentation Master)", elem_classes="battle-card")

    # Analysis Section
    with gr.Group(visible=False) as results_panel:
        gr.Markdown("## 🏆 Battle Report")
        analysis_output = gr.HTML(label="Victory Analysis", elem_classes="battle-card")
        restart_button = gr.Button("🔄 New Challenge", variant="secondary")

    # Upload button click event
    def upload_image(img):
        if img is None:
            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        return img, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    upload_button.click(
        fn=upload_image,
        inputs=[image_input],
        outputs=[image_state, prediction_selection, outputs_panel, results_panel]
    )

    # Category selection event
    def update_prediction_options(category):
        if category is None:
            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        model_update, frcnn_vis, detr_vis, maskrcnn_vis, mask2former_vis = update_model_choices(category)
        return category, model_update, frcnn_vis, detr_vis, maskrcnn_vis, mask2former_vis

    category_choice.change(
        fn=update_prediction_options,
        inputs=[category_choice],
        outputs=[category_state, user_opinion, frcnn_threshold, detr_threshold, maskrcnn_threshold, mask2former_threshold]
    )

    # Detect button click event
    def run_detection(image, category, user_opinion, frcnn_threshold, detr_threshold, maskrcnn_threshold, mask2former_threshold):
        if not category or not user_opinion:
            return "Please select a category and prediction.", None, None, None, None, "No analysis available.", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        def analyze_with_progress(progress=gr.Progress()):
            progress(0.1, desc="⚙️ Models are gearing up...")
            result = analyze_performance(image, category, user_opinion, frcnn_threshold, detr_threshold, maskrcnn_threshold, mask2former_threshold)
            progress(1.0, desc="✅ Battle complete!")
            return result
        
        try:
            message, frcnn_result_img, detr_result_img, maskrcnn_result_img, mask2former_result_img, html_analysis = analyze_with_progress()
            return [
                message,
                gr.update(value=frcnn_result_img, visible=category == "Object Detection"),
                gr.update(value=detr_result_img, visible=category == "Object Detection"),
                gr.update(value=maskrcnn_result_img, visible=category == "Object Segmentation"),
                gr.update(value=mask2former_result_img, visible=category == "Object Segmentation"),
                html_analysis,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=category == "Object Detection"),
                gr.update(visible=category == "Object Segmentation"),
                gr.update(visible=False)
            ]
        except Exception as e:
            return [f"Error: {str(e)}"] + [gr.update()]*9 + [gr.update(visible=False)]

    detect_button.click(
        fn=run_detection,
        inputs=[image_state, category_state, user_opinion, frcnn_threshold, detr_threshold, maskrcnn_threshold, mask2former_threshold],
        outputs=[gr.Textbox(visible=False), frcnn_result, detr_result, maskrcnn_result, mask2former_result, 
                analysis_output, outputs_panel, results_panel, detection_tab, segmentation_tab, loading_status]
    )

    # Restart button click event
    def restart():
        return None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    restart_button.click(
        fn=restart,
        inputs=[],
        outputs=[image_state, category_state, prediction_selection, outputs_panel, results_panel, frcnn_result, detr_result, maskrcnn_result, mask2former_result, analysis_output, user_opinion, category_choice, detection_tab, segmentation_tab]
    )

    # Example images
    example_images = [
        os.path.join(os.getcwd(), "TEST_IMG_1.jpg"),
        os.path.join(os.getcwd(), "TEST_IMG_2.JPG"),
        os.path.join(os.getcwd(), "TEST_IMG_3.jpg"),
        os.path.join(os.getcwd(), "TEST_IMG_4.jpg")
    ]

    valid_examples = [img for img in example_images if os.path.exists(img)]

    if valid_examples:
        gr.Markdown("## 🧩 Try These Example Challenges:")
        gr.Examples(
            examples=valid_examples,
            inputs=image_input,
            examples_per_page=4,
            label=""
        )

if __name__ == "__main__":
    app.launch(debug=True)