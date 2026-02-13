import gradio as gr
import torch
import numpy as np
import supervision as sv
from typing import Iterable
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
import json
import ast
import re
import uuid
from PIL import Image
from spaces import GPU

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.2em !important;
}

/* RadioAnimated Styles */
.ra-wrap{ width: fit-content; }
.ra-inner{
  position: relative; display: inline-flex; align-items: center; gap: 0; padding: 6px;
  background: var(--neutral-200); border-radius: 9999px; overflow: hidden;
}
.ra-input{ display: none; }
.ra-label{
  position: relative; z-index: 2; padding: 8px 16px;
  font-family: inherit; font-size: 14px; font-weight: 600;
  color: var(--neutral-500); cursor: pointer; transition: color 0.2s; white-space: nowrap;
}
.ra-highlight{
  position: absolute; z-index: 1; top: 6px; left: 6px;
  height: calc(100% - 12px); border-radius: 9999px;
  background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.2s, width 0.2s;
}
.ra-input:checked + .ra-label{ color: black; }

/* Dark mode adjustments for Radio */
.dark .ra-inner { background: var(--neutral-800); }
.dark .ra-label { color: var(--neutral-400); }
.dark .ra-highlight { background: var(--neutral-600); }
.dark .ra-input:checked + .ra-label { color: white; }

#gpu-duration-container {
    padding: 10px;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    margin-top: 10px;
}
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "auto"

CATEGORIES = ["Query", "Caption", "Point", "Detect"]

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )

def apply_gpu_duration(val: str):
    return int(val)

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=DTYPE,
    device_map=DEVICE,
    attn_implementation="kernels-community/flash-attn3",
).eval()
qwen_processor = Qwen3VLProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
)

def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def annotate_image(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image

    image = image.convert("RGB")
    original_width, original_height = image.size

    if "points" in result and result["points"]:
        points_list = [
            [int(p["x"] * original_width), int(p["y"] * original_height)]
            for p in result.get("points", [])
        ]
        if not points_list:
            return image

        points_array = np.array(points_list).reshape(1, -1, 2)
        key_points = sv.KeyPoints(xy=points_array)
        vertex_annotator = sv.VertexAnnotator(radius=4, color=sv.Color.RED)
        annotated_image = vertex_annotator.annotate(scene=np.array(image.copy()), key_points=key_points)
        return Image.fromarray(annotated_image)

    if "objects" in result and result["objects"]:
        boxes = []
        for obj in result["objects"]:
            x_min = obj.get("x_min", 0.0) * original_width
            y_min = obj.get("y_min", 0.0) * original_height
            x_max = obj.get("x_max", 0.0) * original_width
            y_max = obj.get("y_max", 0.0) * original_height
            boxes.append([x_min, y_min, x_max, y_max])

        if not boxes:
            return image

        detections = sv.Detections(xyxy=np.array(boxes))

        if len(detections) == 0:
            return image

        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
        annotated_image = box_annotator.annotate(scene=np.array(image.copy()), detections=detections)
        return Image.fromarray(annotated_image)

    return image

def run_qwen_inference(image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=512,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def calc_timeout_process(image: Image.Image, category: str, prompt: str, gpu_timeout: int):
    """Calculate GPU timeout duration for processing."""
    try:
        return int(gpu_timeout)
    except:
        return 60


@GPU(duration=calc_timeout_process)
def process_qwen(image: Image.Image, category: str, prompt: str, gpu_timeout: int = 60):
    if category == "Query":
        return run_qwen_inference(image, prompt), {}
    elif category == "Caption":
        full_prompt = f"Provide a {prompt} length caption for the image."
        return run_qwen_inference(image, full_prompt), {}
    elif category == "Point":
        full_prompt = (
            f"Provide 2d point coordinates for {prompt}. Report in JSON format."
        )
        output_text = run_qwen_inference(image, full_prompt)
        parsed_json = safe_parse_json(output_text)
        points_result = {"points": []}
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if "point_2d" in item and len(item["point_2d"]) == 2:
                    x, y = item["point_2d"]
                    points_result["points"].append({"x": x / 1000.0, "y": y / 1000.0})
        return json.dumps(points_result, indent=2), points_result
    elif category == "Detect":
        full_prompt = (
            f"Provide bounding box coordinates for {prompt}. Report in JSON format."
        )
        output_text = run_qwen_inference(image, full_prompt)
        parsed_json = safe_parse_json(output_text)
        objects_result = {"objects": []}
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                    xmin, ymin, xmax, ymax = item["bbox_2d"]
                    objects_result["objects"].append(
                        {
                            "x_min": xmin / 1000.0,
                            "y_min": ymin / 1000.0,
                            "x_max": xmax / 1000.0,
                            "y_max": ymax / 1000.0,
                        }
                    )
        return json.dumps(objects_result, indent=2), objects_result
    return "Invalid category", {}

def process_inputs(image, category, prompt, gpu_timeout):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt:
        raise gr.Error("Please provide a prompt.")

    image.thumbnail((512, 512))

    qwen_text, qwen_data = process_qwen(image, category, prompt, gpu_timeout)
    qwen_annotated_image = annotate_image(image.copy(), qwen_data)

    return qwen_annotated_image, qwen_text

def on_category_change(category: str):
    if category == "Query":
        return gr.Textbox(placeholder="e.g., Count the total number of boats and describe the environment.")
    elif category == "Caption":
        return gr.Textbox(placeholder="e.g., short, normal, detailed")
    elif category == "Point":
        return gr.Textbox(placeholder="e.g., The gun held by the person.")
    elif category == "Detect":
        return gr.Textbox(placeholder="e.g., The headlight of the car.")
    return gr.Textbox(placeholder="e.g., detect the object.")


with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-3VL: Multimodal Understanding**", elem_id="main-title")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image", height=350)
                category_select = gr.Dropdown(
                    choices=CATEGORIES,
                    value="Query",
                    label="Select Task Category",
                    interactive=True,
                )
                with gr.Row():
                    prompt_input = gr.Textbox(
                        placeholder="e.g., Count the total number of boats and describe the environment.",
                        label="Prompt",
                        lines=3,
                    )
        
                submit_btn = gr.Button("Process Image", variant="primary")

            with gr.Column(scale=2):
                qwen_img_output = gr.Image(label="Output Image")
                qwen_text_output = gr.Textbox(
                    label="Text Output", lines=10, interactive=True)
                
                with gr.Row(elem_id="gpu-duration-container"):
                    with gr.Column():
                        gr.Markdown("**GPU Duration (seconds)**")
                        radioanimated_gpu_duration = RadioAnimated(
                            choices=["60", "90", "120", "180", "240", "300"],
                            value="60",
                            elem_id="radioanimated_gpu_duration"
                        )
                        gpu_duration_state = gr.Number(value=60, visible=False)
                
                gr.Markdown("*Note: Higher GPU duration allows for longer processing but consumes more GPU quota.*")
            
        gr.Examples(
            examples=[
                ["examples/5.jpg", "Point", "Detect the children who are out of focus and wearing a white T-shirt."],
                ["examples/5.jpg", "Detect", "Point out the out-of-focus (all) children."],
                ["examples/4.jpg", "Detect", "Headlight"],
                ["examples/3.jpg", "Point", "Gun"],
                ["examples/1.jpg", "Query", "Count the total number of boats and describe the environment."],
                ["examples/2.jpg", "Caption", "a brief"],
            ],
            inputs=[image_input, category_select, prompt_input],
        )

    radioanimated_gpu_duration.change(
        fn=apply_gpu_duration,
        inputs=radioanimated_gpu_duration,
        outputs=[gpu_duration_state],
        api_visibility="private"
    )

    category_select.change(
        fn=on_category_change,
        inputs=[category_select],
        outputs=[prompt_input],
    )

    submit_btn.click(
        fn=process_inputs,
        inputs=[image_input, category_select, prompt_input, gpu_duration_state],
        outputs=[qwen_img_output, qwen_text_output],
    )

if __name__ == "__main__":
    demo.launch(css=css, theme=steel_blue_theme, mcp_server=True, ssr_mode=False, show_error=True)
