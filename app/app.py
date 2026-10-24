"""Gradio label demo — text paragraph or image → tag suggestions with confidence."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import gradio as gr
from PIL import UnidentifiedImageError

from app.inference.text import predict_text_top_tags
from app.inference.image import predict_image_top_tags

CONFIDENCE_THRESHOLD = 0.35
TOP_K = 3
OTHER = "Other"


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_choices(top_tags: list[tuple[str, float]]) -> tuple[list[str], str]:
    """Format radio choices 'Label (XX%)' and pick default (Other if below threshold)."""
    choices = [f"{label} ({conf * 100:.0f}%)" for label, conf in top_tags] + [OTHER]
    best_conf = top_tags[0][1] if top_tags else 0.0
    default = choices[0] if best_conf >= CONFIDENCE_THRESHOLD else OTHER
    return choices, default


def _history_df(history: list[dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["#", "Time", "Type", "Preview", "Tag"])
    rows = [
        {
            "#": i + 1,
            "Time": h["time"],
            "Type": h["type"],
            "Preview": h["preview"],
            "Tag": h["tag"],
        }
        for i, h in enumerate(history)
    ]
    return pd.DataFrame(rows)


# ── prediction handlers ───────────────────────────────────────────────────────

def suggest_text_tags(text: str):
    if not text or not text.strip():
        return (
            gr.update(choices=[], value=None, visible=False),
            gr.update(value="Enter some text first.", visible=True),
            gr.update(visible=False),
        )
    try:
        top_tags = predict_text_top_tags(text.strip(), top_k=TOP_K)
        choices, default = _build_choices(top_tags)
        best_label, best_conf = top_tags[0]
        note = f"**{best_label}** at {best_conf * 100:.0f}%"
        if best_conf < CONFIDENCE_THRESHOLD:
            note += f" — below {CONFIDENCE_THRESHOLD * 100:.0f}% threshold, defaulting to **Other**"
        return (
            gr.update(choices=choices, value=default, visible=True),
            gr.update(value=note, visible=True),
            gr.update(visible=True),
        )
    except FileNotFoundError as e:
        return (
            gr.update(choices=[], visible=False),
            gr.update(value=f"Model not found — {e}", visible=True),
            gr.update(visible=False),
        )
    except Exception as e:
        return (
            gr.update(choices=[], visible=False),
            gr.update(value=f"Error: {e}", visible=True),
            gr.update(visible=False),
        )


def suggest_image_tags(image):
    if image is None:
        return (
            gr.update(choices=[], value=None, visible=False),
            gr.update(value="Upload an image first.", visible=True),
            gr.update(visible=False),
        )
    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        top_tags = predict_image_top_tags(image, top_k=TOP_K)
        choices, default = _build_choices(top_tags)
        best_label, best_conf = top_tags[0]
        note = f"**{best_label}** at {best_conf * 100:.0f}%"
        if best_conf < CONFIDENCE_THRESHOLD:
            note += f" — below {CONFIDENCE_THRESHOLD * 100:.0f}% threshold, defaulting to **Other**"
        return (
            gr.update(choices=choices, value=default, visible=True),
            gr.update(value=note, visible=True),
            gr.update(visible=True),
        )
    except (UnidentifiedImageError, OSError) as e:
        return (
            gr.update(choices=[], visible=False),
            gr.update(value=f"Unsupported image: {e}", visible=True),
            gr.update(visible=False),
        )
    except FileNotFoundError as e:
        return (
            gr.update(choices=[], visible=False),
            gr.update(value=f"Model not found — {e}", visible=True),
            gr.update(visible=False),
        )
    except Exception as e:
        return (
            gr.update(choices=[], visible=False),
            gr.update(value=f"Error: {e}", visible=True),
            gr.update(visible=False),
        )


# ── publish handler ───────────────────────────────────────────────────────────

def publish_text(text: str, tag_choice: str, history: list[dict]):
    if not tag_choice:
        return history, _history_df(history), gr.update(value="Select a tag first.", visible=True)
    label = tag_choice.split(" (")[0] if " (" in tag_choice else tag_choice
    preview = (text or "")[:80].replace("\n", " ")
    if len(text or "") > 80:
        preview += "…"
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "type": "text", "preview": preview, "tag": label}
    history = history + [entry]
    return history, _history_df(history), gr.update(value=f"Published as **{label}**.", visible=True)


def publish_image(image: Any, tag_choice: str, history: list[dict]):
    if not tag_choice:
        return history, _history_df(history), gr.update(value="Select a tag first.", visible=True)
    label = tag_choice.split(" (")[0] if " (" in tag_choice else tag_choice
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "type": "image", "preview": "[uploaded image]", "tag": label}
    history = history + [entry]
    return history, _history_df(history), gr.update(value=f"Published as **{label}**.", visible=True)


def clear_history(history: list[dict]):
    return [], _history_df([])


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Auto-Label Demo", theme=gr.themes.Soft()) as demo:
    history_state = gr.State([])

    gr.Markdown(
        "# Auto-Label Demo\n"
        "Upload a **text snippet** or an **animal image** — the model suggests the top tags. "
        f"Tags with confidence below **{CONFIDENCE_THRESHOLD * 100:.0f}%** are auto-assigned **Other**. "
        "Confirm or change the tag, then publish."
    )

    with gr.Tab("Text"):
        text_input = gr.Textbox(
            label="News paragraph",
            placeholder="Paste a news article excerpt here…",
            lines=5,
        )
        text_suggest_btn = gr.Button("Suggest Tags", variant="secondary")
        text_status = gr.Markdown("", visible=False)
        text_radio = gr.Radio(
            choices=[], label="Suggested tags — pick one", visible=False, interactive=True
        )
        text_publish_btn = gr.Button("Publish", variant="primary", visible=False)
        text_pub_status = gr.Markdown("", visible=False)

        text_suggest_btn.click(
            fn=suggest_text_tags,
            inputs=text_input,
            outputs=[text_radio, text_status, text_publish_btn],
        )

    with gr.Tab("Image"):
        image_input = gr.Image(label="Upload animal image", type="pil")
        img_suggest_btn = gr.Button("Suggest Tags", variant="secondary")
        img_status = gr.Markdown("", visible=False)
        img_radio = gr.Radio(
            choices=[], label="Suggested tags — pick one", visible=False, interactive=True
        )
        img_publish_btn = gr.Button("Publish", variant="primary", visible=False)
        img_pub_status = gr.Markdown("", visible=False)

        img_suggest_btn.click(
            fn=suggest_image_tags,
            inputs=image_input,
            outputs=[img_radio, img_status, img_publish_btn],
        )

    gr.Markdown("---\n## Published Items")
    history_df = gr.Dataframe(
        value=_history_df([]),
        label="All tagged items (this session)",
        interactive=False,
        wrap=True,
    )
    clear_btn = gr.Button("Clear history", variant="stop", size="sm")

    # Wire publish buttons to the history dataframe (re-declared after df exists)
    text_publish_btn.click(
        fn=publish_text,
        inputs=[text_input, text_radio, history_state],
        outputs=[history_state, history_df, text_pub_status],
    )
    img_publish_btn.click(
        fn=publish_image,
        inputs=[image_input, img_radio, history_state],
        outputs=[history_state, history_df, img_pub_status],
    )
    clear_btn.click(
        fn=clear_history,
        inputs=history_state,
        outputs=[history_state, history_df],
    )


if __name__ == "__main__":
    demo.launch()
