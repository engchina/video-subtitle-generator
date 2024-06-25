import uuid

import gradio as gr

import os

from dotenv import load_dotenv, find_dotenv

from utils import whisper_utils, translation_utils, ffmpeg_utils

# read local .env file
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]


def generate(input_video, use_exist_srt, uploaded_srt, whisper_model, use_translation, source_lang='Chinese',
             target_lang="English",
             font_size=18, margin_v=4, merge_to_video=False,
             use_trim=False):
    video_uuid = uuid.uuid4()

    # Step 1: Prepare video
    trimmed_video = input_video
    if use_trim:
        trimmed_video = f"/tmp/trimmed_{video_uuid}.mp4"
        ffmpeg_utils.trim_video(input_video, trimmed_video, duration=30)

    # Step 2: Generate subtitles using Whisper
    if use_exist_srt:
        srt_file = uploaded_srt.name
    else:
        srt_file = f"/tmp/generated_{video_uuid}.srt"
        whisper_utils.generate_subtitles(trimmed_video, srt_file, False, whisper_model)

    if use_translation:
        # Step 3 & 4: Translate subtitles using gpt-4 via LangChain
        translated_srt = f"/tmp/translated_ast_{video_uuid}.srt"
        translation_utils.translate_subtitles(srt_file, translated_srt, source_lang, target_lang)
    else:
        translated_srt = srt_file

    # Step 5: Add subtitles to video using FFmpeg
    if merge_to_video:
        output_video = f"/tmp/output_{video_uuid}.mp4"
        ffmpeg_utils.add_subtitles_to_video(trimmed_video, translated_srt, output_video, font_size, margin_v)
    else:
        output_video = trimmed_video
    print(f"Video with subtitles generated: {output_video}")
    return gr.Video(value=output_video), gr.File(value=translated_srt)


with gr.Blocks() as app:
    gr.Markdown("# Generate Multilingual Subtitles for Video")
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
        with gr.Column():
            output_video = gr.Video(label="Output Video", show_download_button=True)
    with gr.Row():
        with gr.Column():
            use_exist_srt_radio = gr.Radio(choices=[("Generate srt file by AI", False), ("Use exist srt file", True)],
                                           value=False, show_label=False)
        with gr.Column():
            uploaded_srt_file = gr.File(label="srt File", type="filepath", file_types=[".srt"], interactive=False)
    with gr.Row():
        with gr.Column():
            whisper_model_radio = gr.Radio(label="OpenAI Whisper Model", choices=["large", "large-v2", "large-v3"],
                                           value="large-v3")
    with gr.Accordion(label="Use Translation", open=True, visible=True):
        with gr.Row():
            with gr.Column():
                use_translation_checkbox = gr.Checkbox(label="Use Translation", show_label=False, value=False)
            with gr.Column():
                source_lang_dropdown = gr.Dropdown(label="Source Lang", choices=["Chinese", "English", "Japanese"],
                                                   value="Chinese")
            with gr.Column():
                target_lang_dropdown = gr.Dropdown(label="Target Lang", choices=["Chinese", "English", "Japanese"],
                                                   value="English")
    with gr.Row():
        with gr.Column():
            font_size_slider = gr.Slider(label="Font Size", minimum=16, maximum=32, step=2, value=18)
        with gr.Column():
            margin_v_slider = gr.Slider(label="Margin-V", minimum=0, maximum=80, step=2, value=8)
    with gr.Row():
        with gr.Column():
            merge_to_video_checkbox = gr.Checkbox(label="Merge to Video", show_label=True, value=False)
    with gr.Row():
        with gr.Column():
            use_trim_checkbox = gr.Checkbox(label="Use Trim - Test first 30 seconds", show_label=True, value=True)
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
    use_exist_srt_radio.change(
        lambda x: gr.File(interactive=True) if use_exist_srt_radio else gr.File(
            interactive=False), use_exist_srt_radio, [uploaded_srt_file])
    generate_btn.click(fn=generate,
                       inputs=[input_video, use_exist_srt_radio, uploaded_srt_file, whisper_model_radio,
                               use_translation_checkbox,
                               source_lang_dropdown, target_lang_dropdown,
                               font_size_slider, margin_v_slider, merge_to_video_checkbox, use_trim_checkbox],
                       outputs=[output_video, uploaded_srt_file])

if __name__ == "__main__":
    app.launch()
