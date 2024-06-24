import uuid

import gradio as gr
import cli


def generate(input, use_translation, source_lang, target_lang, font_size, margin_v, use_trim):
    video_uuid = uuid.uuid4()
    output = f"output/output_{video_uuid}.mp4"
    cli.main(input, output, use_trim, use_translation, source_lang, target_lang, font_size, margin_v)

    return gr.Video(value=output)


with gr.Blocks() as app:
    gr.Markdown("Generate Video Subtitles")
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
        with gr.Column():
            output_video = gr.Video(label="Output Video", show_download_button=True)
    with gr.Accordion(label="Use Translation", open=True):
        with gr.Row():
            with gr.Column():
                use_translation_checkbox = gr.Checkbox(label="Use Translation", show_label=False, value=False)
        with gr.Row():
            with gr.Column():
                source_lang_dropdown = gr.Dropdown(label="Source Lang", choices=["Chinese", "English", "Japanese"],
                                                   value="Chinese")
            with gr.Column():
                target_lang_dropdown = gr.Dropdown(label="Target Lang", choices=["Chinese", "English", "Japanese"],
                                                   value="English")
    with gr.Row():
        with gr.Column():
            font_size_slider = gr.Slider(label="Font Size", minimum=16, maximum=28, step=2, value=18)
        with gr.Column():
            margin_v_slider = gr.Slider(label="Margin-V", minimum=0, maximum=80, step=2, value=2)
    with gr.Row():
        with gr.Column():
            use_trim_checkbox = gr.Checkbox(label="Use Trim", show_label=True, value=True)
    with gr.Row():
        generate_btn = gr.Button("Generate Subtitles", variant="primary")
    generate_btn.click(fn=generate,
                       inputs=[input_video, use_translation_checkbox, source_lang_dropdown, target_lang_dropdown,
                               font_size_slider, margin_v_slider, use_trim_checkbox], outputs=[output_video])

if __name__ == "__main__":
    app.launch()
