import uuid

import gradio as gr

import os

from dotenv import load_dotenv, find_dotenv
from gradio import Video
from gradio.utils import NamedString

from utils import whisper_utils, translation_utils, summarization_utils, ffmpeg_utils

# read local .env file
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]


def replace_in_file(file_path, replace_or_delete_list):
    if replace_or_delete_list and len(replace_or_delete_list) > 1:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        replace_or_delete_texts = replace_or_delete_list.split("|")
        for replace_or_delete_text in replace_or_delete_texts:
            key = replace_or_delete_text.split('=')[0]
            value = replace_or_delete_text.split('=')[-1]
            content = content.replace(key, "" if key == value else value)

        # 将修改后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)


def generate_srt(input_video, use_exist_srt, uploaded_srt, whisper_model, use_trim=False, replace_list="",
                 delete_list=""):
    video_uuid = uuid.uuid4()

    # Step 1: Prepare video
    trimmed_video = input_video
    if use_trim:
        trimmed_video = f"/tmp/trimmed_{video_uuid}.mp4"
        ffmpeg_utils.trim_video(input_video, trimmed_video, duration=30)

    # Step 2: Generate subtitles using Whisper
    if use_exist_srt:
        if isinstance(uploaded_srt, NamedString):
            uploaded_srt = uploaded_srt.name
        srt_file = uploaded_srt
    else:
        srt_file = f"/tmp/generated_{video_uuid}.srt"
        whisper_utils.generate_subtitles(trimmed_video, srt_file, False, whisper_model)

    replace_in_file(srt_file, replace_list)
    replace_in_file(srt_file, delete_list)

    return gr.Video(value=trimmed_video), gr.File(value=srt_file)


def translate(uploaded_srt, use_translation, translation_prompt,
              translation_source_lang='Chinese', translation_target_lang="English", replace_list="", delete_list=""):
    video_uuid = uuid.uuid4()
    translated_srt = f"/tmp/translated_ast_{video_uuid}.srt"
    if use_translation:
        if isinstance(uploaded_srt, NamedString):
            uploaded_srt = uploaded_srt.name
        # Step 3 & 4: Translate subtitles using gpt-4 via LangChain
        translation_utils.translate_subtitles(uploaded_srt, translated_srt, translation_prompt, translation_source_lang,
                                              translation_target_lang)
        replace_in_file(translated_srt, replace_list)
        replace_in_file(translated_srt, delete_list)
    else:
        with open(translated_srt, 'w'):
            pass
    return gr.File(value=translated_srt)


def summarize(uploaded_srt, summarization_txt,
              summarization_prompt,
              summarization_source_lang="Chinese", summarization_target_lang="English", replace_list="",
              delete_list=""):
    video_uuid = uuid.uuid4()
    summarized_txt = f"/tmp/summarized_{video_uuid}.txt"
    if summarization_txt:
        if isinstance(uploaded_srt, NamedString):
            uploaded_srt = uploaded_srt.name
        summarization_utils.summarize_subtitles(uploaded_srt, summarized_txt, summarization_prompt,
                                                summarization_source_lang,
                                                summarization_target_lang)
        replace_in_file(summarized_txt, replace_list)
        replace_in_file(summarized_txt, delete_list)
    else:
        with open(summarized_txt, 'w'):
            pass
    return gr.File(value=summarized_txt)


def generate_video(output, uploaded_srt, use_translation, translated_srt,
                   font_size=18, margin_v=4, merge_to_video=False):
    video_uuid = uuid.uuid4()
    if not use_translation:
        translated_srt = uploaded_srt
    # Step 5: Add subtitles to video using FFmpeg
    if merge_to_video:
        if isinstance(uploaded_srt, NamedString):
            translated_srt = translated_srt.name
        generated_output_video = f"/tmp/output_{video_uuid}.mp4"
        ffmpeg_utils.add_subtitles_to_video(output, translated_srt, generated_output_video, font_size, margin_v)
    else:
        generated_output_video = output
    print(f"Video with subtitles generated: {generated_output_video}")
    return gr.Video(value=generated_output_video)


TRANSLATION_PROMPT = """You are a translation expert, particularly skilled at translating {source_lang} into idiomatic {target_lang} expressions. I will provide a statement in {source_lang}, and you should directly output the corresponding idiomatic {target_lang} translation.
There is no need to output any other unrelated language.

**{source_lang} statement**:
```
{text}
```

**Idiomatic {target_lang} translation**:\n
"""

SUMMARIZATION_PROMPT = """You are an expert meeting minutes organizer. Your task is to take raw meeting notes in {source_lang} and transform them into a professional, well-structured document in {target_lang}. Follow these guidelines:

1. Analyze the provided meeting record carefully. It includes a timeline but lacks explicit speaker identification.

2. Based on the content and timeline of each statement, deduce who the likely speaker is. Use contextual clues, speaking patterns, and roles mentioned to make educated guesses about the speakers.

3. Organize the meeting minutes in a clear, chronological format. Include the following elements:
   - Meeting title
   - Date and time
   - Attendees (as deduced from the conversation)
   - Agenda items (if discernible from the discussion)
   - Main discussion points
   - Action items or decisions made
   - Next steps or follow-up tasks

4. For each discussion point or decision, include:
   - The timestamp from the original notes
   - Your best guess at who the speaker is (e.g., "Speaker 1 (likely Project Manager)")
   - The content of their statement or contribution

5. If there are any unclear or ambiguous parts in the original notes, make a note of these in [brackets] and suggest possible interpretations or clarifications.

6. At the end of the document, include a section titled "Additional Observations" where you can add any relevant insights, patterns, or important points that may not be explicitly stated in the original notes.

7. Use professional language throughout and ensure the document is well-formatted and easy to read.

8. If any key information seems to be missing (e.g., meeting duration, specific project names, or expected outcomes), note these as "Recommended Additions" at the end of the document.

9. Provide response in {target_lang}.

Please take the raw meeting notes I provide and transform them into a professional meeting minutes document following these guidelines. If you have any questions or need clarification on any part of the notes, please ask. 

**Raw meeting notes**:
```
 {text}
```

**Summarization**:\n
"""

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
    with gr.Row():
        with gr.Column():
            uploaded_srt_file = gr.File(label="SRT File", type="filepath", file_types=[".srt"], interactive=False)
    with gr.Row():
        with gr.Column():
            whisper_model_radio = gr.Radio(label="OpenAI Whisper Model", choices=["large", "large-v2", "large-v3"],
                                           value="large-v3")
    with gr.Accordion(label="Use Translation", open=False, visible=True):
        with gr.Row():
            with gr.Column():
                use_translation_checkbox = gr.Checkbox(label="Use Translation", show_label=False, value=False)
        with gr.Row():
            with gr.Column():
                translation_source_lang_dropdown = gr.Dropdown(label="Source Lang",
                                                               choices=["Chinese", "English", "Japanese"],
                                                               value="Chinese")
            with gr.Column():
                translation_target_lang_dropdown = gr.Dropdown(label="Target Lang",
                                                               choices=["Chinese", "English", "Japanese"],
                                                               value="English")
        with gr.Row():
            with gr.Column():
                translation_srt_file = gr.File(label="Translated SRT File", type="filepath", file_types=[".srt"],
                                               interactive=False)
        with gr.Accordion(label="Translation Prompt", open=False):
            translation_prompt_text = gr.Textbox(label="Translation Prompt", show_label=False, lines=10,
                                                 interactive=True,
                                                 value=TRANSLATION_PROMPT)

    with gr.Accordion(label="Generate Summarization", open=False, visible=True):
        with gr.Row():
            with gr.Column():
                generate_summarization_checkbox = gr.Checkbox(label="Generate Summarization", show_label=False,
                                                              value=False)
        with gr.Row():
            with gr.Column():
                summarization_source_lang_dropdown = gr.Dropdown(label="Source Lang",
                                                                 choices=["Chinese", "English", "Japanese"],
                                                                 value="Chinese")
            with gr.Column():
                summarization_target_lang_dropdown = gr.Dropdown(label="Target Lang",
                                                                 choices=["Chinese", "English", "Japanese"],
                                                                 value="English")
        with gr.Row():
            with gr.Column():
                summarization_txt_file = gr.File(label="Summarized TXT File", type="filepath", file_types=[".txt"],
                                                 interactive=False)
        with gr.Accordion(label="Summarization Prompt", open=False):
            summarization_prompt_text = gr.Textbox(label="Summarization Prompt", show_label=False, lines=30,
                                                   interactive=True, value=SUMMARIZATION_PROMPT)

    with gr.Row():
        with gr.Column():
            merge_to_video_checkbox = gr.Checkbox(label="Merge to Video", show_label=True, value=False)
    with gr.Row():
        with gr.Column():
            font_size_slider = gr.Slider(label="Font Size", minimum=16, maximum=32, step=2, value=24)
        with gr.Column():
            margin_v_slider = gr.Slider(label="Margin-V", minimum=0, maximum=80, step=2, value=20)
    with gr.Row():
        with gr.Column():
            replace_list_text = gr.Textbox(label="Replace List", placeholder="foo=abc|bar=xyz", info="Split by |")
        with gr.Column():
            delete_list_text = gr.Textbox(label="Delete List", placeholder="foo|bar", info="Split by |")
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        with gr.Column():
            use_trim_checkbox = gr.Checkbox(label="Use Trim - Test first 30 seconds", show_label=True, value=False)
    use_exist_srt_radio.change(
        lambda x: gr.File(interactive=True) if use_exist_srt_radio else gr.File(
            interactive=False), use_exist_srt_radio, [uploaded_srt_file])
    generate_btn.click(fn=generate_srt,
                       inputs=[input_video, use_exist_srt_radio, uploaded_srt_file, whisper_model_radio,
                               use_trim_checkbox,
                               replace_list_text,
                               delete_list_text],
                       outputs=[output_video, uploaded_srt_file]).then(translate,
                                                                       [uploaded_srt_file, use_translation_checkbox,
                                                                        translation_prompt_text,
                                                                        translation_source_lang_dropdown,
                                                                        translation_target_lang_dropdown,
                                                                        replace_list_text,
                                                                        delete_list_text], [translation_srt_file]).then(
        summarize, [uploaded_srt_file, generate_summarization_checkbox, summarization_prompt_text,
                    summarization_source_lang_dropdown,
                    summarization_target_lang_dropdown, replace_list_text,
                    delete_list_text], [summarization_txt_file]).then(generate_video, [output_video, uploaded_srt_file,
                                                                                       use_translation_checkbox,
                                                                                       translation_srt_file,
                                                                                       font_size_slider,
                                                                                       margin_v_slider,
                                                                                       merge_to_video_checkbox],
                                                                      [output_video])

if __name__ == "__main__":
    app.launch()
