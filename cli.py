import os
import argparse

from dotenv import load_dotenv, find_dotenv

from utils import whisper_utils, translation_utils, ffmpeg_utils

# read local .env file
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]


def main(input_video, output_video, use_trim=False, use_translation=False, source_lang='zh', target_lang='en', font_size=18, margin_v=2):
    # Step 1: Prepare video
    trimmed_video = input_video
    if use_trim:
        trimmed_video = "output/trimmed.mp4"
        ffmpeg_utils.trim_video(input_video, trimmed_video, duration=30)

    # Step 2: Generate subtitles using Whisper
    srt_file = "output/output.srt"
    whisper_utils.generate_subtitles(trimmed_video, srt_file, False)

    if use_translation:
        # Step 3 & 4: Translate subtitles using gpt-4 via LangChain
        translated_srt = "output/translated_ast.srt"
        translation_utils.translate_subtitles(srt_file, translated_srt, source_lang, target_lang)
    else:
        translated_srt = srt_file

    # Step 5: Add subtitles to video using FFmpeg
    ffmpeg_utils.add_subtitles_to_video(trimmed_video, translated_srt, output_video, font_size, margin_v)

    print(f"Video with subtitles generated: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subtitled video")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("output_video", help="Path to output video file")
    parser.add_argument("--use-trim", action="store_true", help="Use 30 seconds trimmed video instead of full video")
    parser.add_argument("--use-translation", action="store_true",
                        help="Use translation instead of original language")
    parser.add_argument("--source-lang", default="zh", help="Source language (default: zh)")
    parser.add_argument("--target-lang", default="en", help="Target language (default: en)")
    parser.add_argument("--font-size", type=int, default=18, help="Font size for subtitles (default: 18)")
    parser.add_argument("--margin-v", type=int, default=2, help="Vertical margin for subtitles (default: 2)")
    args = parser.parse_args()

    main(args.input_video, args.output_video, args.use_trim, args.use_translation, args.source_lang, args.target_lang, args.font_size, args.margin_v)
