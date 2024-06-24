# Video Subtitle Generator

This project automatically generates and adds subtitles to a video file using Whisper for speech recognition and
gpt-4 for translation.

## Setup

1. Create a conda environment:
   ```
   conda activate video-subtitle-generator
   ```

2. Activate the environment:
   ```
   conda activate video-subtitle-generator
   ```

3. Install required packages:
   ```
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
   # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
   ```

4. Test Cuda
   ```
   python -c "import torch;print(torch.cuda.is_available());"
   ```

## Usage

Run the script with:

```
python cli.py input_video.mp4 ./output/output_video.mp4 [options]
```

Use the `--use-trim` flag to use 30 seconds instead of full video:
Use the `--use-translation` flag to use translation instead of original language:
- `--source-lang`: Source language (default: Chinese)
- `--target-lang`: Target language (default: English)
- `--font-size`: Font size for subtitles (default: 18)
- `--margin-v`: Vertical margin for subtitles (default: 2)

```
python cli.py input_video.mp4 ./output/output_video.mp4 --use-trim --use-translation --source-lang Chinese --target-lang English --font-size 20 --margin-v 4
```

If you want to generate both languages subtitles:

```
python cli.py input_video.mp4 ./output/original_output.mp4 --font-size 18 --margin-v 2
python cli.py ./output/original_output.mp4 ./output/final_output.mp4 --use-translation --source-lang Japanese --target-lang English --font-size 18 --margin-v 22
```

## Others

Install EPEL（Extra Packages for Enterprise Linux）

```
sudo dnf install -y oracle-epel-release-el8
```

Enable RPM Fusion

```
sudo dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm
sudo dnf install -y https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-8.noarch.rpm
```

Install dnf plugins

```
sudo dnf install -y dnf-plugins-core
```

Enable ol8_codeready_builder

```
sudo dnf config-manager --set-enabled ol8_codeready_builder
```

Install ffmpeg

```
sudo dnf install -y ffmpeg ffmpeg-devel
```

Install Google Noto Sans font

```
sudo dnf install -y google-noto-sans-cjk-ttc-fonts
```


Flush font cache

```
fc-cache -f -v
```

Verify font

```
fc-list :lang=zh
```

Use ffmpeg to set font

```
ffmpeg -i input_video.mp4 -vf subtitles=subtitle.srt:force_style='FontName=SimSun' -c:v libx264 -c:a copy output_video.mp4
```

