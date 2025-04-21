from openai import OpenAI
from env import *  # 确保 api_key 存在
import gradio as gr
import json
from fpdf import FPDF
import os
import tempfile
import requests
from PIL import Image
from io import BytesIO
import base64
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from modelscope.outputs import OutputKeys


# 模型客户端初始化
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key=MODELSCOPE_TOKEN,
)

# ============== 工具函数 =================
def image_to_base64(image_path):
    with open(image_path, "rb") as img_f:
        encoded = base64.b64encode(img_f.read()).decode("utf-8")
        ext = image_path.split('.')[-1]
        return f"data:image/{ext};base64,{encoded}"

def split_story_to_paragraphs(story_text):
    return [p.strip() for p in story_text.split('\n') if p.strip()]

# ============== 图像生成 =================

IMAGE_GEN_URL = 'https://api-inference.modelscope.cn/v1/images/generations'
IMAGE_MODEL_ID = 'MusePublic/489_ckpt_FLUX_1'

def generate_image(prompt_text, save_path):
    headers = {
        'Authorization': f'Bearer {MODELSCOPE_TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': IMAGE_MODEL_ID,
        'prompt': prompt_text
    }
    response = requests.post(IMAGE_GEN_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"图片生成失败: {response.text}")
    image_url = response.json()['images'][0]['url']
    image = Image.open(BytesIO(requests.get(image_url).content))
    image.save(save_path)
    return save_path

def summarize_and_translate(paragraphs):
    english_prompts = []
    for i, para in enumerate(paragraphs):
        prompt = f"""
你是一个专业的孤独症儿童绘本图像提示语设计助手，请对以下中文故事段落进行简要概括，并翻译成一句英文图像提示。要求简练清晰，写明卡通风格，不要解释或加标点，只输出英文。

段落：
{para}

输出英文描述："""

        response = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-V3',
            messages=[
                {'role': 'system', 'content': '你是一个专业绘本图像提示词生成器，只输出英文图像描述。'},
                {'role': 'user', 'content': prompt}
            ]
        )
        english = response.choices[0].message.content.strip()
        english_prompts.append(english)
        print(f"第{i+1}段内容翻译并概括:{english}")
    return english_prompts

# ============== 构建绘本请求 Prompt =================

def build_prompt(age, trait, interest, sensitivity, cognitive, edu_goal, words, max_characters,para_count):
    return f"""
你是孤独症儿童绘本创作专家，请根据以下儿童信息和教育目标，生成一篇适合TA阅读的故事，并以JSON格式输出。

【儿童基本信息】
- 年龄：{age}岁
- 性格特点：{trait}
- 兴趣主题：{interest}
- 感官敏感点：{sensitivity}
- 认知水平：{cognitive}

【故事生成要求
- 教育目标：{edu_goal}
- 期望段落数量：{para_count}段
- 角色数量不超过{max_characters}个
- 字数控制在{words}字左右，语言温和，避免感官刺激
- 每段保持逻辑清晰，适当使用重复句式
- 输出为以下 JSON 格式（不加 Markdown、注释）：

请严格输出如下 JSON 格式：

{{
  "title": "绘本标题",
  "story": "绘本正文内容，约{words}字。请合理分段。",
  "goal": "教育目标（标签+说明）",
  "parent_tip": "给家长的引导建议",
  "qa": [
    {{"question": "问题1", "answer": "回答1"}},
    {{"question": "问题2", "answer": "回答2"}}
  ]
}}

请严格只输出 JSON，不加任何注释、标点或 Markdown。
"""

def generate_story(age, trait, interest, sensitivity, cognitive, edu_goal, max_words, max_characters,para_count):
    prompt = build_prompt(age, trait, interest, sensitivity, cognitive, edu_goal, max_words, max_characters,para_count)
    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-V3',
        messages=[
            {'role': 'system', 'content': '你是一个孤独症儿童绘本创作专家，输出结构化JSON内容。'},
            {'role': 'user', 'content': prompt}
        ],
    )
    print(f"Raw message:\n{response}\n")
    content = response.choices[0].message.content.strip()
    print(f"LLM Answer:\n {content}\n")
    try:
        result = json.loads(content)
        print(f"Json parser:\n {result}\n")
        return result
    except Exception as e:
        return {"error": f"解析失败，请检查模型输出格式。错误：{e}", "raw_output": content}

# ============== PDF 创建函数 =================

def create_pdf(data, mode="full", image_paths=None):
    pdf = FPDF()
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError("缺少字体 SimHei.ttf")

    pdf.add_page()
    pdf.add_font("SimHei", "", font_path)
    pdf.set_font("SimHei", size=14)
    pdf.multi_cell(0, 10, f"绘本标题：{data['title']}\n", align="L")
    pdf.set_font("SimHei", size=12)

    paragraphs = split_story_to_paragraphs(data['story'])

    for idx, para in enumerate(paragraphs):
        pdf.multi_cell(0, 10, para)
        pdf.ln(2)

        # 图文版 或 full 图文版时插图
        if mode in ["illustrated", "full", "question"] and image_paths and idx < len(image_paths):
            img = Image.open(image_paths[idx])
            img.thumbnail((200, 200))
            thumb_path = f"thumb_{idx}.jpg"
            img.save(thumb_path)  # 👈 确保保存缩略图
            x_center = (210 - 100) / 2
            pdf.image(thumb_path, x=x_center, w=100)
            pdf.ln(5)

    if mode == "full":
        pdf.multi_cell(0, 10, f"\n教育目标：{data['goal']}\n")
        pdf.multi_cell(0, 10, f"家长引导建议：{data['parent_tip']}\n")
        for qa in data["qa"]:
            pdf.multi_cell(0, 10, f"问：{qa['question']}\n答：{qa['answer']}\n")

    elif mode == "question":
        for qa in data["qa"]:
            pdf.multi_cell(0, 10, f"问题：{qa['question']}\n答：__________________________\n\n")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ============== 分步执行函数 =================

def generate_story_only(age, trait, interest, sensitivity, cognitive, max_words, max_characters, para_count, edu_goal):
    data = generate_story(age, trait, interest, sensitivity, cognitive, edu_goal, max_words, max_characters, para_count)
    if "error" in data:
        return data["error"], "", "", "", "", data

    qa_str = "\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in data["qa"]])
    paragraphs = split_story_to_paragraphs(data["story"])
    story_html = ""
    for i, p in enumerate(paragraphs):
        story_html += f'<div id="para{i}"><p>{p}</p></div>'

    return data['title'], story_html, data['goal'], data['parent_tip'], qa_str, data

def async_generate_images_and_pdfs(data, progress=gr.Progress()):
    paragraphs = split_story_to_paragraphs(data['story'])
    english_prompts = summarize_and_translate(paragraphs)

    story_html_parts = []
    image_paths = []

    # 初始纯文字
    for i, para in enumerate(paragraphs):
        story_html_parts.append(f'<div id="para{i}"><p>{para}</p></div>')

    yield "\n".join(story_html_parts), "开始生成插图...", None, None, None, None

    for idx, prompt in enumerate(progress.tqdm(english_prompts, desc="生成插图中...")):
        img_path = f"illustration_{idx + 1}.jpg"
        try:
            generate_image(prompt, img_path)
            img = Image.open(img_path)
            img.thumbnail((500, 500))
            img.save(img_path)
            image_paths.append(img_path)

            # 插入图像 HTML 到对应段落
            # story_html_parts[idx] = story_html_parts[idx].replace(
            #     "</div>", f'<img src="file/{img_path}" style="max-width:100%; margin-top:5px;"></div>'
            # )
            base64_src = image_to_base64(img_path)
            story_html_parts[idx] = story_html_parts[idx].replace(
                "</div>",
                f'<img src="{base64_src}" style="max-width:100%; margin-top:5px;"></div>'
            )

        except Exception as e:
            story_html_parts[idx] += f"<p style='color:red;'>插图失败：{e}</p>"

        yield "\n".join(story_html_parts), f"✅ 已完成 {idx+1}/{len(english_prompts)} 张插图", None, None, None, None

    # 四种 PDF 分别生成
    pdf_story = create_pdf(data, mode="story")  # 仅正文，无图
    pdf_full = create_pdf(data, mode="full", image_paths=image_paths)  # 全部文字信息，无图
    pdf_qa = create_pdf(data, mode="question", image_paths=image_paths)  # 问答练习，无图
    pdf_illustrated = create_pdf(data, mode="illustrated", image_paths=image_paths)  # 图文版，只正文+图

    audio_path = synthesize_audio_from_story(data['story'])
    yield "\n".join(story_html_parts), "✅ 所有插图已完成", pdf_story, pdf_full, pdf_qa, pdf_illustrated, audio_path


# ============== 语音合成 =================
def synthesize_audio_from_story(story_text):
    from datetime import datetime

    # 初始化语音模型
    tts_pipeline = pipeline(
        task=Tasks.text_to_speech,
        model='damo/speech_sambert-hifigan_tts_zh-cn_16k'
    )

    # 执行语音合成
    output = tts_pipeline(input=story_text, voice='zhitian_emo')
    wav_data = output[OutputKeys.OUTPUT_WAV]

    # 保存音频文件
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(tmp_path, 'wb') as f:
        f.write(wav_data)

    return tmp_path

# ============== Gradio UI =================

with gr.Blocks(title="图文绘本生成系统") as demo:
    gr.Markdown("## 🎨 孤独症儿童图文绘本生成器")

    with gr.Row():
        with gr.Column():
            age = gr.Slider(2, 18, label="儿童年龄", value=2)
            max_words = gr.Slider(50, 1500, label="最大字数", value=50)
            para_count = gr.Slider(1, 10, step=1, label="段落数量", value=1)
            max_characters = gr.Slider(1, 5, label="最多角色", value=1)
            trait = gr.Textbox(label="性格特点", value="无")
            interest = gr.Textbox(label="兴趣主题", value="无")
            sensitivity = gr.Textbox(label="感官敏感点", value="无")
            cognitive = gr.Textbox(label="认知水平", value="无")
            edu_goal = gr.Textbox(label="教育目标", value="无")
            generate_btn = gr.Button("生成图文绘本")

        with gr.Column():
            title_box = gr.Textbox(label="绘本标题")
            story_display = gr.HTML(label="绘本图文内容")
            goal_box = gr.Textbox(label="教育目标")
            tip_box = gr.Textbox(label="家长引导建议")
            qa_box = gr.Textbox(label="理解问答", lines=4)
            image_progress = gr.Textbox(label="生成进度", interactive=False)
            # audio_output = gr.Audio(label="🎧 故事朗读音频", type='filepath')

    with gr.Row():
        pdf_story = gr.File(label="📘 正文PDF")
        pdf_full = gr.File(label="📕 完整PDF")
        pdf_qa = gr.File(label="🧠 练习PDF")
        pdf_illustrated = gr.File(label="🎨 图文PDF")

    intermediate_state = gr.State()

    generate_btn.click(
        fn=generate_story_only,
        inputs=[age, trait, interest, sensitivity, cognitive, max_words, max_characters, para_count, edu_goal],
        outputs=[title_box, story_display, goal_box, tip_box, qa_box, intermediate_state]
    )

    intermediate_state.change(
        fn=async_generate_images_and_pdfs,
        inputs=[intermediate_state],
        outputs=[story_display, image_progress, pdf_story, pdf_full, pdf_qa, pdf_illustrated],
        show_progress=True
    )

    # intermediate_state.change(
    #     fn=async_generate_images_and_pdfs,
    #     inputs=[intermediate_state],
    #     outputs=[
    #         story_display, image_progress,
    #         pdf_story, pdf_full, pdf_qa, pdf_illustrated,
    #         audio_output
    #     ],
    #     show_progress=True
    # )

    demo.launch()