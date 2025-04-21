import json
import time
import os
from tqdm import tqdm
from openai import OpenAI
from env import *

# 初始化 ModelScope 客户端
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key=MODELSCOPE_TOKEN,
)

# 通用翻译函数，接受字段类型以区分 prompt 或 text
def translate_with_modelscope(text, field='prompt', retry=3):
    if field == 'text':
        prompt = f"请将以下关于孤独症儿童绘本的英文故事翻译成中文，不要输出无关内容，保持段落结构：\n{text}"
        role_description = "你是一位专业的绘本翻译助手。"
    else:
        prompt = f"请将以下英文翻译成中文，只进行翻译：\n{text}"
        role_description = "你是一位专业的翻译助手，只负责准确翻译英文，不生成额外内容。"

    for _ in range(retry):
        try:
            response = client.chat.completions.create(
                model='deepseek-ai/DeepSeek-V3',
                messages=[
                    {'role': 'system', 'content': role_description},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("翻译失败，重试中...", e)
            time.sleep(1)
    return text  # 如果失败，返回原文

# 主函数：逐条处理，并写入 jsonl 文件
def translate_json_file_line_by_line(input_file, output_file, start_index=0):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否存在旧输出文件，如果不从0开始，不删文件
    if os.path.exists(output_file):
        os.remove(output_file)

    # 从指定索引开始处理
    for i in tqdm(range(start_index, len(data)), desc="翻译进度", unit="条", initial=start_index, total=len(data)):
        item = data[i]

        # 翻译 text 字段
        if 'text' in item:
            item['text'] = translate_with_modelscope(item['text'], field='text')

        # 翻译 prompt 字段
        if 'prompt' in item:
            item['prompt'] = translate_with_modelscope(item['prompt'], field='prompt')

        # 每条翻译完就追加写入一行
        with open(output_file, 'a', encoding='utf-8') as f_out:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 所有内容已翻译完成并保存到 {output_file}")

input_json = 'Children-Stories-0-Final.json'     # 输入原始 JSON 文件
output_jsonl = 'translated_output_60018.jsonl'         # 每条记录一行的输出文件
start_from_index = 60018                        # 从第几条开始翻译

translate_json_file_line_by_line(input_json, output_jsonl, start_index=start_from_index)