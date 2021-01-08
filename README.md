# Speaking AI Chatbot
## 中文普通话语音聊天机器人 

## 项目描述
- 本项目旨在实现一个可以实时生成语音进行对话的闲聊机器人。生成的语音为普通话女声，生成音色自然并有韵律停顿。对话机器人则可以一定程度上根据历史对话内容实现下一句。
- 本项目对话机器人部分代码来自[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)项目。使用GPT2模型对中文闲聊语料进行训练，使用 HuggingFace的[transformers](https://github.com/huggingface/transformers)实现GPT2模型的编写与训练。详细实现请参考该项目。本项目暂时只对其中的生成代码稍作改动来进行语音生成部分的融合。
- 语音合成（TTS）部分则使用了中文[tacotron2](https://github.com/JasonWei512/Tacotron-2-Chinese)的代码,并仿造[syang1993/gst-tactron](https://github.com/syang1993/gst-tacotron.git)对tacotron自带的注意力进行了优化，实现了gmm attention大幅减少了训练收敛时间，并提升了对较长文本的合成效果。另对语音合成代码进行了部分修改来读取聊天模型生成的文字并实时合成为语音。
- 使用标贝开源中文普通话女声语音[数据集](https://www.data-baker.com/open_source.html)进行训练，训练了十几万歩，vocoder部分则使用的是griffin-lim。接下来准备训练Wavernn来作为vocoder。
-生成的wav格式语音会被存到本地并自动播放

## 运行环境
python3.6、 transformers==1.12、pytorch==1.3.1，其他环境要求请见requirements.txt

## 项目结构
- models:存放tacotron主要模型框架代码
- modules:存放编码器，解码器及注意力机制等模块的代码
- synthesizer: 存放合成器，拼音文字转换代码
- text： 存放拼音编码转换代码
- dialogue_model:存放对话生成的模型
- mmi_model:存放MMI模型(maximum mutual information scoring function)，用于预测P(Source|response)
- sample:存放人机闲聊生成的历史聊天记录
- vocabulary:存放GPT2模型的字典
- egs:
  -example:
    -preprocess.sh: 用于处理语音数据
    -synthesis.sh： 用于单独合成语音（须单独分割一部分label不建议使用）
    -train.sh: 用于训练tacotron声学模型
- train_gpt.py:训练对话机器人代码
- interact.py:人机交互代码（无语音合成功能)
- chatbot_entry.py: 直接使用这个代码来运行对话机器人语音生成一体化程序


## 模型参数
- log_dir: 用于存放tacotron模型训练好的权重（等待上传）
- dialogue_model: 对话模型训练好的权重（等待上传）
- mmi_model：对话模型训练好的权重（等待上传）


## 使用说明
clone该项目到本地后将训练好的模型权重（待上传）下载到本地并放入对应目录下，运行python -W ignore chatbot_entry.py即可使用对话机器人语音生成一体化程序。只要输入中文文字即可听到语音合成的回答。


## TODO
- 上传训练好的模型权重
- 整理训练代码
- 添加注释并规范代码

## Future Work
- 训练多种vocoder以供挑选
- 改进对话生成模型
- 增加意图识别模块
- 增加关系识别模块

## Reference




