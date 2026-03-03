import soundfile as sf
from transformers import Qwen3OmniProcessor
from transformers import Qwen3OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info

MODEL_PATH = "/models/Qwen3-Omni-4B-Instruct-multilingual"

model = Qwen3OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen3OmniProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
            {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
            {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
        ],
    },
]


# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, 
                                 speaker="f245", 
                                 thinker_return_dict_in_generate=True,
                                 use_audio_in_video=USE_AUDIO_IN_VIDEO,
                                 talker_max_new_tokens = 128)

text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
print(text)
if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )