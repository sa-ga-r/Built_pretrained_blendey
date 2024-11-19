import torch
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import streamlit as st
import time

class Blendey():
    def __init__(self):
        self.model_name = 'facebook/blenderbot-400M-distill'
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)
        self.qtz_model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)
        self.qtz_model.eval()
        
    def generate_responce(self, user_input):
        UTTERANCE = user_input
        t_in = self.tokenizer([UTTERANCE], return_tensors = 'pt')
        gen_resp = self.qtz_model.generate(**t_in, max_length=30, top_k=50, top_p=0.95, do_sample=True)
        resp_idx = self.tokenizer.batch_decode(gen_resp, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        return resp_idx[0]
        
bot = Blendey()

def main():
    st.title("Blendey")
    st.caption("A general purpose chatbot")
    if "message" not in st.session_state:
        st.session_state.message = []
    for message in st.session_state.message:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if user_input := st.chat_input('Say something...'):
        with st.chat_message("User"):
            st.markdown(user_input)
            st.session_state.message.append({"role":"User", "content":user_input})
        start_time = time.time()
        bot_resp = bot.generate_responce(user_input)
        responce_time = time.time() - start_time
        
        with st.chat_message("Blendey"):
            st.markdown(bot_resp)
            st.write(f"Responce Time:{responce_time:.2f}Sec")
            st.session_state.message.append({"role":"Blendey", "content":bot_resp})

if __name__ == '__main__':
    main()
