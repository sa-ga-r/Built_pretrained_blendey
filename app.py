import torch
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import streamlit as st 

class Blendey():
    def __init__(self):
        model_name = 'facebook/blenderbot-400M-distill'
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def generate_responce(self, user_input):
        UTTERANCE = user_input
        t_in = self.tokenizer([UTTERANCE], return_tensors = 'pt')
        gen_resp = self.model.generate(**t_in)
        resp_idx = self.tokenizer.batch_decode(gen_resp, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        return resp_idx[0]

def main():
    bot = Blendey()
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
        bot_resp = bot.generate_responce(user_input)
        with st.chat_message("Blendey"):
            st.markdown(bot_resp)
            st.session_state.message.append({"role":"Blendey", "content":bot_resp})           


if __name__ == '__main__':
    main()
