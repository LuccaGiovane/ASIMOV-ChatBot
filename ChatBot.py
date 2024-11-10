import os
import gradio as gr
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# Colocar a Key do chatGPT aqui
os.environ["OPENAI_API_KEY"] = "SUA_KEY_AQUI"


llm = OpenAI(temperature=0.7)

# Manter o contexto da conversa
memory = ConversationBufferMemory()


conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

def chat(user_input, history=[]):
    """ Função para o input do usuario """

    response = conversation.predict(input=user_input)
    history.append((user_input, response))

    return history, history


# Interface do Gradio
iface = gr.Interface(
    fn=chat,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    title="Chatbot com LangChain e Gradio",
)


iface.launch()