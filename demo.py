import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = "winninghealth/WiNGPT2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().to("cuda")
model = model.eval()
user_role, assistant_role, stop = 'User: ', 'Assistant: ', '<|endoftext|>\n '


def generate(text):
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, repetition_penalty=1.1, max_new_tokens=1024)
    output = tokenizer.decode(outputs[0])
    return output.split(assistant_role)[-1].strip(stop.strip())


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def bot(message, chat_history):
        formatted_chat_history = stop.join([f'{user_role}{i[0]}{stop}{assistant_role}{i[1]}' for i in chat_history])
        formatted_message = f'{user_role}{message}{stop}{assistant_role}'
        inputs = f'{formatted_chat_history}{stop}{formatted_message}' if formatted_chat_history else formatted_message
        response = generate(inputs)
        chat_history.append((message, response))
        return "", chat_history

    msg.submit(bot, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
