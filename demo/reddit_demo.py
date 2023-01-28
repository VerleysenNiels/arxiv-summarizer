import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration

if __name__ == "__main__":
    # Load finetuned model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("NielsV/distilbart-cnn-6-6-reddit", cache_dir="cache")
    model = BartForConditionalGeneration.from_pretrained("NielsV/distilbart-cnn-6-6-reddit", cache_dir="cache")

    # Function to write a TLDR
    def generate_tldr(input_txt):
        inputs = tokenizer(input_txt, max_length=1024, return_tensors="pt")
        summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=60)
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    demo = gr.Interface(
        fn=generate_tldr, 
        inputs=gr.Textbox(lines=5, placeholder="...", label="Post to summarize..."), 
        outputs=gr.Textbox(lines=2, label="Too long, didn't read:")
        )

    demo.launch()