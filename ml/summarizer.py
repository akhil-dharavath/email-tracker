from transformers import T5ForConditionalGeneration, T5Tokenizer

class EmailSummarizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def summarize(self, text, max_length=50):
        # T5 requires "summarize: " prefix
        input_text = "summarize: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        summary_ids = self.model.generate(
            input_ids, 
            max_length=max_length, 
            min_length=10, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
