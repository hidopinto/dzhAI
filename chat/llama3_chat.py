import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Llama3:
    """
    In order to run this class you need a local Llama3 model(we used 8B instruct) to provide its path.
    This model requires ~31 GB of GPU memory.
    """
    def __init__(self, order):
        self.access_token = "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"
        self.order = order
        self.tokenizer = AutoTokenizer.from_pretrained("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/dzhai",
                                                  token=self.access_token, device_map=0)
        self.model = AutoModelForCausalLM.from_pretrained("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/dzhai",
                                                     token=self.access_token, device_map=0)

        self.pipeline = transformers.pipeline("text-generation", model=self.model,
                                         tokenizer=self.tokenizer,
                                         model_kwargs={"torch_dtype": torch.bfloat16}, device_map=0)

        self.terminators = [self.pipeline.tokenizer.eos_token_id,
                            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.message_history = [{"role": "system", "content": """"You are LLAMA3 8B, 
                                                               a highly capable language model
                                                                designed to serve as a hospital 
                                                               emergency room manager. Your primary role is to 
                                                               communicate with patients about their wait 
                                                               times and position in line. Your responses
                                                                are always honest, kind, compassionate, caring,
                                                                sweet, and patient. You always respect patient confidentiality, and th
                                                                means you especially never give up any information on
                                                                one patient to another patient, except how 
                                                                many patients are waiting in line before them.
                                                                Your goal is to provide clear  
                                                               information while offering reassurance and empathy  
                                                               to patients who may be feeling anxious or unwell. 
                                                               Respond to the patient with kindness and compassion,
                                                                aand provide honest and clear information to patients 
                                                               about their wait time. The patients are waiting in the
                                                                following order (from first to last): """
                                                               + str(order)},
                                 {"role": "user", "content": "Who are you?"}]

    def get_response(self, query, identity, max_tokens=400, temperature=0.6, top_p=0.9):
        query = identity + " asks " + query
        user_prompt = self.message_history + [{"role": "user", "content": query}]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def ask(self, identity, question):
        response, conversation = self.get_response(question, identity)
        self.message_history += conversation
        print(f"Assistant: {response}")


if __name__ == "__main__":
    bot = Llama3(order=["Lucy", "Sally", "Hido", "Tomer", "Hido", "Stephanie", "Barbara"])
    bot.ask("Tomer", "Where am I in the queue?")
