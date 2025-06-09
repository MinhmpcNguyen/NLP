import os
import asyncio

import torch 
from tool.internet_search import get_interest_search
from api_search import search_rrf

# Load model


llama_prompt = """
Question:
{relevant_passage}
{user_input}?
Answer:

"""
def get_relevant(results: dict):
    return " ".join(result["text"] for result in results)

def make_rag_prompt_vi(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = llama_prompt.format(user_input=query, relevant_passage=escaped)

    return prompt

def make_interet_prompt_vi(query, search):
    relevant_passage = " ".join(search)
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = llama_prompt.format(user_input=query, relevant_passage=escaped)

    return prompt

def extract_model_output(full_text: str) -> str:
    """
    Extract the assistant's response content from the full raw model output.

    Assumes the output text contains "Answer:" and the response follows it.

    Returns the text between "Answer:" and the next part or end of the text.
    """
    start_token = "Answer:\n\n"
    end_token = "\n"

    start_idx = full_text.find(start_token)
    if start_idx == -1:
        return ""  # "Answer:" part not found

    # move start index to after the "Answer:" token
    start_idx += len(start_token)

    # find the next newline after the "Answer:" content
    next_idx = full_text.find(end_token, start_idx)
    if next_idx == -1:
        # no further token found, take till end
        output = full_text[start_idx:].strip()
    else:
        output = full_text[start_idx:next_idx].strip()

    return output

def generate_text(input_rag, tokenizer, model, device, 
                  temperature=0.7, top_p=0.9, max_new_tokens=200):
    input_ids = tokenizer(input_rag, return_tensors="pt",truncation = True).input_ids.cuda()
    # with torch.no_grad():
    output_tokens = model.generate(input_ids = input_ids,
                            max_new_tokens = 100,
                            temperature = 0.1,
                            num_return_sequences=1)
    output_text = tokenizer.batch_decode(output_tokens.detach().cpu().numpy(),skip_special_tokens = False)[0]
    output_text = extract_model_output(output_text)
    return output_text


async def answer_query(query: str, tokenizer, llama_model, device, rag_threshold: float = 0.2, top_k: int = 5,):
    """
    Fixed version with better fallback logic:
    - Try RAG first
    - If no good RAG response, fallback to Internet Search
    """
    # Step 1: Try RAG
    global model
    rag_results = search_rrf(model, query)
    
    if rag_results and len(rag_results) > 0:
        relevant_passage = get_relevant(rag_results)
        print("Answering with RAG...")
        rag_prompt = make_rag_prompt_vi(query, relevant_passage)
        response = generate_text(rag_prompt, tokenizer, llama_model, device)

       # Robust fallback condition
        should_fallback = (
            not response or 
            response.strip() == "Tôi không biết." or
            "Tôi không biết" in response or
            len(response.strip()) < 10  # Very short responses
        )
        
        if should_fallback:
            print("Falling back to Internet Search...")
            try:
                search_results = await get_interest_search(query, crawl=False)
                if search_results:
                    internet_prompt = make_interet_prompt_vi(query, search_results)
                    internet_response = generate_text(internet_prompt, tokenizer, llama_model, device)
                    return internet_response
                else:
                    print("No internet search results found")
                    return "Xin lỗi, tôi không thể tìm thấy thông tin về câu hỏi này."
            except Exception as e:
                print(f"Error in internet search: {e}")
                return "Đã xảy ra lỗi khi tìm kiếm thông tin."
        else:
            # RAG response is good enough
            return response
    else:
        # No RAG results, fallback immediately
        print("No RAG results, falling back to Internet Search...")
        try:
            search_results = await get_interest_search(query, crawl=False)
            if search_results:
                internet_prompt = make_interet_prompt_vi(query, search_results)
                internet_response = generate_text(internet_prompt, tokenizer, llama_model, device)
                return internet_response
            else:
                print("No internet search results found")
                return "Xin lỗi, tôi không thể tìm thấy thông tin về câu hỏi này."
        except Exception as e:
            print(f"Error in internet search: {e}")
            return "Đã xảy ra lỗi khi tìm kiếm thông tin."


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import AutoPeftModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vinallama_model = "/kaggle/input/vinallama-7b-finetuned-25-percent/pytorch/default/1/checkpoint-2510"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = False,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(vinallama_model)
    
    llama2_model = AutoPeftModelForCausalLM.from_pretrained(
        vinallama_model,
        quantization_config = bnb_config,
        low_cpu_mem_usage = True,
        return_dict = True,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    async def main():
        query = "Các phương thức xét tuyển Bách Khoa 2025"
        response_text = await answer_query(query, tokenizer, llama2_model, device)
        print(response_text)

    asyncio.run(main())
