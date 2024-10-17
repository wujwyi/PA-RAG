import logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import openai
import tiktoken
import argparse
import yaml
import os
import numpy as np
import json
from utils import *
from tqdm import tqdm
from vllm import LLM, SamplingParams
import random
import copy
        
remain_num=[1,1,1,2,2]
def main():
    logging.getLogger("httpx").disabled = True
    # logging.getLogger("openai").disabled = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")
    parser.add_argument("--span_test", nargs='*', type=int, default=None)

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="Whether to remove the documents in the demos")
    parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--start_key_number", type=int, default=0)

    # Decoding
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_ctx_length", type=int, default=4096, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")
    parser.add_argument("--docs_select", type=str, default=None)



    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    if "gpt-3.5" in args.model or '16k' in args.model:
        # ChatGPT has a longer max length
        args.max_ctx_length = 16384
    elif "Llama-2-7b-chat-hf" in args.model:
        args.max_ctx_length = 4096
    
    logger.info(f"Set the model max length to {args.max_ctx_length} (if not correct, check the code)")

    
    # Load the model or setup the API
    llm = LLM(model=args.model, tensor_parallel_size=1, tokenizer_mode='slow', trust_remote_code=True)
    tokenizer=llm.get_tokenizer()
    chat_template = open(args.model+'/chat.jinja').read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    tokenizer.chat_template = chat_template
    
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file))

    # Generate the demonstration part
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        ndoc = args.ndoc
        if args.no_doc_in_demo:
            ndoc = 0
        elif args.fewer_doc_in_demo:
            assert args.ndoc_in_demo is not None
            ndoc = args.ndoc_in_demo
        head_prompt += make_demo(
            train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter
        )
        head_prompt += prompt_data["demo_sep"]

    if args.quick_test is not None:
        eval_data=eval_data[:args.quick_test]
    if args.span_test is not None:
        eval_data=eval_data[args.span_test[0]:args.span_test[1]]

    logger.info("Generating prompts...") 
    sens=[]
    length=[]
    for idx, eval_item in enumerate(tqdm(eval_data)):
        if not args.docs_select:
            doc_list = get_shorter_text(eval_item, eval_item["docs"], args.ndoc, args.use_shorter) if args.use_shorter is not None else eval_item["docs"][:args.ndoc]
        elif args.docs_select=="golden":
            doc_list = get_shorter_text(eval_item, eval_item["golden_docs"], args.ndoc, args.use_shorter) if args.use_shorter is not None else eval_item["golden_docs"][:args.ndoc]
        elif args.docs_select=="golden+noise":
            golden_docs=eval_item["golden_docs"]
            noise_docs=eval_item["irrelevant_docs"]+eval_item["relevant_docs"]
            golden_docs_with_index = [('golden_docs', index, value) for index, value in enumerate(golden_docs)]
            noise_docs_with_index = [('noise_docs', index, value) for index, value in enumerate(noise_docs)]
            
            for element in noise_docs_with_index:
                index = random.randint(0, len(golden_docs_with_index))
                golden_docs_with_index.insert(index, element)

            doc_list = [item[2] for item in golden_docs_with_index]
            eval_item['original_index'] = [[item[0],item[1]] for item in golden_docs_with_index]
        elif args.docs_select=="subgolden":
            golden_docs=eval_item["golden_docs"]
            indices = sorted(random.sample(range(len(golden_docs)), remain_num[len(golden_docs)-1]))
            doc_list=[golden_docs[i] for i in indices]
            eval_item["original_index"] = indices
        elif args.docs_select=="subgolden+noise":
            golden_docs=eval_item["golden_docs"]
            noise_docs=eval_item["irrelevant_docs"]+eval_item["relevant_docs"]
            indices = sorted(random.sample(range(len(golden_docs)), remain_num[len(golden_docs)-1]))
            subgolden=[golden_docs[i] for i in indices]

            subgolden_docs_with_index = [('golden_docs', indices[index], value) for index, value in enumerate(subgolden)]
            noise_docs_with_index = [('noise_docs', index, value) for index, value in enumerate(noise_docs)]

            for element in noise_docs_with_index:
                index = random.randint(0, len(subgolden_docs_with_index))
                subgolden_docs_with_index.insert(index, element)
            
            doc_list = [item[2] for item in subgolden_docs_with_index]
            eval_item['original_index'] = [[item[0],item[1]] for item in subgolden_docs_with_index]
        elif args.docs_select=="no":
            doc_list=[]
        else:
            import pdb; pdb.set_trace()
            a=1

        eval_item['select_docs'] = doc_list

        prompt = head_prompt + make_demo(
            eval_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter, 
            test=True, doc_key = 'select_docs'
        )
        chat = [
            {'role': 'system', 'content': "You are a helpful assistant that answers the following questions with proper citations."},
            {'role': 'user', 'content': prompt}
        ]
        eval_item['prompt'] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)[3:]
        # print(eval_item['prompt'])
        sens.append(eval_item['prompt'])
        # length.append(len(tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)))

        
    logger.info("Done.")
    # print(max(length))
    # print(sum(length)/len(length))

    
    stop = ["\n","<0x0A>"]
    stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
    print(stop_token_ids)
    sampling_param = SamplingParams(
        max_tokens=args.max_new_tokens, 
        top_p=args.top_p, 
        temperature=args.temperature,
        stop_token_ids=stop_token_ids
    )

    outputs = llm.generate(sens, sampling_params=sampling_param)
    # import pdb; pdb.set_trace()
    output_data=[]
    for idx, item in enumerate(tqdm(eval_data)):      
        item['output'] = outputs[idx].outputs[0].text
        output_data.append(item)
        # with open(args.output_file,"w") as fout:
        #     json.dump(output_data, fout, indent=4)
    
    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    
    with open(args.output_file,"w") as fout:
        json.dump(eval_data, fout, indent=4)



if __name__ == "__main__":
    main()








