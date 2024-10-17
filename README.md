# PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization

# Dataset
## Training data
We provide the training data for PA-RAG, available at  
https://drive.google.com/file/d/1agP7fi1iX-3qFK7XFBvRu6rC5X_-M8Iy/view?usp=drive_link

Include
 * `sft_data.json`: 58.9k instruction fine-tuning data
 * `dpo_data_ri.json`: 11.8k response informative preference data
 * `dpo_data_rr.json`: 13.4k response robustness preference data
 * `dpo_data_cq.json`: 22.5k citation quality preference data

The question used for constracting our training data are sourced from  ASQA, WebQuestions, and Natural Questions. Detailed statistics are as follows:

||IFT|RI|RR|CQ|
|:--|:--:|:--:|:--:|:--:|
|ASQA|1,714|1,046|962|631|
|WebQ|1,681|326|357|653|
|NQ|55,463|10,416|12,080|21,241|
|Sum|58,858|11,788|13,399|22,525|

## Evaluation data
The data for evaluation is available at  
https://drive.google.com/file/d/1vn5O_PtUnV3rOC7CAbSsZITG6NQ1EZtx/view?usp=drive_link.  
The qustions are sourced from the test split from ASQA, WebQustions, Natural Questions, and TriviaQA. The retrieved documents are retrieved by dense retriever [GTR](https://huggingface.co/sentence-transformers/gtr-t5-xxl) from Wikipedia dump from December 20, 2018.

# Training
We use the framework [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train our models. We selected three general LLMs as the base RAG generator: [Llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).  
We utilized full fine-tuning for all training stages and employed the same hyperparameter settings for all models.  
During the instruction fine-tuning phase, we set the batch size to 128, the learning rate to 2e-5, and trained for one epoch.  
In the preference optimization phase, we set the batch size to 64 and trained for one epoch for all stages. For the optimization stages of response informativeness and response robustness, the learning rate is 2e-6. In the citation quality optimization stage, the learning rate is 2e-7.



# Inference
Inference with zero-shot setting
```
CUDA_VISIBLE_DEVICES=0 python inference/inference_vllm.py \
    --model model_path \
    --prompt_file prompts/default.json \
    --eval_file data_path (e.g.data/asqa_dev.json) \
    --output_file output_path \
    --shot 0 \
    --ndoc 5 \
```



# Evaluation
Download the NLI model [TRUE](https://huggingface.co/google/t5_xxl_true_nli_mixture) before evaluate.

```
CUDA_VISIBLE_DEVICES=0 python inference/eval.py --f response_to_eval_path --no_rouge --citation
```