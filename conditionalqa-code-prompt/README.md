# ConditionalQA Code Prompt

This repository includes the code and prompts used for ConditionalQA dataset in 2024 arXiv paper "Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs." The link to the original repo is [here](https://github.com/UKPLab/arxiv2024-conditional-reasoning-llms).

## Project structure
### Scripts
* `conditionalqa_code_prompt.ipynb` -- This notebook runs `code prompts` on `ConditionalQA`
* `conditionalqa_text_prompt.ipynb` -- This notebook runs `text prompts` on `ConditionalQA`
  
### Backend
* `src` -- This folder contain the classes that define `text prompts` and `code prompts` for `ConditionalQA`.
* `data` -- This folder contains the training, dev, and ICL demonstrations used in the experiments (including ablations).
* `outputs` -- This folder contains all the prompts (inputs and outputs). It also includes the evaluation results of each prompt. 

## Requirements
* openai
* langchain
* scikit-learn
* vllm

You also need an Azure OpenAI or OpenAI API account and put your key in the notebook to run them.

## Installation
```
conda create --name code_prompting python=3.9
conda activate code_prompting
pip install -r requirements.txt
```

## Running the experiments 
To reproduce results for Llama3 model, you just need to run below command on the terminal first:
```
python -m vllm.entrypoints.openai.api_server --model TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ --dtype float16 --api-key token-abc123
```
Then run these notebooks:
* `conditionalqa_code_prompt.ipynb`
* `conditionalqa_text_prompt.ipynb`

To reproduce results for OpenAI model, simply add the OpenAI API keys to the notebooks and run the notebook. 

## Code Prompt Experiment results on Dev Set:

| **Model**                	| **no. of ICL examples** 	| **F1 score** 	|
|--------------------------	|-------------------------	|--------------	|
|          GPT 3.5         	|       _4 (paper)_       	|   **_57.64_** |
|          Mistral         	|       _4 (paper)_       	|    _28.26_   	|
|          Mistral         	|            4            	|     2.97     	|
|          Mistral         	|   4 (updated prompts)   	|     2.89     	|
|          Mistral         	|            1            	|     4.37     	|
|          Mixtral         	|       _4 (paper)_        	|    _40.88_   	|
|          Mixtral         	|            1            	|     15.79    	|
|   Llama-3-8B- Instrcut   	|            4            	|     35.86    	|
| Llama3-70B-Instruct-GPTQ 	|            4            	|     45.61    	|

## Text Prompt Experiment results on Dev Set:

| **Model**                	| **no. of ICL examples** 	| **F1 score** 	|
|--------------------------	|-------------------------	|--------------	|
|          GPT 3.5         	|       _6 (paper)_       	|    _56.54_   	|
|          Mistral         	|       _6 (paper)_       	|    _28.84_   	|
|          Mistral         	|            1            	|     7.52     	|
|          Mistral         	|            6	            |     12.15     |
|          Mistral         	|   6 (updated prompts)     |     8.71     	|
|          Mixtral         	|       _6 (paper)_        	|    _46.60_   	|
|          Mixtral         	|            1            	|     19.44    	|
|          Mixtral         	|   1 (updated prompts)     |     16.99    	|
|   Llama-3-8B- Instrcut   	|            6            	|     45.92    	|
| Llama3-70B-Instruct-GPTQ 	|            6            	|     **62.00** |