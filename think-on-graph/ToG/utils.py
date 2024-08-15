from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
# from rank_bm25 import BM25Okapi
# from sentence_transformers import util
# from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate #,  LLMChain
# from auto_gptq import exllama_set_max_input_length
import torch
from langchain_community.llms import VLLM

llm = None
def initialise_llama3(temperature, max_tokens):
    print("--initialising llama3")
    global llm
    llm = VLLM(
            model = "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",
            trust_remote_code=True,  
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_llama2(temperature, max_tokens):
    print("--initialising llama2")
    global llm
    llm = VLLM(
            model = "TheBloke/Llama-2-70B-Chat-GPTQ",
            trust_remote_code=True,  
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def initialise_mixtral(temperature, max_tokens):
    print("--initialising Mixtral")
    global llm
    llm = VLLM(
            model = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
            trust_remote_code=True,  
            dtype="float16",
            max_new_tokens=max_tokens,
            top_k=-1,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.1,
            return_full_text=False
        )

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"  # your local llama server port
        engine = openai.Model.list()["data"][0]["id"]
    else:
        openai.api_key = opeani_api_keys

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        try:
            response = openai.ChatCompletion.create(
                    model=engine,
                    messages = messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response["choices"][0]['message']['content']
            f = 1
        except:
            print("openai error, retry")
            time.sleep(2)
    return result

def run_llama3(prompt_input):

    template="""You are an AI assistant that helps people find information.
    {prompt_input}
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    result = llm_chain.invoke(prompt_input)
    
    return result

def run_llama2(prompt_input):

    template="""[INST] <<SYS>> You are an AI assistant that helps people find information.<</SYS>>
    {prompt_input}[/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm

    result = llm_chain.invoke(prompt_input)
    
    return result

def run_mixtral(prompt_input):

    template="""<s>[INST] You are an AI assistant that helps people find information.
    {prompt_input} [/INST]
    """
    prompt = PromptTemplate.from_template(template)

    llm_chain = prompt | llm
    # print(prompt_input)
    result = llm_chain.invoke(prompt_input)
    
    return result

def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        # print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, ans_from):
    dict = {"question":question, "ans_from": ans_from ,"results": answer, "reasoning_chains": cluster_chain_of_entities}
    # print("\n\n--FINAL RESULT:--\n", dict)
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    # print("!! Answer without KG !!!")
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    if (args.LLM_type).startswith("gpt"):
        response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    elif args.LLM_type == "llama3":
        response = run_llama3(prompt)
    elif args.LLM_type == "llama2":
        response = run_llama2(prompt)
    elif args.LLM_type == "mixtral":
        response = run_mixtral(prompt)
    # print("!! LLM Answer\n", response)
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string