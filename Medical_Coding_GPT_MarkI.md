# PACE Coding GPT Mark I

Created: August 9, 2023 4:22 PM

# Operative Note Generation

```python
import warnings
warnings.filterwarnings("ignore")

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import torch
#from instruct_pipeline import InstructionTextGenerationPipeline LlamaForCausalLM, LlamaTokenizer,
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModel, AutoConfig, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
#from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, PyPDFDirectoryLoader, BSHTMLLoader, TextLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain, SequentialChain, create_extraction_chain
from langchain.llms import CTransformers
from langchain.output_parsers import CommaSeparatedListOutputParser
import re
import os
import gc
```

## Import Data

### Load PDF

```python
loader = DirectoryLoader('code_docs', glob="**/*.txt", loader_cls=TextLoader)
data_raw = loader.load()
```

### Chunk Text

```python
text_splitter = RecursiveCharacterTextSplitter(separators = ["\\n"], chunk_size = 150, chunk_overlap=0)
texts = text_splitter.split_documents(data_raw)
for text in texts:
    text.page_content = re.compile(r'\\n').sub('',text.page_content)
    text.page_content.strip()
    text.page_content = re.compile(r'""').sub('"',text.page_content)
print (f'{len(texts)} document(s) in your data')
```

### Create Embeddings

```python
embedding_function = SentenceTransformerEmbeddings(model_name = 'sentence-transformers_all-MiniLM-L6-v2', model_kwargs = {'device': 'cpu'})
```

## Load Embedding Vectors into FAISS

### Store and Save on Disk

```python
db = FAISS.from_documents(texts, embedding_function1)
db.save_local("~/faiss_index")
```

```python
db = FAISS.from_documents(texts, embedding_function2)
db.save_local("~/faiss_index_gt")
```

### Load Vector Database from Disk

```python
db1 = FAISS.load_local("faiss_index", embedding_function)
```

## Instantiate LLM

```python
#config = AutoConfig.from_pretrained(mpath) load_in_8bit=True
max_memory_mapping = {0: "5GB", 1: "10GB"}
tokenizer = LlamaTokenizer.from_pretrained('Llama-2-7b')
model = LlamaForCausalLM.from_pretrained('Llama-2-7b', torch_dtype=torch.float16, device_map='auto', max_memory=max_memory_mapping)
```

```python
lp = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 500,
    temperature = 0,
    top_p=0.90,
    repetition_penalty=1.1,
    device_map='auto'
    )
llm = HuggingFacePipeline(pipeline=lp)
```

## Load Operative Report

```python
opnote = """Operative Note

SURGERY DATE:  DEIDENTIFIED PATIENT 4

PRE-OP DIAGNOSIS: Dehiscence of fascia [T81.30XA]

POST-OP DIAGNOSIS: Post-Op Diagnosis Codes:
Dehiscence of fascia [T81.30XA]

PROCEDURES PERFORMED:
1) reopening of recent laparotomy
2) placement of 12" x 12" Vicryl (absorbable synthetic) mesh in intraperitoneal sublay position
3) primary fascial reclosure with interrupted suture
4) placement of Prevena vacuum-assisted wound management device

SURGEON: Surgeon(s) and Role:
DEIDENTIFIED

ASSISTANT(S): None

STAFF:
DEIDENTIFIED

ANESTHESIA: General

INDICATION(S): DEIDENTIFIED one week s/p abdominal closure following small bowel resection now with copious salmon-colored midline wound drainage

OPERATIVE FINDINGS:
1) small subcentimeter fascial defect just below knot with ongoing leakage responsible for clinical presentation
2) copious bland ascites; no evidence of bowel ischemia or anastomotic dehiscence
3) congested but viable-appearing liver
4) omental adhesions to small bowel, left in situ

OPERATIVE REPORT:

Written informed consent was garnered and the patient was taken to the operating room.  There was a gentle induction of general anesthesia with endotracheal intubation by the anesthesia provider team.  The abdomen was exposed, the staples were removed, and then the wound and skin prepped and draped in the typical sterile fashion using water-soluble prep, i.e. Hibiclens.  2 g of Ancef was provided for SSI prophylaxis and a timeout for safety was called.

We reopen the prior incision and performed a careful inspection of the fascia.  There was a roughly 5 x 5 mm defect just below the knot in the fascia that was draining thin ascites.  We could find no other palpable nor visible defect.  The knot from the prior closure was still intact.  We then removed the fascial sutures and disposed of them.  We reinspected the small bowel, and found some omental adhesions which were left alone.  All inspected small bowel appeared viable, and there is no evidence of anastomotic breakdown.  We encountered substantial amount of bland ascites but no succus nor pus.  The liver appeared congested but viable, and there was no other frank abdominal pathology.  As such, we elected to trim a 12 x 12 inch Vicryl mesh to fit the defect in the fascia, and used circumferential 0 Vicryl interrupted sutures to carefully oppose this mesh to the anterior abdominal wall at roughly 4 cm intervals.  This left a synthetic, absorbable Vicryl mesh in the intraperitoneal subway position.  Satisfied with this mesh placement, we then performed a tension-free primary closure of the fascia, which was found to be viable.  We used interrupted #1 Maxon sutures in a figure-of-eight fashion to reapproximate the edges of the fascia.  The skin was closed with staples, and a Prevena incisional vacuum device was placed.  The patient was awoken from anesthesia and transferred back to the ICU in hemodynamically stable but mechanically ventilated condition.  All instrument, needle, and sponge counts were correct at the conclusion of the procedure.  Dr. DEIDENTIFIED was present for the entire procedure.

ESTIMATED BLOOD LOSS:

50 mL DEIDENTIFIED

TOTAL IV FLUIDS: 1L crystalloid; 500 mL albumin

SPECIMENS:
No specimens in log

IMPLANTS:
DEIDENTIFIED

COMPLICATIONS: None immediate

DISPOSITION: ICU - intubated and hemodynamically stable.
"""
```

## Prompt Chain

```python
proc_template ="""[INST] <<SYS>>
You are an expert surgical assistance program for task completion. Complete the extraction task with the provided context. Do not provide additional help or information. Do not make up information.
<</SYS>>

Task: Extract the surgical procedures performed from the provided context. Output a bulleted list.

Context: {report}

Output:
[/INST]"""

proc_prompt = PromptTemplate(
    template=proc_template, input_variables=["report"]
)
proc_info_chain = LLMChain(llm=llm, prompt=proc_prompt, output_key="proc_info")
```

```python
proc_info = proc_info_chain({"report": opnote.lower()})
gc.collect()
torch.cuda.empty_cache()
print(proc_info['proc_info'])
```

```
• Reopening of recent laparotomy
• Placement of 12” x 12” Vicryl (absorbable synthetic) mesh in intraperitoneal sublay position
• Primary fascial reclosure with interrupted suture
• Placement of Prevena vacuum-assisted wound management device
```

```python
dx_template = """[INST] <<SYS>>
You are an expert surgical assistance program for task completion. Complete the extraction task with the provided context. Do not provide additional help or information. Do not make up information.
<</SYS>>

Task: Extract the surgical diagnoses from the provided context. Output a bulleted list.

Context: {report}

Output:
[/INST]"""

dx_prompt = PromptTemplate(
    template=dx_template, input_variables=["report"]
)
dx_info_chain = LLMChain(llm=llm, prompt=dx_prompt, output_key="dx_info")
```

```python
dx_info = dx_info_chain({"report": opnote})
gc.collect()
torch.cuda.empty_cache()
print(dx_info['dx_info'])
```

```
• Dehiscence of fascia (T81.30XA)
• Dehiscence of fascia (T81.30XA)
• Omental adhesions (unspecified)
```

```python
key_template = """[INST] <<SYS>>
You are an expert surgical assistance program for task completion. Complete the extraction task with the provided context. Do not provide additional help or information. Do not make up information.
<</SYS>>

Task: Extract the key details regarding the operation from the provided context. Output a bulleted list.

Context: {report}

Output:
[/INST]"""

key_prompt = PromptTemplate(
    template=key_template, input_variables=["report"]
)
key_info_chain = LLMChain(llm=llm, prompt=key_prompt, output_key="key_info")

```

```python
key_info = key_info_chain({"report": opnote})
gc.collect()
torch.cuda.empty_cache()
print(key_info['key_info'])

```

```
• Dehiscence of fascia (T81.30XA)
• Reopening of recent laparotomy
• Placement of 12” x 12” Vicryl (absorbable synthetic) mesh in intraperitoneal sublay position
• Primary fascial reclosure with interrupted suture
• Placement of Prevena vacuum-assisted wound management device
• Small subcentimeter fascial defect just below knot with ongoing leakage responsible for clinical presentation
• Copious bland ascites; no evidence of bowel ischemia or anastomotic dehiscence
• Congested but viable-appearing liver
• Omental adhesions to small bowel, left in situ
```

```python
def get_info(inputs: dict) -> dict:
    text = inputs["key_info"]
    x = text.strip().split('• ')
    x = list(filter(None, x))
    details=[]
    for i in range(len(x)):
        x[i] = re.compile(r'\\n').sub('',x[i])
        info = db2.similarity_search(x[i],2)
        details = details + [(doc.page_content+'.\\n') for doc in info]
    detail_text = ''.join(details)
    return {"other_details": detail_text}

def get_proc(inputs: dict) -> dict:
    text = inputs["proc_info"]
    x = text.strip().split('• ')
    x = list(filter(None, x))
    details=[]
    for i in range(len(x)):
        x[i] = re.compile(r'\\n').sub('',x[i])
        info = db2.similarity_search(x[i],2,filter={"source": "code_docs/CPT_Billing_Codes.txt", "source": "code_docs/CPT_Procedure_Codes.txt"})
        details = details + [(doc.page_content+'.\\n') for doc in info]
    procdocs = ''.join(details)
    return {"proc": procdocs}

def get_dx(inputs: dict) -> dict:
    text = inputs["dx_info"]
    x = text.strip().split('• ')
    x = list(filter(None, x))
    details=[]
    for i in range(len(x)):
        x[i] = re.compile(r'\\n').sub('',x[i])
        info = db1.similarity_search(text, 2, filter={"source": "code_docs/ICD10_Diagnosis_Codes.txt"})
        details = details + [(doc.page_content+'.\\n') for doc in info]
    dxdocs = ''.join(details)
    return {"dx": dxdocs}
```

```python
info_chain = TransformChain(
    input_variables=["key_info"], output_variables=["other_details"], transform=get_info
)
```

```python
proc_chain = TransformChain(
    input_variables=["proc_info"], output_variables=["proc"], transform=get_proc
)
```

```python
dx_chain = TransformChain(
    input_variables=["dx_info"], output_variables=["dx"], transform=get_dx
)
```

```python
final = """[INST] <<SYS>>
You are an expert medical coding assistance program for task completion. Complete the task with the provided context. Do not make up information.
<</SYS>>

Task: Using the provided information about the procedures, diagnoses, other details, and possible codes to suggest a list of diagnosis codes and CPT codes. Output a bulleted list.

Procedures:
{proc_info}

Diagnoses:
{dx_info}

Other Details:
{key_info}

Possible Codes:
{proc}
{dx}
{other_details}

Output:
[/INST]"""

final_prompt = PromptTemplate(
    template=final, input_variables=["proc_info","dx_info","key_info", "proc", "dx", "other_details"]
)
cpt_chain = LLMChain(llm=llm, prompt=final_prompt, output_key="cpt_list")

```

```python
overall_chain = SequentialChain(
    chains=[proc_info_chain, proc_chain, dx_info_chain, dx_chain, key_info_chain, info_chain, cpt_chain],
    input_variables=["report"],
    # Here we return multiple variables
    output_variables=["key_info", "proc_info", "dx_info", "cpt_list"],
    verbose=True)

```

## Generate Codes for OP Report

```python
result = overall_chain({"report": opnote})
gc.collect()
torch.cuda.empty_cache()
```

```
[1m> Entering new SequentialChain chain...[0m
[1m> Finished chain.[0m
```

```python
print(result["cpt_list"])
```

```
Based on the information provided, here are some potential diagnosis and procedure codes that may be relevant for this case:
Diagnosis Codes:
* T81.30XA: Dehiscence of fascia, initial encounter
* T81.31XA: Superficial dehiscence of operation wound, initial encounter
* T81.30XD: Wound dehiscence, surgical, subsequent encounter

Procedure Codes:
* CPT 49002: Reopening of recent laparotomy
* CPT 49900: Suture, secondary, of abdominal wall for evisceration or dehiscence
* CPT 49568: Implantation of mesh or other prosthesis for open incisional or ventral hernia repair or mesh for closure of debridement for necrotizing soft tissue infection
* CPT 49570: Repair of small omphalocele, with primary closure
* CPT 49600: Repair of epigastric hernia (eg, preperitoneal fat); incarcerated or strangulated
* CPT 49572: Closure of enterostomy, large or small intestine; with resection and colorectal anastomosis (eg, closure of Hartmann type procedure)
* CPT 44005: Freeing bowel adhesion, enterolysis
* CPT 44180: Laparoscopy, surgical; enterolysis (freeing of intestinal adhesion) (separate procedure)

It's important to note that these codes are based on the information provided in the scenario, and the actual codes used may vary depending on the specific details of the case. Additionally, it's important to consult with a qualified medical professional for accurate coding and billing information.

```