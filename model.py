from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import chainlit as cl

DB_FAISS_PATH = 'vectorstores1/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Based on the symptoms provided, list all possible diseases or conditions that could be associated with these symptoms. For each disease, provide a description, the probability (in percentage), and suggest which type of doctor specialist the user may want to see.

Ensure each disease is unique.

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Cache to store symptoms and their corresponding responses
cache = {}

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True)
    return qa_chain

# Loading the model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.1,
        repetition_penalty=1.2
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    prompt = set_custom_prompt()
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        memory=memory,
        verbose=True
    )
    return qa

def create_action_buttons():
    return [
        cl.Action(name="add_symptom", value="add_symptom", label="Do you experience any other symptoms?"),
        cl.Action(name="suggest_doctors", value="suggest_doctors", label="Shall we suggest doctors?"),
        cl.Action(name="new_conversation", value="new_conversation", label="Start new conversation"),
        cl.Action(name="end_conversation", value="end_conversation", label="Can we end the conversation?")
    ]

# Output function
def final_result(query):
    qa = qa_bot()
    response = qa({"question": query})
    return response['answer']

# List of greeting terms
greeting_terms = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "who are you?", "how are you?"]

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    cl.user_session.set("symptoms", [])
    msg = cl.Message(content="Hi, I'm LIZ, your medical assistant. Shall we start?")
    await msg.send()
    actions = [
        cl.Action(name="yes_start", value="yes_start", label="Yes"),
        cl.Action(name="no_start", value="no_start", label="No")
    ]
    await cl.Message(content="Please select an option:", actions=actions).send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    symptoms = cl.user_session.get("symptoms", [])
    awaiting_postcode = cl.user_session.get("awaiting_postcode", False)
    awaiting_name = cl.user_session.get("awaiting_name", True)
    user_message = message.content.strip()

    # Initial greeting response
    if awaiting_name:
        if user_message.lower() == "yes":
            await cl.Message(content="Great! Please enter your name?").send()
            cl.user_session.set("awaiting_name", False)
            cl.user_session.set("awaiting_postcode", True)
        elif user_message.lower() == "no":
            await cl.Message(content="Alright, feel free to start the conversation whenever you're ready.").send()
            actions = [
                cl.Action(name="start_conversation", value="start_conversation", label="Start the conversation"),
                cl.Action(name="end_conversation", value="end_conversation", label="End the conversation")
            ]
            await cl.Message(content="What would you like to do?", actions=actions).send()
        return

    if awaiting_postcode:
        if cl.user_session.get("name") is None:
            cl.user_session.set("name", user_message)
            await cl.Message(content=f"Nice to meet you, {user_message}! What's your postcode?").send()
        else:
            postcode = user_message
            cl.user_session.set("postcode", postcode)
            cl.user_session.set("awaiting_postcode", False)
            await cl.Message(content=f"Thank you, {cl.user_session.get('name')}. What symptoms are you experiencing?").send()
    else:
        if cl.user_session.get("name") is None:
            cl.user_session.set("name", user_message)
            await cl.Message(content=f"Nice to meet you, {user_message}! What's your postcode?").send()
            cl.user_session.set("awaiting_postcode", True)
        else:
            new_symptom = user_message
            if new_symptom.lower() not in (symptom.lower() for symptom in symptoms):
                symptoms.append(new_symptom)
            cl.user_session.set("symptoms", symptoms)

            # Check cache for existing response
            symptoms_key = ', '.join(symptoms)
            if symptoms_key in cache:
                answer = cache[symptoms_key]
            else:
                query = f"What could be the diseases based on these symptoms: {symptoms_key}? Please provide all possible diseases with a complete sentence."
                res = await chain.acall({"question": query})
                answer = res["answer"]

                # Ensure the answer does not contain repetitions
                answer_lines = answer.split('\n')
                unique_answer_lines = []
                for line in answer_lines:
                    if line not in unique_answer_lines:
                        unique_answer_lines.append(line)
                answer = '\n'.join(unique_answer_lines)

                # Update cache with new response
                cache[symptoms_key] = answer

            # Send the answer directly without any processing
            await cl.Message(content=answer).send()

            actions = [
                cl.Action(name="yes_add_symptom", value="yes_add_symptom", label="Yes"),
                cl.Action(name="no_add_symptom", value="no_add_symptom", label="No")
            ]
            await cl.Message(content="Do you experience any other symptoms?", actions=actions).send()

@cl.action_callback("yes_start")
async def on_yes_start(action):
    await cl.Message(content="Great! So What's your name?").send()
    cl.user_session.set("awaiting_name", False)
    cl.user_session.set("awaiting_postcode", True)

@cl.action_callback("no_start")
async def on_no_start(action):
    await cl.Message(content="Alright, feel free to start the conversation whenever you're ready.").send()
    actions = [
        cl.Action(name="start_conversation", value="start_conversation", label="Start the conversation"),
        cl.Action(name="end_conversation", value="end_conversation", label="End the conversation")
    ]
    await cl.Message(content="What would you like to do?", actions=actions).send()

@cl.action_callback("yes_add_symptom")
async def on_yes_add_symptom(action):
    await cl.Message(content="What other symptom are you experiencing?").send()

@cl.action_callback("no_add_symptom")
async def on_no_add_symptom(action):
    actions = [
        cl.Action(name="suggest_doctors", value="suggest_doctors", label="Shall we suggest doctors?"),
        cl.Action(name="end_conversation", value="end_conversation", label="End the conversation")
    ]
    await cl.Message(content="What would you like to do?", actions=actions).send()

@cl.action_callback("start_conversation")
async def on_start_conversation(action):
    cl.user_session.set("symptoms", [])
    cl.user_session.set("awaiting_name", True)
    await cl.Message(content="Starting a new conversation. Can we start? (yes/no)").send()

@cl.action_callback("add_symptom")
async def on_add_symptom(action):
    await cl.Message(content="What other symptom are you experiencing?").send()

@cl.action_callback("suggest_doctors")
async def on_suggest_doctors(action):
    postcode = cl.user_session.get("postcode", "")
    if postcode:
        await cl.Message(content=f"Here are some doctors near {postcode}:").send()
    else:
        await cl.Message(content="Please provide your postcode first.").send()

@cl.action_callback("end_conversation")
async def on_end_conversation(action):
    await cl.Message(content="Thanks for using our bot! Feel free to ask anything if you need help in the future.").send()
