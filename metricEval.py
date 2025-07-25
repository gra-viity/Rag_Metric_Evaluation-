# python3
# -*- coding: utf-8 -*-

import numpy as np
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings  # Using langchain-aws
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Configuration
config = {
    "credentials_profile_name": "default",  # E.g "default"
    "region_name": "us-east-2",  # E.g. "us-east-1"
    "llm": "us.anthropic.claude-3-haiku-20240307-v1:0",  # E.g "anthropic.claude-3-5-sonnet-20241022-v2:0" ----(Inference profile ID)(https://us-east-2.console.aws.amazon.com/bedrock/home?region=us-east-2#/inference-profiles)
    "embeddings": "amazon.titan-embed-text-v2:0",  # E.g "amazon.titan-embed-text-v2:0"
    "temperature": 0.4,
}

# Sample Documents
# sample_docs = [
#     "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
#     "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
#     "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
#     "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
#     "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
# ]

# Sample Queries and Expected Responses
# sample_queries = [
#     "Who introduced the theory of relativity?",
#     "Who was the first computer programmer?",
#     "What did Isaac Newton contribute to science?",
#     "Who won two Nobel Prizes for research on radioactivity?",
#     "What is the theory of evolution by natural selection?"
# ]

# expected_responses = [
#     "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
#     "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
#     "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
#     "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
#     "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
# ]

# Article on Quantum Computing and Renewable Energy
sample_docs = [
    "Quantum computing continues to be a frontier of rapid innovation, promising to revolutionize various fields from medicine to materials science. Recent breakthroughs have focused on improving qubit stability and coherence times, crucial factors for building practical quantum computers. Researchers at IBM recently announced a significant reduction in error rates for their quantum processors, a step forward in overcoming one of the biggest challenges in the field. Simultaneously, companies like Google and IonQ are exploring new qubit architectures, including superconducting and trapped-ion systems, each with unique advantages and disadvantages. The long-term goal remains achieving fault-tolerant quantum computation, which would unlock the full potential of these machines for complex problems currently intractable for classical computers. Investment in quantum technologies is surging globally, with governments and private firms pouring billions into research and development.",
    "The transition to renewable energy sources is accelerating worldwide, driven by concerns over climate change and the volatile nature of fossil fuel markets. Solar photovoltaic (PV) and wind power remain at the forefront of this shift, with capacity additions reaching record highs in 2023. Countries like China and the United States are leading in solar installations, while Europe continues to expand its offshore wind farms. Energy storage solutions, particularly large-scale battery systems, are becoming increasingly vital to ensure grid stability as the share of intermittent renewables grows. While challenges such as grid modernization and securing critical raw materials persist, policy support, technological advancements, and falling costs are making renewables increasingly competitive. The International Energy Agency (IEA) projects that renewable energy will account for over 90% of global electricity expansion in the coming years, signaling a profound transformation of the energy landscape."

]

sample_queries=[
    "What are the primary factors that researchers are focusing on to improve quantum computers, according to the article?",
    "Which companies are mentioned as exploring new qubit architectures, and what types of systems are they investigating?",
    "What is the long-term goal of quantum computation as stated in the article?",
    "What recent breakthrough did IBM announce regarding their quantum processors?",
    "What is the overall trend in investment in quantum technologies globally?",
    "What are the two main drivers behind the acceleration of renewable energy adoption worldwide?",
    "Which renewable energy sources are highlighted as being at the forefront of this shift?",
    "What role do large-scale battery systems play in the context of renewable energy, according to the article?",
    "According to the International Energy Agency (IEA) projection, what percentage of global electricity expansion will renewable energy account for in the coming years?",
    "What are some of the persistent challenges mentioned in the article regarding the adoption of renewables?",
    "What is the exact number of qubits in IBM's latest quantum processor?",
    "What is the capital cost of installing a solar farm in China?",
    "When was the International Energy Agency (IEA) founded?",
    "Which country has the most offshore wind farms?",
    "Does the article provide a definition of 'qubit coherence'?"

]

expected_responses = [
    "Researchers are focusing on improving qubit stability and coherence times, which are crucial for building practical quantum computers.",
    "The companies mentioned are IBM, Google, and IonQ. They are investigating superconducting and trapped-ion systems as new qubit architectures.",
    "The long-term goal of quantum computation is to solve complex problems that are currently intractable for classical computers.",
    "IBM announced a breakthrough in their quantum processors by increasing the number of qubits and improving error correction techniques.",
    "The overall trend in investment in quantum technologies globally is increasing, with significant funding from both public and private sectors.",
    "Concerns over climate change and the volatile nature of fossil fuel markets are the two main drivers.",
    "Solar photovoltaic (PV) and wind power are highlighted.",
    "Large-scale battery systems play a crucial role in the context of renewable energy by providing storage solutions that help balance supply and demand.",
    "According to the International Energy Agency (IEA) projection, renewable energy will account for a significant portion of global electricity expansion in the coming years.",
    "Challenges include grid modernization and securing critical raw materials.",
    "The information is not available in the provided text.",
    "The article does not specify that information.",
    "I cannot answer that question based on the given articles.",
    "No, the article mentions \"qubit coherence\" but does not provide a definition."
]



class RAG:
    def __init__(self, config):
        self.llm = ChatBedrockConverse(
            credentials_profile_name=config["credentials_profile_name"],
            region_name=config["region_name"],
            model=config["llm"],
            temperature=config["temperature"],
        )
        self.llm_model_id = config["llm"]  # Store the model ID
        self.embeddings = BedrockEmbeddings(  # Using langchain-aws
            credentials_profile_name=config["credentials_profile_name"],
            region_name=config["region_name"],
            model_id=config["embeddings"],
        )
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")

        query_embedding = self.embeddings.embed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def generate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"""You are a knowledgeable research assistant. Answer the question below using only the information provided in the document. Be concise and avoid repetition. If the document does not contain the answer, state that you cannot answer the question.

        Question: {query}

        Document: {relevant_doc}

        Answer:"""  # Improved prompt
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content, self.llm_model_id  # Return the answer and model ID

# Initialize RAG instance
rag = RAG(config=config)
rag.load_documents(sample_docs)

# Generate Dataset for Evaluation
dataset = []
for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = rag.get_most_relevant_docs(query)
    response, model_id = rag.generate_answer(query, relevant_docs)  # Capture the model ID
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": reference,
            "model_id": model_id  # Store the model ID in the dataset
        }
    )
    print(f"Query: {query}")
    print(f"Answer: {response}")
    print(f"LLM Used: {model_id}")
    print("-" * 40)

# Ragas Evaluation
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Initialize Langchain LLM and Embeddings Wrappers for Ragas
bedrock_llm = ChatBedrockConverse(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    model=config["llm"],
    temperature=config["temperature"],
)

bedrock_embeddings = BedrockEmbeddings(  # Using langchain-aws
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    model_id=config["embeddings"],
)

evaluator_llm = LangchainLLMWrapper(bedrock_llm)
embedding_model = LangchainEmbeddingsWrapper(bedrock_embeddings)

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)

# Evaluate
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
    embeddings=embedding_model
)

print(result)
