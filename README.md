# chatbot_public
Chatbot Version 2
This GitHub repository contains the code for my personal chatbot project, rebuilt using the robust capabilities of the LangChain framework.

Background
I initially created a basic chatbot (v0.1) as a coding novice by following some tutorials. However, it had severe limitations around chunking, embeddings, databases, metadata and memory. Eager to improve, I struggled through more advanced AI courses without much success.

Eventually LangChain's modular pipelines helped me reconstruct a more capable chatbot (v2) from the ground up. This repo chronicles that rebuilding journey using LangChain’s tools for:

Ingesting blog data
Embedding text
Indexing vectors
Retrieving context
Architecting conversational agent logic
While still a work-in-progress, v2 shows significantly more intelligence in query responses while maintaining dialogue context.

Key Components
DataIngestionAndIndexing.ipynb: Jupyter notebook detailing the process of ingesting data from a source and indexing it for the chatbot.
xml_to_json_html.py: Python script used to transform WordPress exports into structured HTML and JSON, suitable for ingestion by the chatbot.
all_posts.json: The output JSON file containing the processed blog posts ready for use by the chatbot.
example_app.py: The full example application showcasing the chatbot in action.