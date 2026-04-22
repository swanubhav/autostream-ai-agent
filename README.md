1. How to run the project locally:
   A. Clone the repository
   B. Install dependencies:
     pip install -r requirements.txt
   C. Add your API key in .env file
   D. Run:
     python app.py

2. Architecture Explanation:
    I used LangGraph to structure the conversational flow as a state-based system rather than a simple chatbot. 
    The agent maintains state using a memory buffer which stores previous user inputs and extracted details like name, email, and platform.
    For knowledge retrieval, I implemented a simple RAG pipeline using a local JSON file. Before responding, the agent fetches relevant data to ensure accurate answers about pricing and policies.
    Tool execution is handled carefully by checking if all required user details are collected before triggering the mock API.
3. WhatsApp Deployment Question:
    To integrate this agent with WhatsApp, I would use the WhatsApp Business API along with webhooks. Incoming user messages would be received on a webhook endpoint, which would pass the message to the agent backend.
   The agent processes the input, maintains conversation state, and generates a response. This response is then sent back to the user via the WhatsApp API.
