import json

class RAG:
    def __init__(self):
        with open("knowledge_base.json") as f:
            self.data = json.load(f)

    def retrieve(self, query):
        query = query.lower()

        if "price" in query or "plan" in query:
            return f"""
            Basic Plan: {self.data['pricing']['basic']}
            Pro Plan: {self.data['pricing']['pro']}
            """

        if "refund" in query or "policy" in query:
            return f"""
            Refund: {self.data['policies']['refund']}
            Support: {self.data['policies']['support']}
            """

        return "No relevant info found."