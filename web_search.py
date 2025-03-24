# web_search.py
import requests
from dotenv import load_dotenv
import os

load_dotenv()

class WebSearch:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "YOUR_DEFAULT_API_KEY")
        self.cx = os.getenv("CX_KEY", "YOUR_DEFAULT_CX_KEY")
        self.call_count = 0  # Track API usage

    def search(self, query, num_results=3):
        """
        Fetch web search results using Google Custom Search API.
        
        Args:
            query (str): Search query.
            num_results (int): Number of results (max 10).
        
        Returns:
            list: List of strings with web result snippets.
        """
        if self.call_count >= 10:  # Enforce 10-call limit
            print("API call limit reached (10). Skipping web search.")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": min(num_results, 10),
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json().get("items", [])
            self.call_count += 1  # Increment call counter
            return [f"Web: {item['title']} - {item['snippet']} from {item['link']}" for item in results]
        except Exception as e:
            print(f"Web search failed: {str(e)}")
            return []

# if __name__ == "__main__":
#     searcher = WebSearch()
#     query = input("Enter a search query: ")
#     results = searcher.search(query)
#     for result in results:
#         print(result)