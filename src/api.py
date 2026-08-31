from fastapi import FastAPI
from .cli import search, answer
from .models import MinimalSource


app = FastAPI()


@app.get("/search")
def search_endpoint(query: str, k: int = 5,
                    save_directory: str = "data/processed"
                    ) -> list[MinimalSource]:
    return search(query, k, save_directory)


@app.get("/answer")
def answer_endpoint(query: str, k: int,
                    save_directory: str = "data/processed"
                    ) -> dict[str, str]:
    ans = answer(query, k, save_directory)
    return {"question": query, "answer": ans}
