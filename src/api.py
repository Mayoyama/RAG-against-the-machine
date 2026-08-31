from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .cli import search, answer
from .models import MinimalSource


app = FastAPI()


@app.exception_handler(ValueError)
def val_error_handler(request: Request, exception: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "error": str(exception)
    })


@app.exception_handler(KeyError)
def key_error_handler(request: Request, exception: KeyError) -> JSONResponse:
    return JSONResponse(status_code=500, content={
        "error": f"KeyError: {exception}"
    })


@app.exception_handler(IOError)
def io_error_handler(request: Request, exception: IOError) -> JSONResponse:
    return JSONResponse(status_code=500, content={
        "error": f"IOError: {exception}"
	})


@app.exception_handler(Exception)
def exception_handler(request: Request, exception: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={
        "error": f"Unexpected error: {exception}"
	})


@app.get("/search")
def search_endpoint(query: str, k: int = 5, method: str = "bm25"
                    ) -> list[MinimalSource]:
    """Search the index for a query.
    
    Args:
        query: The search query.
        k: Number of results (default 5).
        directory: The directory to save results in.
        method: "bm25", "semantic", or "hybrid" (default "bm25").
    """
        
    return search(query, k, method=method)


@app.get("/answer")
def answer_endpoint(query: str, k: int) -> dict[str, str]:
    """Answer a query using retrieved context.
    
    Args:
        query: The question to answer.
        k: Number of sources to retrieve (default 5).
    """
    
    ans = answer(query, k)
    return {"question": query, "answer": ans}
