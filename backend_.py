from typing import Annotated, List, TypedDict
import operator

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# if u use your api key of openai and claude code
# then u can change the model here and enter the api key in .env 
# and import your langchain model using import langchain_OpenAi etc
model = ChatOllama(model="qwen3:8b")


class ModelOutput(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(ModelOutput)


class ReviewState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    average_score: float


def language_review(state: ReviewState):
    prompt = (
        "Evaluate language quality of this essay. "
        "Give concise feedback and a score out of 10.\n\n"
        f"Essay:\n{state['essay']}"
    )
    output = structured_model.invoke(prompt)
    return {
        "language_feedback": output.feedback,
        "individual_scores": [output.score],
    }


def analysis_review(state: ReviewState):
    prompt = (
        "Evaluate depth of analysis in this essay. "
        "Give concise feedback and a score out of 10.\n\n"
        f"Essay:\n{state['essay']}"
    )
    output = structured_model.invoke(prompt)
    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score],
    }


def thought_clarity_review(state: ReviewState):
    prompt = (
        "Evaluate clarity of thought in this essay. "
        "Give concise feedback and a score out of 10.\n\n"
        f"Essay:\n{state['essay']}"
    )
    output = structured_model.invoke(prompt)
    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score],
    }


def final_evaluation(state: ReviewState):
    summary_prompt = (
        "Create an overall review summary from these points:\n"
        f"1) Language: {state.get('language_feedback', '')}\n"
        f"2) Analysis: {state.get('analysis_feedback', '')}\n"
        f"3) Thought Clarity: {state.get('clarity_feedback', '')}"
    )
    summary = model.invoke(summary_prompt).content

    scores = state.get("individual_scores", [])
    average = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {"overall_feedback": summary, "average_score": average}


graph = StateGraph(ReviewState)
graph.add_node("language_review", language_review)
graph.add_node("analysis_review", analysis_review)
graph.add_node("thought_clarity_review", thought_clarity_review)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "language_review")
graph.add_edge(START, "analysis_review")
graph.add_edge(START, "thought_clarity_review")

graph.add_edge("language_review", "final_evaluation")
graph.add_edge("analysis_review", "final_evaluation")
graph.add_edge("thought_clarity_review", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()


def run_review(essay: str) -> dict:
    result = workflow.invoke({"essay": essay})
    return {
        "language_feedback": result.get("language_feedback", ""),
        "analysis_feedback": result.get("analysis_feedback", ""),
        "clarity_feedback": result.get("clarity_feedback", ""),
        "overall_feedback": result.get("overall_feedback", ""),
        "average_score": result.get("average_score", 0.0),
    }




