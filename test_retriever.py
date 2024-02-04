import unittest

from src.rank_llm.result import Result
from src.rank_llm.retrieve.retriever import Retriever


class TestRetrieverWithResultObjects(unittest.TestCase):
    def setUp(self):
        # create a list of mock Result objects
        self.mock_results = [
            Result(
                query="What is Python?",
                hits=[
                    {"content": "Python is a programming language", "score": 1.0},
                    {"content": "Python is a snake", "score": 0.8},
                ],
            ),
            Result(
                query="What is the University of Waterloo?",
                hits=[
                    {
                        "content": "The University of Waterloo is a university in Waterloo, Canada.",
                        "score": 1.0,
                    },
                    {
                        "content": "The University of Waterloo is a computer science school.",
                        "score": 0.9,
                    },
                ],
            ),
        ]

    def test_retriever_with_result_objects(self):
        # Initialize Retriever with mock results
        retriever = Retriever.from_results(self.mock_results)
        retrieved_results = retriever.retrieve()

        self.assertEqual(len(retrieved_results), 2, "Should return two Result objects")
        self.assertEqual(
            retrieved_results[0].query,
            "What is Python?",
            "The query of the first Result object should match",
        )
        self.assertEqual(
            len(retrieved_results[0].hits),
            2,
            "The first Result object should have two hits",
        )
        self.assertEqual(
            retrieved_results[0].hits[0]["content"],
            "Python is a programming language",
            "The content of the first hit should match",
        )


if __name__ == "__main__":
    unittest.main()
