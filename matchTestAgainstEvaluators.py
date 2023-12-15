import chess
from ChessEngine import chess_visuals as chess_visuals

from Algorithm.MinMax import play_min_max
from Algorithm.EvaluationFunctions import (relative_random_forest_classifier_evaluator,
                                           relative_simple_material_evaluator, 
                                           relative_random_forest_regression_evaluator,
                                           relative_polynomial_regression_evaluator)

def run_match(evaluator1, evaluator2, num_games=10):
    results = {"evaluator1 wins": 0, "evaluator2 wins": 0, "draws": 0}
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn:
                board.push(play_min_max(board, evaluator1))
            else:
                board.push(play_min_max(board, evaluator2))

        outcome = board.outcome()
        if outcome.winner is None:
            results["draws"] += 1
        elif outcome.winner == chess.WHITE:
            results["evaluator1 wins"] += 1
        else:
            results["evaluator2 wins"] += 1
    return results

def main():
    evaluator_mapping = {
        "relative_polynomial_regression_evaluator": relative_polynomial_regression_evaluator,
        "relative_random_forest_classifier_evaluator": relative_random_forest_classifier_evaluator,
        "relative_random_forest_regression_evaluator": relative_random_forest_regression_evaluator,
        "relative_simple_material_evaluator": relative_simple_material_evaluator
    }

    baseline_evaluator = "relative_simple_material_evaluator"
    test_evaluators = ["relative_polynomial_regression_evaluator", "relative_random_forest_classifier_evaluator", "relative_random_forest_regression_evaluator"]

    for test_evaluator in test_evaluators:
        print(f"\nTesting {test_evaluator} against {baseline_evaluator}")
        match_results = run_match(evaluator_mapping[test_evaluator], evaluator_mapping[baseline_evaluator])
        print("Match Results:", match_results)

if __name__ == "__main__":
    main()
