import csv
import chess
from Algorithm.EvaluationFunctions import relative_simple_material_evaluator, stockfish_evaluator, initializeStockfishEngine, closeStockfishEngine, relative_tapered_piece_squares_evaluator, relative_king_safety_evaluator, relative_pieces_attacked_evaluator, relative_simple_piece_count_evaluator


def process_fen_data(input_csv, output_csv, batch_size=100):
    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header if present
        fen_data = [(row[0], row[1]) for row in reader] # as FEN is second column in our input csv file

    batch = []
    processed_games = 0

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game Number", "FEN",
                         "Relative_simple_material_evaluator (White)", 
                         "Relative_tapered_piece_squares_evaluator (White)",
                         "Relative_king_safety_evaluator (White)",
                         "Relative_simple_piece_count_evaluator (White)",
                         "Relative_pieces_attacked_evaluator (White)",
                         "Stockfish Evaluator (White)"])

        for game_number, fen in fen_data:
            board = chess.Board(fen)
            evaluations = [game_number, fen]
            
            evaluations.append(relative_simple_material_evaluator(board, 'white'))
            evaluations.append(relative_tapered_piece_squares_evaluator(board, 'white'))
            evaluations.append(relative_king_safety_evaluator(board, 'white'))
            evaluations.append(relative_simple_piece_count_evaluator(board, 'white'))
            evaluations.append(relative_pieces_attacked_evaluator(board, 'white'))
            evaluations.append(stockfish_evaluator(board, 'white'))

            batch.append(evaluations)
            processed_games += 1

            if processed_games % batch_size == 0 or processed_games == len(fen_data):
                print(f"Processed {processed_games} games so far.")

            if len(batch) >= batch_size:
                writer.writerows(batch)
                batch = [] # clean the arrary for next 100 lines

        # write any remaining data
        if batch:
            writer.writerows(batch)

initializeStockfishEngine()
process_fen_data('Util/chess_games_fen_output.csv', 'Util/evaluation_results.csv')
closeStockfishEngine()
