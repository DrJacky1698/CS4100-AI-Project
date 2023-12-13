import csv
import chess
from Algorithm.EvaluationFunctions import basic_piece_squares_and_material_evaluator, stockfish_evaluator, initializeStockfishEngine, closeStockfishEngine


def process_fen_data(input_csv, output_csv, batch_size=100):
    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header if present
        fen_data = [(row[0], row[1]) for row in reader] # as FEN is second column in our input csv file

    batch = []

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Game Number", "FEN", "Simple Material Evaluator (White)", "Simple Material Evaluator (Black)", 
                         "Stockfish Evaluator (White)", "Stockfish Evaluator (Black)"])

        for game_number, fen in fen_data:
            board = chess.Board(fen)
            evaluations = [game_number, fen]
            
            evaluations.append(basic_piece_squares_and_material_evaluator(board, 'white'))
            # evaluations.append(basic_piece_squares_and_material_evaluator(board, 'black'))
            evaluations.append(stockfish_evaluator(board, 'white'))
            # evaluations.append(stockfish_evaluator(board, 'black'))

            batch.append(evaluations)

            if len(batch) >= batch_size:
                writer.writerows(batch)
                batch = [] # clean the arrary for next 100 lines

        # write any remaining data
        if batch:
            writer.writerows(batch)

initializeStockfishEngine()
process_fen_data('Util/chess_games_fen_output.csv', 'Util/evaluation_results.csv')
closeStockfishEngine()
